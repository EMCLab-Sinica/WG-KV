import os
import json
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import subprocess
import wandb
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, get_wsd_schedule
from datasets import load_dataset, concatenate_datasets
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from einops import rearrange
from transformers.modeling_layers import set_duo_attn_alpha

try:
    from third_party.duo_attn.duo_attn.utils import load_attn_pattern, sparsify_attention_heads
except:
    pass


def get_config():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--load_pretrained_weights', type=str)
    parser.add_argument('--load_duo_attn_pattern', type=str)
    parser.add_argument('--duo_attn_sparsity', type=float)

    parser.add_argument('--fineweb_config', nargs='+', type=str, default=["4-8k:4096:4000", "8-16k:8192:2000", "16-32k:16384:1000", "32-64k:32768:500"])
    parser.add_argument('--fineweb_config_val', nargs='+', type=str, default=["4-8k:4096:400", "8-16k:8192:200", "16-32k:16384:100", "32-64k:32768:50"])
    parser.add_argument('--nemotron_config', nargs='+', type=str, default=None)
    parser.add_argument('--nemotron_config_val', nargs='+', type=str, default=None)

    # Recommended hyperparameters for llama-3.2-1b
    # lr = 8e-4 ~ 2e-3
    # lr = 0.2 (for use_duo_attn)
    # weight_decay = 0 (for use_duo_attn)

    parser.add_argument('--global_batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lambda_reg', type=float, default=4e-2)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--decay_ratio', type=float, default=0.9)

    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--group_name', type=str, default="0000-default")
    parser.add_argument('--save_every', type=int, default=1000)

    parser.add_argument('--checkpointing_threshold', type=int)
    parser.add_argument('--local_window_size', type=int)

    parser.add_argument('--g_inputs', nargs='+', type=str, choices=["pre_k", "post_k", "v"])
    parser.add_argument('--g_act_fn', type=str, choices=["silu", "gelu"])
    parser.add_argument('--g_expand', type=float)
    parser.add_argument('--g_epsilon', type=float)
    parser.add_argument('--g_rms', type=int)
    parser.add_argument('--g_threshold', type=float)
    parser.add_argument('--g_ungate_count', type=int)

    parser.add_argument('--use_duo_attn', action='store_true', default=None)
    parser.add_argument('--duo_attn_sink_size', type=int)

    config = parser.parse_args()
    config.device = torch.device(config.device)

    if config.load_duo_attn_pattern:
        _, sink_size, recent_size = load_attn_pattern(config.load_duo_attn_pattern)
        config.use_duo_attn = True
        config.duo_attn_sink_size = sink_size
        config.local_window_size = recent_size

    return config


def create_exp_name(config):
    current_time = datetime.now().strftime('%b%d_%H-%M')
    exp_name = (
        (f"{current_time}") +
        (f"_gbs{config.global_batch_size}") +
        (f"_lr{config.lr}") +
        (f"_wd{config.weight_decay}") +
        (f"_lambda{config.lambda_reg}") +
        (f"_warmup{config.warmup_ratio}") +
        (f"_decay{config.decay_ratio}")
    )

    exp_name += (
        (f"_window{config.local_window_size}" if config.local_window_size is not None else "") +
        (f"_inputs-{'-'.join(config.g_inputs)}" if config.g_inputs is not None else "") +
        (f"_act-{config.g_act_fn}" if config.g_act_fn is not None else "") +
        (f"_expand{config.g_expand}" if config.g_expand is not None else "") +
        (f"_eps{config.g_epsilon}" if config.g_epsilon is not None else "") +
        (f"_rms{config.g_rms}" if config.g_rms is not None else "") +
        (f"_thres{config.g_threshold}" if config.g_threshold is not None else "") +
        (f"_ungate{config.g_ungate_count}" if config.g_ungate_count is not None else "")
    )

    if config.use_duo_attn is not None:
        exp_name += (
            (f"_duo") +
            (f"_sink{config.duo_attn_sink_size}" if config.duo_attn_sink_size is not None else "")
        )

    git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    exp_name += f"_git-{git_hash}"

    return exp_name


def save_param_names(output_dir, filename, param_names):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write("\n".join(param_names))


def _process_dataset_from_config(dataset, bins, config_strs, select_from_end):
    bins_dataset = load_dataset(bins, split="train")
    category_indices = bins_dataset[0]

    sub_datasets = []
    seq_lens = []

    for config_str in config_strs:
        category, max_seq_len_str, num_samples_str = config_str.split(':')
        max_seq_len = int(max_seq_len_str)
        num_samples = int(num_samples_str)

        available_indices = category_indices[category]
        if select_from_end:
            indices = available_indices[-num_samples:]
        else:
            indices = available_indices[:num_samples]
        sub_dataset = dataset.select(indices)
        
        sub_datasets.append(sub_dataset)
        seq_lens.extend([max_seq_len] * len(sub_dataset))
    
    dataset = concatenate_datasets(sub_datasets)
    dataset = dataset.add_column("max_seq_len", seq_lens)

    return dataset


def _process_dataset_from_split_config(dataset, config_strs, select_from_end):
    sub_datasets = []
    seq_lens = []

    for config_str in config_strs:
        category, max_seq_len_str, num_samples_str = config_str.split(':')
        max_seq_len = int(max_seq_len_str)
        num_samples = int(num_samples_str)

        sub_dataset = dataset[category]
        if select_from_end:
            indices = list(range(len(sub_dataset) - num_samples, len(sub_dataset)))
        else:
            indices = list(range(num_samples))
        sub_dataset = sub_dataset.select(indices)

        sub_datasets.append(sub_dataset)
        seq_lens.extend([max_seq_len] * len(sub_dataset))

    dataset = concatenate_datasets(sub_datasets)
    dataset = dataset.add_column("max_seq_len", seq_lens)

    return dataset


def _normalize_messages_dataset(dataset, to_messages_fn):
    mapped_dataset = dataset.map(to_messages_fn)
    return mapped_dataset.select_columns(["messages_norm", "max_seq_len"])


def _build_fineweb_messages_for_map(example):
    return {
        "messages_norm": [
            {"role": "user", "content": f"Please analyze and summarize the following text:\n\n{example['text']}"},
        ]
    }


def _build_nemotron_messages_for_map(example):
    messages = example["messages"]
    assert isinstance(messages, list) and len(messages) == 2

    user_msg = messages[0]
    assistant_msg = messages[1]
    assert isinstance(user_msg, dict)
    assert isinstance(assistant_msg, dict)

    assert user_msg["role"] == "user"
    assert isinstance(user_msg["content"], str)
    assert user_msg["reasoning_content"] is None

    assert assistant_msg["role"] == "assistant"
    assert isinstance(assistant_msg["content"], str)
    assert isinstance(assistant_msg["reasoning_content"], str)

    assistant_content = f"<think>{assistant_msg['reasoning_content']}</think>{assistant_msg['content']}"
    return {
        "messages_norm": [
            {"role": user_msg["role"], "content": user_msg["content"]},
            {"role": assistant_msg["role"], "content": assistant_content},
        ]
    }


def preprocess_data(dataset, config, tokenizer):
    def data_generator():
        def _create_and_yield_batch(batch_messages, max_seq_len):
            input_ids = tokenizer.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                return_tensors="pt",
                max_length=max_seq_len,
                truncation=True,
                padding='do_not_pad'
            )
            assert input_ids.size(1) == max_seq_len
            return input_ids, max_seq_len

        buffers = {}
        effective_batch_sizes = {}

        for example in dataset:
            messages = example["messages_norm"]
            max_seq_len = example['max_seq_len']

            if max_seq_len not in buffers:
                buffers[max_seq_len] = []
                effective_batch_sizes[max_seq_len] = (config.global_batch_size + max_seq_len - 1) // max_seq_len

            buffers[max_seq_len].append(messages)

            # If a buffer is full, yield a batch
            if len(buffers[max_seq_len]) == effective_batch_sizes[max_seq_len]:
                yield _create_and_yield_batch(buffers[max_seq_len], max_seq_len)
                buffers[max_seq_len] = []

        # Yield any remaining samples in the buffers
        for max_seq_len, batch_messages in buffers.items():
            if batch_messages:
                yield _create_and_yield_batch(batch_messages, max_seq_len)
    
    return len(dataset), data_generator


def create_regularization_loss_fn(local_window_size, model, use_duo_attn):
    def regularization_loss_fn_duo(_):
        duo_attn_alphas = [
            torch.sigmoid(layer.self_attn.duo_attn_alpha)
            for layer in model.model.layers
        ]
        return torch.stack(duo_attn_alphas).mean()

    def regularization_loss_fn(all_g_scores):
        _, _, _, kv_seq_len = all_g_scores.shape
        s_idx = torch.arange(0, kv_seq_len, dtype=torch.int32, device=all_g_scores.device)
        offset = torch.clamp(local_window_size / (kv_seq_len - s_idx), max=1.0) \
                      .reshape(1, 1, 1, kv_seq_len) \
                      .to(all_g_scores.dtype)
        scaling = 1 - offset
        sparsified_g_scores = -((all_g_scores - 1) ** 2) + 1
        return torch.mean(offset + scaling * sparsified_g_scores)

    if use_duo_attn:
        return regularization_loss_fn_duo
    else:
        return regularization_loss_fn


def causal_mask_mod(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def set_hard_threshold(model, value):
    for layer in model.model.layers:
        layer.self_attn.use_hard_threshold = value


def compute_loss_and_backward(
    distillation_loss_fn,
    regularization_loss_fn,
    total_loss_fn,
    hidden_states_teacher,
    hidden_states_student,
    all_g_scores,
):
    assert hidden_states_teacher.size() == hidden_states_student.size()
    assert hidden_states_teacher.requires_grad == False
    assert hidden_states_student.requires_grad == True
    assert all(g_scores.requires_grad for g_scores in all_g_scores)

    all_g_scores = torch.stack(all_g_scores, dim=0)

    distillation_loss = distillation_loss_fn(hidden_states_student, hidden_states_teacher)
    regularization_loss = regularization_loss_fn(all_g_scores)
    total_loss = total_loss_fn(distillation_loss, regularization_loss)

    total_loss.backward()

    return total_loss.item(), distillation_loss.item(), regularization_loss.item(), all_g_scores.detach()


def save_model(model, config, name):
    base_path = os.path.join(config.output_dir, "weights", name)
    os.makedirs(base_path, exist_ok=True)

    if config.use_duo_attn:
        duo_attn_dir = os.path.join(base_path, "duo_attn_pattern")
        os.makedirs(duo_attn_dir, exist_ok=True)

        tsv_path = os.path.join(duo_attn_dir, "full_attention_heads.tsv")
        with torch.no_grad(), open(tsv_path, "w") as f:
            for layer_idx in range(model.config.num_hidden_layers):
                duo_attn_alpha = torch.nn.functional.sigmoid(model.model.layers[layer_idx].self_attn.duo_attn_alpha)
                f.write("\t".join(f"{value.item():.10f}" for value in duo_attn_alpha) + "\n")

        config_payload = {
            "sink_size": model.config.duo_attn_sink_size,
            "recent_size": model.config.local_window_size
        }
        with open(os.path.join(duo_attn_dir, "config.json"), "w") as f:
            json.dump(config_payload, f)
    else:
        other_params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        other_path = os.path.join(base_path, "other.pt")
        torch.save(other_params, other_path)


def generate_ecdf_plots(histogram_counts, histogram_bin_edges, num_kv_heads, g_threshold, output_dir):
    ecdf_dir = os.path.join(output_dir, "ecdf")
    os.makedirs(ecdf_dir, exist_ok=True)

    ecdf_x = histogram_bin_edges.to(dtype=torch.float64, device='cpu')

    for layer_idx, layer_counts in enumerate(histogram_counts):
        for head_idx in range(num_kv_heads):
            counts = layer_counts[head_idx]
            total = counts.sum().item()

            cumulative = torch.cumsum(counts, dim=0, dtype=torch.float64) / total
            ecdf_y = torch.cat([cumulative.new_zeros((1,)), cumulative]).cpu()

            plt.step(ecdf_x.numpy(), ecdf_y.numpy(), where='post')
            plt.xlim(ecdf_x[0].item(), ecdf_x[-1].item())
            plt.ylim(0.0, 1.0)
            plt.xlabel("g")
            plt.ylabel("ECDF")
            plt.title(f"Layer {layer_idx} Head {head_idx} (n={total})")
            plt.axvline(x=g_threshold, color='r', linestyle='--')
            plt.grid(alpha=0.5)
            plt.tight_layout()

            plot_path = os.path.join(ecdf_dir, f"layer_{layer_idx}_head_{head_idx}.png")
            plt.savefig(plot_path)
            plt.clf()

    print(f"Validation ECDF plots saved to {ecdf_dir}")


def main():
    config = get_config()
    exp_name = create_exp_name(config)
    run = wandb.init(project="sparse", group=config.group_name, id=exp_name, config=config, save_code=True)

    model_config = AutoConfig.from_pretrained(config.model_name)
    attr_names = [
        "checkpointing_threshold",
        "local_window_size",
        "g_inputs",
        "g_act_fn",
        "g_expand",
        "g_epsilon",
        "g_rms",
        "g_threshold",
        "g_ungate_count",
        "use_duo_attn",
        "duo_attn_sink_size",
    ]
    for attr_name in attr_names:
        value = getattr(config, attr_name)
        if value is not None:
            setattr(model_config, attr_name, value) 

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map=config.device,
    ).eval()

    for param in model.parameters():
        param.requires_grad = False

    flex_attention_compiled = torch.compile(flex_attention, mode="max-autotune")
    create_block_mask_compiled = torch.compile(create_block_mask)

    for layer in model.model.layers:
        layer.self_attn.flex_attention_compiled = flex_attention_compiled

        if config.use_duo_attn:
            layer.self_attn.duo_attn_alpha.to(torch.float32)
            layer.self_attn.duo_attn_alpha.data.fill_(0.0)
        else:
            layer.self_attn.g_predictors.to(torch.float32)
            layer.self_attn.g_predictors.reset_parameters()
            layer.self_attn.g_predictors.train()

    total_param_counts = 0
    for name, param in model.named_parameters():
        if ('g_predictors' in name) or ('duo_attn_alpha' in name):
            param.requires_grad = True
        total_param_counts += param.numel()

    trainable_param_counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {trainable_param_counts:,d} || all params: {total_param_counts:,d} || trainable%: {100 * trainable_param_counts / total_param_counts:.4f}")

    config.output_dir = f"{config.output_dir}/{config.group_name}/{exp_name}"
    os.makedirs(config.output_dir, exist_ok=True)
    print(f"Output directory: {config.output_dir}")

    other_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    freezed_param_names = [n for n, p in model.named_parameters() if not p.requires_grad]

    save_param_names(config.output_dir, "other_params.txt", other_param_names)
    save_param_names(config.output_dir, "freezed_params.txt", freezed_param_names)

    other_params = [p for n, p in model.named_parameters() if n in other_param_names]

    fineweb_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    fineweb_train_dataset = _normalize_messages_dataset(
        _process_dataset_from_config(fineweb_dataset, model_config.fineweb_bins, config.fineweb_config, False),
        _build_fineweb_messages_for_map,
    )
    fineweb_val_dataset = _normalize_messages_dataset(
        _process_dataset_from_config(fineweb_dataset, model_config.fineweb_bins, config.fineweb_config_val, True),
        _build_fineweb_messages_for_map,
    )

    nemotron_dataset = load_dataset("WG-KV/nemotron-math-v2-qwen3-bins")
    nemotron_train_dataset = (
        _normalize_messages_dataset(
            _process_dataset_from_split_config(nemotron_dataset, config.nemotron_config, False),
            _build_nemotron_messages_for_map,
        )
        if config.nemotron_config else None
    )
    nemotron_val_dataset = (
        _normalize_messages_dataset(
            _process_dataset_from_split_config(nemotron_dataset, config.nemotron_config_val, True),
            _build_nemotron_messages_for_map,
        )
        if config.nemotron_config_val else None
    )

    train_dataset = (
        fineweb_train_dataset
        if nemotron_train_dataset is None
        else concatenate_datasets([fineweb_train_dataset, nemotron_train_dataset])
    ).shuffle(seed=42)
    val_dataset = (
        fineweb_val_dataset
        if nemotron_val_dataset is None
        else concatenate_datasets([fineweb_val_dataset, nemotron_val_dataset])
    )

    data_len, data_generator = preprocess_data(train_dataset, config, tokenizer)
    val_data_len, val_data_generator = preprocess_data(val_dataset, config, tokenizer)
    predictor_optimizer = torch.optim.AdamW(other_params, lr=config.lr, weight_decay=config.weight_decay)
    predictor_scheduler = get_wsd_schedule(
        predictor_optimizer,
        num_warmup_steps=int(config.warmup_ratio * data_len),
        num_decay_steps=int(config.decay_ratio * data_len),
        num_training_steps=data_len,
    )

    distillation_loss_fn = F.mse_loss
    regularization_loss_fn = create_regularization_loss_fn(model.config.local_window_size, model, config.use_duo_attn)
    total_loss_fn = lambda distillation_loss, regularization_loss: distillation_loss + config.lambda_reg * regularization_loss

    max_seq_len_so_far = -1
    block_mask_cache = {}
    current_block_mask_seq_len = -1

    def ensure_block_mask(seq_len):
        nonlocal current_block_mask_seq_len
        if seq_len != current_block_mask_seq_len:
            if seq_len not in block_mask_cache:
                block_mask_cache[seq_len] = create_block_mask_compiled(causal_mask_mod, 1, 1, seq_len, seq_len, device=config.device)
            new_mask = block_mask_cache[seq_len]
            for layer in model.model.layers:
                layer.self_attn.block_mask = new_mask
            current_block_mask_seq_len = seq_len

    def train_loop(iter, input_ids, seq_len):
        input_ids = input_ids.to(config.device)
        assert input_ids.size(1) == seq_len

        nonlocal max_seq_len_so_far
        max_seq_len_so_far = max(max_seq_len_so_far, seq_len)

        ensure_block_mask(seq_len)

        model.gating_mode = 2
        with torch.no_grad():
            outputs_teacher = model(input_ids=input_ids)

        model.gating_mode = 1
        set_hard_threshold(model, True)
        with torch.no_grad():
            with torch.autocast(model.device.type, dtype=torch.bfloat16):
                outputs_student = model(input_ids=input_ids)
                distillation_loss_hard = distillation_loss_fn(
                    outputs_student.last_hidden_state,
                    outputs_teacher.last_hidden_state,
                )
        set_hard_threshold(model, False)

        model.gating_mode = 1
        torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(model.device.type, dtype=torch.bfloat16):
            outputs_student = model(input_ids=input_ids)
            total_loss, distillation_loss, regularization_loss, all_g_scores = compute_loss_and_backward(
                distillation_loss_fn,
                regularization_loss_fn,
                total_loss_fn,
                outputs_teacher.last_hidden_state,
                outputs_student.last_hidden_state,
                outputs_student.all_g_scores,
            )

        predictor_optimizer.step()

        batch_size = input_ids.size(0)
        for _ in range(batch_size):
            predictor_scheduler.step()

        predictor_optimizer.zero_grad()

        all_g_mask = all_g_scores > model.config.g_threshold
        g_mean = all_g_scores.float().mean().item()
        g_mask_mean = all_g_mask.float().mean().item()
        current_lr = predictor_scheduler.get_last_lr()[0]

        run.log({
            'loss/total': total_loss,
            'loss/distill': distillation_loss,
            'loss/distill_hard': distillation_loss_hard,
            'loss/reg': regularization_loss,
            'metrics/g_mean': g_mean,
            'metrics/g_mask_mean': g_mask_mean,
            'params/iter': iter,
            'params/lr': current_lr,
            'params/N': seq_len,
        })

        pbar.set_description(
            f"Loss: {total_loss:.4f}, "
            f"Distill: {distillation_loss:.4f}, "
            f"Reg: {regularization_loss:.4f}, "
            f"G: {g_mean:.4f}, "
            f"N: {seq_len}/{max_seq_len_so_far}"
        )

        if config.save_every > 0 and (iter % config.save_every == 0 or iter == data_len - 1):
            # Aggregated plot
            aggregated_dir = os.path.join(config.output_dir, "aggregated")
            os.makedirs(aggregated_dir, exist_ok=True)
            flattened_g_scores_np = all_g_scores.to(torch.float32).flatten().cpu().numpy()
            plt.hist(flattened_g_scores_np, bins=50, edgecolor='black', range=(0, 1))
            plt.ylim(0, flattened_g_scores_np.size)
            plt.savefig(os.path.join(aggregated_dir, f"iter_{iter}.png"))
            plt.clf()

            # Head-wise plots
            head_wise_dir = os.path.join(config.output_dir, "head_wise")
            all_dir = os.path.join(head_wise_dir, "all")
            all_mask_dir = os.path.join(head_wise_dir, "all_mask")
            os.makedirs(all_dir, exist_ok=True)
            os.makedirs(all_mask_dir, exist_ok=True)
            num_layers, _, num_heads, _ = all_g_scores.shape
            for layer_idx in range(num_layers):
                layer_dir = os.path.join(head_wise_dir, f"layer_{layer_idx}")
                for head_idx in range(num_heads):
                    head_dir = os.path.join(layer_dir, f"head_{head_idx}")
                    os.makedirs(head_dir, exist_ok=True)

                    head_scores_np = all_g_scores[layer_idx, 0, head_idx].to(torch.float32).cpu().numpy()
                    plt.hist(head_scores_np, bins=50, edgecolor='black', range=(0, 1))
                    plt.ylim(0, head_scores_np.size)
                    plt.savefig(os.path.join(head_dir, f"iter_{iter}.png"))
                    plt.savefig(os.path.join(all_dir, f"layer_{layer_idx}_head_{head_idx}.png"))
                    plt.clf()

                    head_masks_np = all_g_mask[layer_idx, 0, head_idx].to(torch.float32).cpu().numpy()
                    plt.hist(head_masks_np, bins=50, edgecolor='black', range=(0, 1))
                    plt.ylim(0, head_masks_np.size)
                    plt.savefig(os.path.join(all_mask_dir, f"layer_{layer_idx}_head_{head_idx}.png"))
                    plt.clf()

            save_model(model, config, f"iter_{iter}")

    #torch.cuda.memory._record_memory_history()
    if not config.load_pretrained_weights and not config.load_duo_attn_pattern:
        pbar = tqdm(total=data_len)
        for input_ids, seq_len in data_generator():
            iter = pbar.n
            train_loop(iter, input_ids, seq_len)
            batch_size = input_ids.size(0)
            pbar.update(batch_size)
        pbar.close()
    #torch.cuda.memory._dump_snapshot("snapshot.pickle")
    #torch.cuda.memory._record_memory_history(enabled=None)

    if config.load_pretrained_weights:
        state_dict = torch.load(config.load_pretrained_weights)
        _, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0
        print(f"Loaded pretrained weights from {config.load_pretrained_weights}")
    
    if config.load_duo_attn_pattern:
        attn_heads, _, _ = load_attn_pattern(config.load_duo_attn_pattern)
        attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=config.duo_attn_sparsity)
        set_duo_attn_alpha(model, attn_heads)
        print(f"Loaded DuoAttention pattern from {config.load_duo_attn_pattern} with sparsity {sparsity}")

    histogram_num_bins = 1000
    histogram_bin_edges = torch.linspace(0.0, 1.0, histogram_num_bins + 1, device=config.device)
    histogram_counts = [
        torch.zeros(model.config.num_key_value_heads, histogram_num_bins, dtype=torch.int64, device=config.device)
        for _ in range(model.config.num_hidden_layers)
    ]

    val_weight = 0
    val_distill_sum = 0.0
    val_distill_hard_sum = 0.0
    val_g_sum = 0.0
    val_g_mask_sum = 0.0

    def val_loop(input_ids, seq_len):
        nonlocal val_weight, val_distill_sum, val_distill_hard_sum, val_g_sum, val_g_mask_sum

        input_ids = input_ids.to(config.device)
        assert input_ids.size(1) == seq_len

        ensure_block_mask(seq_len)

        with torch.no_grad():
            model.gating_mode = 2
            outputs_teacher = model(input_ids=input_ids)

            model.gating_mode = 1
            set_hard_threshold(model, True)
            with torch.autocast(model.device.type, dtype=torch.bfloat16):
                outputs_student = model(input_ids=input_ids)
                distillation_loss_hard = distillation_loss_fn(
                    outputs_student.last_hidden_state,
                    outputs_teacher.last_hidden_state,
                )
            set_hard_threshold(model, False)

            model.gating_mode = 1
            with torch.autocast(model.device.type, dtype=torch.bfloat16):
                outputs_student = model(input_ids=input_ids)
                distillation_loss = distillation_loss_fn(
                    outputs_student.last_hidden_state,
                    outputs_teacher.last_hidden_state,
                )

            token_count = input_ids.size(0) * seq_len
            val_weight += token_count

            val_distill_sum += distillation_loss.item() * token_count
            val_distill_hard_sum += distillation_loss_hard.item() * token_count

            for layer_idx, layer_scores in enumerate(outputs_student.all_g_scores):
                flat_scores = rearrange(layer_scores, 'b h t -> h (b t)')
                bin_indices = (torch.bucketize(flat_scores, histogram_bin_edges) - 1).clamp(min=0, max=histogram_num_bins - 1)
                ones = torch.ones_like(bin_indices, dtype=torch.int64)
                histogram_counts[layer_idx].scatter_add_(1, bin_indices, ones)

            all_g_scores = torch.stack(outputs_student.all_g_scores, dim=0)
            all_g_mask = all_g_scores > model.config.g_threshold
            g_mean = all_g_scores.float().mean().item()
            g_mask_mean = all_g_mask.float().mean().item()
            val_g_sum += g_mean * token_count
            val_g_mask_sum += g_mask_mean * token_count

    val_pbar = tqdm(total=val_data_len)
    for input_ids, seq_len in val_data_generator():
        val_loop(input_ids, seq_len)
        batch_size = input_ids.size(0)
        val_pbar.update(batch_size)
    val_pbar.close()

    val_distill_loss = val_distill_sum / val_weight
    val_distill_loss_hard = val_distill_hard_sum / val_weight
    val_g_mean = val_g_sum / val_weight
    val_g_mask_mean = val_g_mask_sum / val_weight

    generate_ecdf_plots(
        histogram_counts,
        histogram_bin_edges,
        model.config.num_key_value_heads,
        model.config.g_threshold,
        config.output_dir
    )

    run.summary["val/distill_loss"] = val_distill_loss
    run.summary["val/distill_loss_hard"] = val_distill_loss_hard
    run.summary["val/g_mean"] = val_g_mean
    run.summary["val/g_mask_mean"] = val_g_mask_mean

    run.finish()
    

if __name__ == "__main__":
    main()
