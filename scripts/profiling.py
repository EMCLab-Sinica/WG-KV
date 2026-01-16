import argparse
import torch
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
from dataclasses import dataclass

@dataclass
class ElapsedTimeTracker:
    value: float = 0.0

@dataclass
class TimingPhase:
    name: str = "warmup"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--g_expand",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--g_rms",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max_tokens_per_head",
        type=int,
        default=33000,
    )
    parser.add_argument(
        "--random_sparsity",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--prompt_length",
        type=int,
        default=32768,
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--num_warmup_decode_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num_decode_steps",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--profile_attn_mlp",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--attn_granularity",
        choices=["rms_linear_attn", "linear_attn", "attn", "wrapper", "kernel", "predictor"],
        default="wrapper",
    )
    return parser.parse_args()


def measure_time(start_event, end_event, func, *args, **kwargs):
    torch.cuda.synchronize()
    start_event.record()
    result = func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return result, elapsed_time_ms


def prefill_forward(model, inputs, cache):
    return model(
        **inputs,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )


def decode_forward(model, token, cache):
    return model(
        input_ids=token,
        past_key_values=cache,
        use_cache=True,
    )


def select_next_token(logits):
    return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)


def get_avg_tokens_in_kv_cache(past_key_values, model):
    total_tokens_in_kv_cache = past_key_values.next_free_block * model.config.block_size
    average_tokens_in_kv_cache = total_tokens_in_kv_cache / (model.config.num_hidden_layers * model.config.num_key_value_heads)
    return average_tokens_in_kv_cache


def bytes_to_mib(num_bytes):
    return num_bytes / (1024 * 1024)


def tensor_bytes(tensor):
    assert tensor.is_cuda
    return tensor.numel() * tensor.element_size()


def get_model_gpu_memory_bytes(model):
    seen_storages = set()
    total_bytes = 0

    for tensor in model.parameters():
        storage_id = tensor.untyped_storage().data_ptr()
        assert storage_id not in seen_storages
        seen_storages.add(storage_id)
        total_bytes += tensor_bytes(tensor)

    return total_bytes


def get_kv_cache_memory_bytes(past_key_value):
    block_usage_ratio = past_key_value.next_free_block / past_key_value.total_blocks
    return sum([
        tensor_bytes(past_key_value.block_table),
        tensor_bytes(past_key_value.block_pool_k) * block_usage_ratio,
        tensor_bytes(past_key_value.block_pool_v) * block_usage_ratio,
    ])


def get_func_with_timing(start_event_layer, end_event_layer, func, elapsed_time_tracker_getter):
    def func_with_timing(*args, **kwargs):
        elapsed_time_tracker = elapsed_time_tracker_getter()
        if elapsed_time_tracker is None:
            return func(*args, **kwargs)
        result, elapsed_time_ms = measure_time(start_event_layer, end_event_layer, func, *args, **kwargs)
        elapsed_time_tracker.value += elapsed_time_ms
        return result
    return func_with_timing


def main():
    args = parse_args()
    model_config = AutoConfig.from_pretrained(args.model_name)

    if args.g_expand is not None:
        model_config.g_expand = args.g_expand

    model_config.g_rms = args.g_rms
    model_config.max_total_tokens = args.max_tokens_per_head * model_config.num_hidden_layers * model_config.num_key_value_heads
    model_config.max_tokens_per_head = args.max_tokens_per_head
    model_config.random_sparsity = args.random_sparsity
    model_config.g_fast_path = True

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=model_config,
        torch_dtype="auto",
        device_map="auto"
    )
    model.gating_mode = 3

    inputs = {
        "input_ids": torch.zeros((1, args.prompt_length), dtype=torch.long, device=model.device),
        "attention_mask": torch.ones((1, args.prompt_length), dtype=torch.long, device=model.device),
    }

    timing_phase = TimingPhase()

    if args.profile_attn_mlp:
        attn_prefill_elapsed_time_tracker = ElapsedTimeTracker()
        attn_decode_elapsed_time_tracker = ElapsedTimeTracker()
        mlp_prefill_elapsed_time_tracker = ElapsedTimeTracker()
        mlp_decode_elapsed_time_tracker = ElapsedTimeTracker()
        lm_head_prefill_elapsed_time_tracker = ElapsedTimeTracker()
        lm_head_decode_elapsed_time_tracker = ElapsedTimeTracker()

        def get_attn_tracker():
            if timing_phase.name == "prefill":
                return attn_prefill_elapsed_time_tracker
            if timing_phase.name == "decode":
                return attn_decode_elapsed_time_tracker
            return None

        def get_mlp_tracker():
            if timing_phase.name == "prefill":
                return mlp_prefill_elapsed_time_tracker
            if timing_phase.name == "decode":
                return mlp_decode_elapsed_time_tracker
            return None

        def get_lm_head_tracker():
            if timing_phase.name == "prefill":
                return lm_head_prefill_elapsed_time_tracker
            if timing_phase.name == "decode":
                return lm_head_decode_elapsed_time_tracker
            return None

        start_event_layer = torch.cuda.Event(enable_timing=True)
        end_event_layer = torch.cuda.Event(enable_timing=True)

        for layer in model.model.layers:
            if args.attn_granularity == "rms_linear_attn":
                layer.rmsnorm_and_self_attn = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.rmsnorm_and_self_attn,
                    get_attn_tracker
                )
            elif args.attn_granularity == "linear_attn":
                layer.self_attn.forward = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.self_attn.forward,
                    get_attn_tracker
                )
            elif args.attn_granularity == "attn":
                layer.self_attn.self_attn_forward_patch = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.self_attn.self_attn_forward_patch,
                    get_attn_tracker
                )
            elif args.attn_granularity == "wrapper":
                layer.self_attn.prefill_wrapper = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.self_attn.prefill_wrapper,
                    get_attn_tracker
                )
                layer.self_attn.decode_wrapper = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.self_attn.decode_wrapper,
                    get_attn_tracker
                )
            elif args.attn_granularity == "kernel":
                layer.self_attn.prefill_kernel = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.self_attn.prefill_kernel,
                    get_attn_tracker
                )
                layer.self_attn.decode_kernel = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.self_attn.decode_kernel,
                    get_attn_tracker
                )
            else:  # predictor
                layer.self_attn.g_predictors.forward = get_func_with_timing(
                    start_event_layer,
                    end_event_layer,
                    layer.self_attn.g_predictors.forward,
                    get_attn_tracker
                )

            layer.mlp.forward = get_func_with_timing(
                start_event_layer,
                end_event_layer,
                layer.mlp.forward,
                get_mlp_tracker
            )

        model.lm_head.forward = get_func_with_timing(
            start_event_layer,
            end_event_layer,
            model.lm_head.forward,
            get_lm_head_tracker
        )

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        # ==========
        # Warmup
        # ==========
        timing_phase.name = "warmup"

        def warmup():
            warmup_cache = DynamicCache()

            warmup_outputs = prefill_forward(model, inputs, warmup_cache)
            warmup_next_token = select_next_token(warmup_outputs.logits)

            for _ in range(args.num_warmup_decode_steps):
                warmup_outputs = decode_forward(model, warmup_next_token, warmup_cache)
                warmup_next_token = select_next_token(warmup_outputs.logits)

        for _ in range(args.num_warmup):
            warmup()

        torch.cuda.synchronize()

        # ==========
        # Prefill
        # ==========
        timing_phase.name = "prefill"
        cache = DynamicCache()

        outputs, prefill_time_ms = measure_time(start_event, end_event, prefill_forward, model, inputs, cache)
        next_token = select_next_token(outputs.logits)

        avg_tokens_before = get_avg_tokens_in_kv_cache(cache, model)

        # ==========
        # Decode
        # ==========
        timing_phase.name = "decode"
        decode_times = []

        for _ in range(args.num_decode_steps):
            outputs, decode_time_ms = measure_time(start_event, end_event, decode_forward, model, next_token, cache)
            next_token = select_next_token(outputs.logits)
            decode_times.append(decode_time_ms)

        avg_tokens_after = get_avg_tokens_in_kv_cache(cache, model)

        # ==========
        # Stats
        # ==========
        prefill_throughput = args.prompt_length / (prefill_time_ms / 1000)
        total_decode_time = sum(decode_times)
        avg_decode_time = sum(decode_times) / len(decode_times)
        decode_throughput = 1000 / avg_decode_time
        model_mem_mib = bytes_to_mib(get_model_gpu_memory_bytes(model))
        kv_cache_mem_mib = bytes_to_mib(get_kv_cache_memory_bytes(cache))

        print(f"Prompt Length         : {args.prompt_length}")
        print(f"Prefill Time          : {prefill_time_ms:.2f} ms")
        print(f"Prefill Throughput    : {prefill_throughput:.2f} tokens/s")
        print(f"Avg Tokens in KV Cache: {avg_tokens_before:.0f} (before decoding)")
        print(f"Total Decode Time     : {total_decode_time:.2f} ms (for {args.num_decode_steps} steps)")
        print(f"Avg Decode Time       : {avg_decode_time:.2f} ms")
        print(f"Decode Throughput     : {decode_throughput:.2f} tokens/s")
        print(f"Avg Tokens in KV Cache: {avg_tokens_after:.0f} (after decoding)")
        print(f"Model GPU Memory      : {model_mem_mib:.2f} MiB")
        print(f"KV Cache GPU Memory   : {kv_cache_mem_mib:.2f} MiB")

        if args.profile_attn_mlp:
            print(f"Prefill Attention Time: {attn_prefill_elapsed_time_tracker.value:.2f} ms")
            print(f"Prefill MLP Time      : {mlp_prefill_elapsed_time_tracker.value:.2f} ms")
            print(f"Prefill LM Head Time  : {lm_head_prefill_elapsed_time_tracker.value:.2f} ms")
            print(f"Decode Attention Time : {attn_decode_elapsed_time_tracker.value:.2f} ms")
            print(f"Decode MLP Time       : {mlp_decode_elapsed_time_tracker.value:.2f} ms")
            print(f"Decode LM Head Time   : {lm_head_decode_elapsed_time_tracker.value:.2f} ms")
        else:
            print("Attention/MLP profiling disabled (--profile_attn_mlp to enable)")


if __name__ == "__main__":
    main()
