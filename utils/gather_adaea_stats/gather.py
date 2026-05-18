import argparse
import torch
import math
import os
from torch import nn
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from kvpress import BasePress, ExpectedAttentionPress, AdaKVPress
from kvpress.utils import get_prerope_query_states
from transformers.models.llama.modeling_llama import repeat_kv
from tqdm import tqdm

from unittest.mock import MagicMock, patch
import transformers.modeling_layers
transformers.modeling_layers.set_duo_attn_alpha = MagicMock()

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train import (
    _normalize_messages_dataset,
    _process_dataset_from_config,
    _build_fineweb_messages_for_map,
    preprocess_data,
)


def get_config():
    parser = argparse.ArgumentParser(allow_abbrev=False)

    parser.add_argument('sparsity', nargs='+', type=float)
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument('--fineweb_config', nargs='+', type=str, default=["4-8k:4096:4000", "8-16k:8192:2000", "16-32k:16384:1000", "32-64k:32768:500"])
    parser.add_argument('--rope_start_pos', type=int, default=32768)

    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--subset', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action="store_true")
    parser.add_argument('--inference', action="store_true")

    config = parser.parse_args()
    config.global_batch_size = 1

    return config


class QueryTracker:
    def __init__(self):
        self.mean = None
        self.cov = None
        self.count = 0

    def add(self, query_states):
        assert query_states.shape[0] == 1
        query_states = query_states[0].to(torch.float64)

        _, num_queries, _ = query_states.shape
        new_count = self.count + num_queries

        mean = query_states.mean(dim=1)

        center = query_states - mean[:, None, :]
        cov = torch.bmm(center.transpose(1, 2), center)

        if self.count == 0:
            self.mean = mean
            self.cov = cov
            self.count = num_queries
        else:
            delta = mean - self.mean
            self.mean = self.mean + delta * (num_queries / new_count)
            delta_outer = torch.bmm(delta.unsqueeze(2), delta.unsqueeze(1))
            self.cov = self.cov + cov + delta_outer * (self.count * num_queries / new_count)
            self.count = new_count
    
    def get_stats(self):
        cov = self.cov / (self.count - 1)
        return self.mean, cov


class QueryStatisticsPress(BasePress):
    def __init__(self, num_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.query_trackers = [QueryTracker() for _ in range(num_layers)]

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        query_states = get_prerope_query_states(module, hidden_states)
        self.query_trackers[module.layer_idx].add(query_states)
        return keys, values
    
    def get_stats(self):
        mean_query_prerope = [tracker.get_stats()[0] for tracker in self.query_trackers]
        cov_query_prerope = [tracker.get_stats()[1] for tracker in self.query_trackers]
        return mean_query_prerope, cov_query_prerope


class ExpectedAttentionStatisticsPress(ExpectedAttentionPress):
    def __init__(self, mean_query_prerope, cov_query_prerope, rope_start_pos, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mean_query_prerope = mean_query_prerope
        self.cov_query_prerope = cov_query_prerope
        self.rope_start_pos = rope_start_pos

        self.mean_query = [None for _ in range(len(mean_query_prerope))]
        self.cov_query = [None for _ in range(len(cov_query_prerope))]

        self.pre_max_scores = {}

    def get_query_statistics(self, module, hidden_states):
        if self.mean_query[module.layer_idx] is None:
            mean_query_prerope = self.mean_query_prerope[module.layer_idx].unsqueeze(0).to(torch.bfloat16)
            cov_query_prerope = self.cov_query_prerope[module.layer_idx].unsqueeze(0).to(torch.bfloat16)
            self.mean_query[module.layer_idx], self.cov_query[module.layer_idx] = self.apply_avg_rope(module, mean_query_prerope, cov_query_prerope, self.rope_start_pos)
        return self.mean_query[module.layer_idx], self.cov_query[module.layer_idx]
    
    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        # Remove sink tokens
        assert keys.size(2) > self.n_sink, f"Input should contain more tokens than n_sink={self.n_sink}"
        keys = keys[:, :, self.n_sink :]
        values = values[:, :, self.n_sink :]

        # Compute query statistics
        mean_query, cov_query = self.get_query_statistics(module, hidden_states)

        # Compute scores
        bsz, num_key_value_heads, q_len, d = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        keys = repeat_kv(keys, num_key_value_groups).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), keys).squeeze(2) / math.sqrt(d)
        if self.use_covariance:
            scores += torch.einsum("bhin, bhij, bhjn->bhn", keys, cov_query, keys) / d / 2

        # Average scores across groups
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        pre_max_scores = scores

        scores = F.softmax(scores, dim=-1)
        scores = scores.max(dim=2)[0]

        # Add back the sink tokens. Use max score to make sure they are not pruned.
        scores = F.pad(scores, (self.n_sink, 0), value=scores.max().item())
        self.pre_max_scores[module.layer_idx] = F.pad(pre_max_scores, (self.n_sink, 0), value=pre_max_scores.max().item())

        return scores


class AdaExpectedAttentionPress(BasePress):
    def __init__(
        self,
        mean_query,
        cov_query,
        threshold,
        n_sink=4,
    ):
        super().__init__()
        self.mean_query = mean_query
        self.cov_query = cov_query
        self.threshold = threshold
        self.n_sink = n_sink

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        bsz, num_key_value_heads, k_len, d = keys.shape

        mean_query = self.mean_query[module.layer_idx]
        cov_query = self.cov_query[module.layer_idx]

        expanded_keys = keys.repeat_interleave(module.num_key_value_groups, dim=1).transpose(2, 3)
        scores = torch.matmul(mean_query.unsqueeze(2), expanded_keys).squeeze(2) / math.sqrt(d)
        scores += torch.einsum("bhin, bhij, bhjn->bhn", expanded_keys, cov_query, expanded_keys) / d / 2

        scores = scores.view(bsz, num_key_value_heads, module.num_key_value_groups, k_len)
        scores[:, :, :, :self.n_sink] = float("inf")

        threshold = self.threshold[module.layer_idx].reshape(num_key_value_heads, module.num_key_value_groups)[None, :, :, None]
        mask = (scores <= threshold).all(dim=2)
        batch_indices, head_indices, seq_indices = torch.where(mask)

        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values


if __name__ == "__main__":
    config = get_config()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16,
        device_map="cuda",
    )

    if "llama" in config.model_name.lower():
        fineweb_bins = "WG-KV/fineweb-llama-bins"
    elif "qwen3" in config.model_name.lower():
        fineweb_bins = "WG-KV/fineweb-qwen3-bins"
    else:
        assert False

    fineweb_dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train")
    fineweb_train_dataset = _normalize_messages_dataset(
        _process_dataset_from_config(fineweb_dataset, fineweb_bins, config.fineweb_config, False),
        _build_fineweb_messages_for_map,
    ).shuffle(seed=config.seed)

    if config.subset > 0:
        fineweb_train_dataset = fineweb_train_dataset.select(range(config.subset))
    data_len, data_generator = preprocess_data(fineweb_train_dataset, config, tokenizer)

    stats_press = QueryStatisticsPress(model.config.num_hidden_layers)

    for input_ids, seq_len in tqdm(data_generator(), total=data_len):
        input_ids = input_ids.to(model.device)
        with torch.no_grad(), stats_press(model):
            model(input_ids=input_ids)

    mean_query_prerope, cov_query_prerope = stats_press.get_stats()

    for sparsity in config.sparsity:
        ea_stats_press = ExpectedAttentionStatisticsPress(
            mean_query_prerope=mean_query_prerope,
            cov_query_prerope=cov_query_prerope,
            rope_start_pos=config.rope_start_pos,
            compression_ratio=sparsity,
        )
        ada_ea_stats_press = AdaKVPress(press=ea_stats_press, alpha_safeguard=0.0)

        retain_ratio_sum = torch.zeros(
            model.config.num_hidden_layers, model.config.num_key_value_heads,
            dtype=torch.float64, device=model.device
        )
        threshold_sum = torch.zeros(
            model.config.num_hidden_layers, model.config.num_attention_heads,
            dtype=torch.float64, device=model.device
        )

        for input_ids, seq_len in tqdm(data_generator(), total=data_len):
            input_ids = input_ids.to(model.device)
            with torch.no_grad(), ada_ea_stats_press(model):
                model(input_ids=input_ids)

            for layer_idx in range(model.config.num_hidden_layers):
                batch_indices, head_indices, seq_indices = model.model.layers[layer_idx].self_attn.masked_key_indices
                assert (batch_indices == 0).all().item()

                head_counts = torch.bincount(head_indices, minlength=model.config.num_key_value_heads)
                head_retain_ratio = 1.0 - head_counts / seq_len
                assert head_retain_ratio.shape == (model.config.num_key_value_heads,)
                retain_ratio_sum[layer_idx] += head_retain_ratio

                pre_max_scores = ea_stats_press.pre_max_scores.pop(layer_idx)[0]
                selected_scores = pre_max_scores[head_indices, :, seq_indices]

                num_key_value_groups = model.config.num_attention_heads // model.config.num_key_value_heads
                head_indices = head_indices.unsqueeze(1).expand(-1, num_key_value_groups)

                threshold = pre_max_scores.min(dim=2)[0]
                threshold.scatter_reduce_(dim=0, index=head_indices, src=selected_scores, reduce="amax")
                threshold_sum[layer_idx] += threshold.view(model.config.num_attention_heads)

        retain_ratio = retain_ratio_sum / data_len
        threshold = threshold_sum / data_len

        os.makedirs(config.output_dir, exist_ok=True)
        retain_ratio_tsv_path = f"{config.output_dir}/{sparsity}_retain_ratio.tsv"
        threshold_tsv_path = f"{config.output_dir}/{sparsity}_threshold.tsv"

        with open(retain_ratio_tsv_path, "w") as f:
            for row in retain_ratio:
                f.write("\t".join(f"{value.item():.4f}" for value in row) + "\n")

        with open(threshold_tsv_path, "w") as f:
            for row in threshold:
                f.write("\t".join(f"{value.item():10.4f}" for value in row) + "\n")

        print(f"Saved retain_ratio to {retain_ratio_tsv_path}")
        print(f"Saved threshold to {threshold_tsv_path}")

        ada_ea_press = AdaExpectedAttentionPress(
            mean_query=ea_stats_press.mean_query,
            cov_query=ea_stats_press.cov_query,
            threshold=threshold,
        )

        query_stats_path = f"{config.output_dir}/query_stats.pt"
        mean_query_cpu = [t.squeeze(0).cpu() for t in ea_stats_press.mean_query]
        cov_query_cpu = [t.squeeze(0).cpu() for t in ea_stats_press.cov_query]
        query_stats = {"mean_query": mean_query_cpu, "cov_query": cov_query_cpu}
        torch.save(query_stats, query_stats_path)
        print(f"Saved query_stats to {query_stats_path}")

        if config.verify:
            retain_ratio_verify_sum = torch.zeros(
                model.config.num_hidden_layers, model.config.num_key_value_heads,
                dtype=torch.float64, device=model.device
            )

            for input_ids, seq_len in tqdm(data_generator(), total=data_len):
                input_ids = input_ids.to(model.device)
                with torch.no_grad(), ada_ea_press(model):
                    model(input_ids=input_ids)

                for layer_idx in range(model.config.num_hidden_layers):
                    batch_indices, head_indices, _ = model.model.layers[layer_idx].self_attn.masked_key_indices
                    assert (batch_indices == 0).all().item()

                    head_counts = torch.bincount(head_indices, minlength=model.config.num_key_value_heads)
                    head_retain_ratio = 1.0 - head_counts / seq_len
                    assert head_retain_ratio.shape == (model.config.num_key_value_heads,)
                    retain_ratio_verify_sum[layer_idx] += head_retain_ratio

            retain_ratio_verify = retain_ratio_verify_sum / data_len

            retain_ratio_verify_tsv_path = f"{config.output_dir}/{sparsity}_retain_ratio_verify.tsv"

            with open(retain_ratio_verify_tsv_path, "w") as f:
                for row in retain_ratio_verify:
                    f.write("\t".join(f"{value.item():.4f}" for value in row) + "\n")

            print(f"Saved retain_ratio_verify to {retain_ratio_verify_tsv_path}")

        if config.inference:
            dataset = load_dataset('THUDM/LongBench', 'multifieldqa_en', split='test', trust_remote_code=True)
            prompt = dataset[7]['context'] + "\n\n" + dataset[7]['input'] + " Please also elaborate your reasoning."

            messages = [
                {"role": "user", "content": prompt}
            ]

            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)

            with ada_ea_press(model):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    streamer=TextStreamer(tokenizer),
                )