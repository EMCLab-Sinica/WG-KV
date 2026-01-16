import argparse
from dataclasses import dataclass

import torch
from benchmark_utils import (
    setup_cuda_device,
    add_common_benchmark_args,
    benchmark_forward,
    format_run_description,
    format_iteration_summary,
)
from transformers.modeling_layers import flash_attn_with_kvcache_wrapper


@dataclass
class BenchmarkInputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    cache_seqlens: torch.Tensor
    block_table: torch.Tensor
    softmax_scale: float


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_benchmark_args(
        parser,
        warmup_default=100,
        measure_default=100,
    )
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--no_randomize_block_table", action="store_true")
    return parser.parse_args()


def prepare_inputs(
    *,
    seq_len: int,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    no_randomize_block_table: bool,
):
    max_num_blocks_per_head = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_kv_heads * max_num_blocks_per_head

    q = torch.randn(
        batch_size, 1, num_query_heads, head_dim,
        dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        total_blocks, block_size, head_dim,
        dtype=torch.bfloat16
    )
    v_cache = torch.randn_like(k_cache)

    cache_seqlens = torch.full(
        (batch_size, num_kv_heads),
        seq_len,
        dtype=torch.int32,
    )

    if no_randomize_block_table:
        block_ids = torch.arange(total_blocks, dtype=torch.int32)
    else:
        block_ids = torch.randperm(total_blocks, dtype=torch.int32)

    block_table = block_ids.reshape(batch_size, num_kv_heads, max_num_blocks_per_head)
    softmax_scale = head_dim ** -0.5

    return BenchmarkInputs(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
    )


def benchmark_kernel(
    inputs: BenchmarkInputs,
    *,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    def forward():
        flash_attn_with_kvcache_wrapper(
            inputs.q,
            inputs.k_cache,
            inputs.v_cache,
            inputs.cache_seqlens,
            inputs.block_table,
            inputs.softmax_scale,
        )

    avg_ms, _ = benchmark_forward(
        forward,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )
    return avg_ms


def main():
    args = parse_args()
    device = setup_cuda_device()
    assert args.num_query_heads % args.num_kv_heads == 0

    print(
        format_run_description(
            "flash_attn_with_kvcache_wrapper",
            device=device,
            args=args,
            extra_fields=(("block", args.block_size),),
        )
    )
    print(
        format_iteration_summary(
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
    )
    print(f"{'seq_len':>12} {'avg_ms':>12} {'tokens/s':>14}")

    for seq_len in args.sequence_lengths:
        inputs = prepare_inputs(
            seq_len=seq_len,
            batch_size=args.batch_size,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            block_size=args.block_size,
            no_randomize_block_table=args.no_randomize_block_table,
        )

        avg_ms = benchmark_kernel(
            inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )

        tokens_processed = args.batch_size * seq_len
        tokens_per_s = tokens_processed / (avg_ms / 1000.0)

        print(f"{seq_len:>12d} {avg_ms:>12.3f} {tokens_per_s:>14.1f}")


if __name__ == "__main__":
    main()
