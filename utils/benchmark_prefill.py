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
from transformers.modeling_layers import generate_sparse_indices
from vllm.vllm_flash_attn import sparse_attn_func, flash_attn_with_kvcache


@dataclass
class BenchmarkInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g_mask: torch.Tensor
    softmax_scale: float


def parse_args():
    parser = argparse.ArgumentParser()
    add_common_benchmark_args(
        parser,
        warmup_default=10,
        measure_default=10,
    )
    parser.add_argument("--local_window_size", type=int, default=256)
    parser.add_argument("--sparsity", type=float, default=0.5)
    return parser.parse_args()


def prepare_inputs(
    *,
    seq_len: int,
    batch_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_dim: int,
    sparsity: float,
) -> BenchmarkInputs:
    q = torch.randn(
        batch_size, seq_len, num_query_heads, head_dim,
        dtype=torch.bfloat16,
    )
    k = torch.randn(
        batch_size, seq_len, num_kv_heads, head_dim,
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    g_mask = (torch.rand(batch_size, num_kv_heads, seq_len) >= sparsity).to(torch.bool)
    softmax_scale = head_dim ** -0.5

    return BenchmarkInputs(
        q=q,
        k=k,
        v=v,
        g_mask=g_mask,
        softmax_scale=softmax_scale,
    )


def measure_sparse_index_generation(
    g_mask: torch.Tensor,
    *,
    local_window_size: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    warmup_iters: int,
    measure_iters: int,
) -> tuple[float, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    def generate():
        return generate_sparse_indices(
            g_mask,
            local_window_size,
            batch_size,
            num_kv_heads,
            seq_len,
            g_mask.device,
        )

    avg_ms, outputs = benchmark_forward(
        generate,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )

    block_count, block_offset, column_count, column_index = outputs
    return avg_ms, (block_count, block_offset, column_count, column_index)


def benchmark_kernel(
    inputs: BenchmarkInputs,
    *,
    block_count: torch.Tensor,
    block_offset: torch.Tensor,
    column_count: torch.Tensor,
    column_index: torch.Tensor,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    def forward():
        sparse_attn_func(
            inputs.q,
            inputs.k,
            inputs.v,
            block_count,
            block_offset,
            column_count,
            column_index,
            softmax_scale=inputs.softmax_scale,
            causal=True,
            return_softmax_lse=False,
        )

    avg_ms, _ = benchmark_forward(
        forward,
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
    )
    return avg_ms


def benchmark_dense_flash_attn(
    inputs: BenchmarkInputs,
    *,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    def forward():
        flash_attn_with_kvcache(
            inputs.q,
            inputs.k,
            inputs.v,
            softmax_scale=inputs.softmax_scale,
            causal=True,
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
            "sparse_attn_func",
            device=device,
            args=args,
            extra_fields=(
                ("local_window", args.local_window_size),
                ("sparsity", f"{args.sparsity:.3f}"),
            ),
        )
    )
    print(
        format_iteration_summary(
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
    )
    print(
        f"{'seq_len':>10} {'gen_ms':>12} {'sparse_ms':>12} {'total_ms':>12} "
        f"{'sparse_tok/s':>16} {'dense_ms':>12} {'dense_tok/s':>16}"
    )
    for seq_len in args.sequence_lengths:
        inputs = prepare_inputs(
            seq_len=seq_len,
            batch_size=args.batch_size,
            num_query_heads=args.num_query_heads,
            num_kv_heads=args.num_kv_heads,
            head_dim=args.head_dim,
            sparsity=args.sparsity,
        )

        gen_ms, (block_count, block_offset, column_count, column_index) = measure_sparse_index_generation(
            inputs.g_mask,
            local_window_size=args.local_window_size,
            batch_size=args.batch_size,
            num_kv_heads=args.num_kv_heads,
            seq_len=seq_len,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )

        sparse_ms = benchmark_kernel(
            inputs,
            block_count=block_count,
            block_offset=block_offset,
            column_count=column_count,
            column_index=column_index,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )

        dense_ms = benchmark_dense_flash_attn(
            inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )

        tokens_processed = args.batch_size * seq_len
        total_ms = gen_ms + sparse_ms
        sparse_tokens_per_s = tokens_processed / (total_ms / 1000.0)
        dense_tokens_per_s = tokens_processed / (dense_ms / 1000.0)

        print(
            f"{seq_len:>10d} {gen_ms:>12.3f} {sparse_ms:>12.3f} {total_ms:>12.3f} "
            f"{sparse_tokens_per_s:>16.1f} {dense_ms:>12.3f} {dense_tokens_per_s:>16.1f}"
        )


if __name__ == "__main__":
    main()
