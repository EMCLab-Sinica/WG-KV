import torch
import torch.nn.functional as F
import pytest
from einops import rearrange
from transformers.modeling_layers import flash_attn_with_kvcache_wrapper

NUM_HEADS = [(4, 4), (8, 2), (8, 4)]  # (num_query_heads, num_kv_heads)
HEAD_SIZE = [64, 128]
BLOCK_SIZE = [16, 32, 64]
DTYPE = [torch.bfloat16]
BATCH_SIZE = [1, 3]
MAX_SEQLEN = [123, 456, 789, 4000]
NUM_BLOCKS = [4096]


def flash_attn_with_kvcache_ref(
    q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale
):
    batch_size, _, num_query_heads, head_size = q.shape
    _, block_size, _ = k_cache.shape
    _, num_kv_heads = cache_seqlens.shape
    num_kv_groups = num_query_heads // num_kv_heads

    # We process one sequence at a time
    outputs = []
    for batch_idx in range(batch_size):
        # Reconstruct K and V for each KV head separately
        k_heads = []
        v_heads = []

        for kv_head_idx in range(num_kv_heads):
            seqlen_i = cache_seqlens[batch_idx, kv_head_idx].item()

            # Reconstruct the K and V tensors for this KV head from the cache
            k_head = torch.empty((seqlen_i, head_size), device=q.device, dtype=q.dtype)
            v_head = torch.empty((seqlen_i, head_size), device=q.device, dtype=q.dtype)

            num_blocks_i = (seqlen_i + block_size - 1) // block_size
            for block_idx in range(num_blocks_i):
                physical_block_id = block_table[batch_idx, kv_head_idx, block_idx].item()

                start_idx = block_idx * block_size
                end_idx = min(start_idx + block_size, seqlen_i)
                len_block = end_idx - start_idx

                k_head[start_idx:end_idx] = k_cache[physical_block_id, :len_block]
                v_head[start_idx:end_idx] = v_cache[physical_block_id, :len_block]

            k_heads.append(k_head)
            v_heads.append(v_head)

        # Calculate attention for each Q head
        output_heads = []

        for q_head_idx in range(num_query_heads):
            kv_head_idx = q_head_idx // num_kv_groups

            q_head = q[batch_idx, 0, q_head_idx]  # Shape: (head_size,)
            k_head = k_heads[kv_head_idx]  # Shape: (seqlen, head_size)
            v_head = v_heads[kv_head_idx]  # Shape: (seqlen, head_size)

            attn_scores = torch.matmul(q_head, k_head.T) * softmax_scale  # Shape: (seqlen,)
            attn_probs = F.softmax(attn_scores, dim=-1)
            output_head = torch.matmul(attn_probs, v_head)  # Shape: (head_size,)
            output_heads.append(output_head)

        output_heads = rearrange(output_heads, 'num_query_heads head_size -> 1 num_query_heads head_size')
        outputs.append(output_heads)

    # Concatenate results for all sequences in the batch
    return torch.stack(outputs, dim=0)


@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZE)
@pytest.mark.parametrize("block_size", BLOCK_SIZE)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("batch_size", BATCH_SIZE)
@pytest.mark.parametrize("max_seqlen", MAX_SEQLEN)
@pytest.mark.parametrize("num_blocks", NUM_BLOCKS)
@torch.inference_mode()
def test_flash_attn_with_kvcache(
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    dtype: torch.dtype,
    batch_size: int,
    max_seqlen: int,
    num_blocks: int,
):
    torch.set_default_device("cuda")

    num_query_heads, num_kv_heads = num_heads
    assert num_query_heads % num_kv_heads == 0

    max_num_blocks_per_head = (max_seqlen + block_size - 1) // block_size
    scale = head_size ** -0.5
    cache_seqlens = torch.randint(1, max_seqlen + 1, (batch_size, num_kv_heads), dtype=torch.int32)

    block_table = torch.empty(batch_size, num_kv_heads, max_num_blocks_per_head, dtype=torch.int32)
    available_blocks = torch.randperm(num_blocks)

    block_offset = 0
    for batch_idx in range(batch_size):
        for kv_head_idx in range(num_kv_heads):
            num_blocks_i = (cache_seqlens[batch_idx, kv_head_idx].item() + block_size - 1) // block_size
            block_table[batch_idx, kv_head_idx, :num_blocks_i] = available_blocks[block_offset:block_offset + num_blocks_i]
            block_table[batch_idx, kv_head_idx, num_blocks_i:] = -1  # Padding with -1
            block_offset += num_blocks_i
            assert block_offset <= num_blocks, "Not enough blocks in the pool"

    q = torch.randn(batch_size, 1, num_query_heads, head_size, dtype=dtype)
    k_cache = torch.randn(num_blocks, block_size, head_size, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, head_size, dtype=dtype)

    flash_out = flash_attn_with_kvcache_wrapper(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
    )

    ref_out = flash_attn_with_kvcache_ref(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=scale,
    )

    atol, rtol = 1e-2, 1e-2
    max_diff = torch.max(torch.abs(flash_out - ref_out)).item()
    mean_diff = torch.mean(torch.abs(flash_out - ref_out)).item()

    print(f"Max absolute diff: {max_diff:.6f}")
    print(f"Mean absolute diff: {mean_diff:.6f}")

    torch.testing.assert_close(
        flash_out,
        ref_out,
        atol=atol,
        rtol=rtol,
    )

if __name__ == "__main__":
    pytest.main([__file__])
