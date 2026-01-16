import torch
import pytest
from test_sparse_utils import generate_random_data
from transformers.modeling_layers import generate_sparse_indices
from vllm.vllm_flash_attn import sparse_attn_func

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_kv_groups", [1, 2, 3, 4])
@pytest.mark.parametrize("seq_len", [2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 1000, 2000])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("local_window_size", [64, 128, 256])
@pytest.mark.parametrize("global_size", [0, 123, 345, 567, 789, 1000])
def test_sparse_attn(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim, local_window_size, global_size):
    num_heads, device, block_size, q, k, v, g_mask = generate_random_data(
        batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim, global_size
    )

    block_count, block_offset, column_count, column_index = generate_sparse_indices(
        g_mask, local_window_size, batch_size, num_kv_heads, seq_len, device, block_size
    )

    o = sparse_attn_func(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        block_count, block_offset, column_count, column_index,
        causal=True,
        return_softmax_lse=False,
    ).transpose(1, 2)

    # ====================

    attn_mask = torch.zeros(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool, device=device)
    num_rows_of_blocks = (seq_len + block_size - 1) // block_size

    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(num_rows_of_blocks):
                start_row = i * block_size
                end_row = min(start_row + block_size, seq_len)

                b_count = block_count[0, 0, i].item()
                if b_count > 0:
                    b_indices = block_offset[0, 0, i, :b_count]
                    for b_idx in b_indices:
                        start_col = b_idx.item()
                        end_col = min(start_col + block_size, seq_len)
                        attn_mask[b, h, start_row:end_row, start_col:end_col] = True
                
                c_count = column_count[b, h // num_kv_groups, i].item()
                if c_count > 0:
                    c_indices = column_index[b, h // num_kv_groups, 0, :c_count]
                    for c_idx in c_indices:
                        attn_mask[b, h, start_row:end_row, c_idx.item()] = True

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    final_mask = attn_mask & causal_mask

    o_manual_fp64 = torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float64), k.to(torch.float64), v.to(torch.float64), attn_mask=final_mask, enable_gqa=True
    )

    o_manual = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=final_mask, enable_gqa=True
    )

    o_max_diff = torch.max(torch.abs(o - o_manual_fp64))
    print(o_max_diff.item())

    o_manual_max_diff = torch.max(torch.abs(o_manual - o_manual_fp64))
    print(o_manual_max_diff.item())

    assert abs(o_max_diff - o_manual_max_diff) < 0.01

    # ====================

    ideal_mask = g_mask.unsqueeze(2).expand(-1, -1, seq_len, -1)
    ideal_mask = ideal_mask.repeat_interleave(num_kv_groups, dim=1)

    for i in range(num_rows_of_blocks):
        start_row = i * block_size
        end_row = min(start_row + block_size, seq_len)
        start_col = max(0, start_row - local_window_size + 1)
        ideal_mask[:, :, start_row:end_row, start_col:end_row] = True

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    ideal_mask = ideal_mask & causal_mask

    masks_are_identical = torch.all(final_mask == ideal_mask).item()
    print(masks_are_identical)

    assert masks_are_identical


if __name__ == "__main__":
    test_sparse_attn(1, 8, 4, 4096, 64, 256, 123)
