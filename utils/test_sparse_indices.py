import torch
import pytest
from test_sparse_utils import generate_random_data
from transformers.modeling_layers import generate_sparse_indices, generate_sparse_indices_ref

@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_kv_heads", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("num_kv_groups", [1, 2, 3, 4])
@pytest.mark.parametrize("seq_len", [2, 63, 64, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 1000, 2000])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("local_window_size", [64, 128, 256])
@pytest.mark.parametrize("global_size", [0, 123, 345, 567, 789, 1000])
def test_sparse_indices(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim, local_window_size, global_size):
    num_heads, device, block_size, q, k, v, g_mask = generate_random_data(
        batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim, global_size
    )

    block_count, block_offset, column_count, column_index = generate_sparse_indices(
        g_mask, local_window_size, batch_size, num_kv_heads, seq_len, device, block_size
    )

    block_count_ref, block_offset_ref, column_count_ref, column_index_ref = generate_sparse_indices_ref(
        g_mask, local_window_size, batch_size, num_kv_heads, seq_len, device, block_size, block_size
    )

    # ====================

    num_rows_of_blocks = (seq_len + block_size - 1) // block_size

    block_count_match = torch.all(block_count == block_count_ref).item()

    block_offset_match = True
    for i in range(num_rows_of_blocks):
        b_count = block_count_ref[0, 0, i].item()
        if not torch.all(block_offset[0, 0, i, :b_count] == block_offset_ref[0, 0, i, :b_count]).item():
            block_offset_match = False

    column_count_match = torch.all(column_count == column_count_ref).item()

    column_index_match = True
    for b in range(batch_size):
        for h in range(num_kv_heads):
            for i in range(num_rows_of_blocks):
                c_count = column_count_ref[b, h, i].item()
                if not torch.all(column_index[b, h, 0, :c_count] == column_index_ref[b, h, 0, :c_count]).item():
                    column_index_match = False

    all_match = block_count_match and block_offset_match and column_count_match and column_index_match
    print(all_match)

    assert all_match


if __name__ == "__main__":
    test_sparse_indices(1, 8, 4, 4096, 64, 256, 123)
