import torch

def generate_random_data(batch_size, num_kv_heads, num_kv_groups, seq_len, head_dim, global_size):
    torch.manual_seed(0)
    num_heads = num_kv_heads * num_kv_groups
    device = torch.device('cuda')
    block_size = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    k = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    v = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, dtype=torch.bfloat16, device=device)
    v_idx_list = [[torch.randperm(seq_len, dtype=torch.int32, device=device)[:global_size] for _ in range(num_kv_heads)] for _ in range(batch_size)]

    g_mask = torch.zeros(batch_size, num_kv_heads, seq_len, dtype=torch.bool, device=device)
    for b in range(batch_size):
        for h in range(num_kv_heads):
            g_mask[b, h, v_idx_list[b][h]] = True
    
    return num_heads, device, block_size, q, k, v, g_mask