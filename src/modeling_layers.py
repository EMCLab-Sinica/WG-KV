# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial

import torch
import torch.nn as nn

from transformers.utils import logging

from typing import Optional, List
from dataclasses import dataclass
from einops import rearrange, repeat
from .modeling_outputs import CausalLMOutputWithPast
from .integrations import use_kernel_forward_from_hub

try:
    from flash_attn import flash_attn_func
except:
    pass

try:
    from vllm.vllm_flash_attn import sparse_attn_func, flash_attn_with_kvcache
except:
    pass


logger = logging.get_logger(__name__)


class GradientCheckpointingLayer(nn.Module):
    """Base class for layers with gradient checkpointing.

    This class enables gradient checkpointing functionality for a layer. By default, gradient checkpointing is disabled
    (`gradient_checkpointing = False`). When `model.set_gradient_checkpointing()` is called, gradient checkpointing is
    enabled by setting `gradient_checkpointing = True` and assigning a checkpointing function to `_gradient_checkpointing_func`.

    Important:

        When using gradient checkpointing with `use_reentrant=True`, inputs that require gradients (e.g. hidden states)
        must be passed as positional arguments (`*args`) rather than keyword arguments to properly propagate gradients.

        Example:

            ```python
            >>> # Correct - hidden_states passed as positional arg
            >>> out = self.layer(hidden_states, attention_mask=attention_mask)

            >>> # Incorrect - hidden_states passed as keyword arg
            >>> out = self.layer(hidden_states=hidden_states, attention_mask=attention_mask)
            ```
    """

    gradient_checkpointing = False

    def __call__(self, *args, **kwargs):
        if self.gradient_checkpointing and self.training:
            do_warn = False
            layer_name = self.__class__.__name__
            message = f"Caching is incompatible with gradient checkpointing in {layer_name}. Setting"

            if "use_cache" in kwargs and kwargs["use_cache"]:
                kwargs["use_cache"] = False
                message += " `use_cache=False`,"
                do_warn = True

            # different names for the same thing in different layers
            if "past_key_value" in kwargs and kwargs["past_key_value"] is not None:
                kwargs["past_key_value"] = None
                message += " `past_key_value=None`,"
                do_warn = True

            if "past_key_values" in kwargs and kwargs["past_key_values"] is not None:
                kwargs["past_key_values"] = None
                message += " `past_key_values=None`,"
                do_warn = True

            if "layer_past" in kwargs and kwargs["layer_past"] is not None:
                kwargs["layer_past"] = None
                message += " `layer_past=None`,"
                do_warn = True

            # warn if anything was changed
            if do_warn:
                message = message.rstrip(",") + "."
                logger.warning(message)

            return self._gradient_checkpointing_func(partial(super().__call__, **kwargs), *args)
        return super().__call__(*args, **kwargs)


@dataclass
class DistillationCausalLMOutputWithPast(CausalLMOutputWithPast):
    last_hidden_state: Optional[torch.Tensor] = None
    all_g_scores: Optional[List[torch.Tensor]] = None


@use_kernel_forward_from_hub("RMSNorm")
class GatingRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        GatingRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class GatingPredictor(nn.Module):
    def __init__(self, head_dim, num_kv_heads, g_inputs, g_act_fn, g_expand, g_epsilon, g_rms, g_fast_path):
        super().__init__()
        self.g_inputs = g_inputs
        self.g_epsilon = g_epsilon
        self.g_fast_path = g_fast_path
        self.input_dim = head_dim * len(self.g_inputs)

        inner_dim = int(head_dim * g_expand)

        self.pre_key_norm = GatingRMSNorm(head_dim) if g_rms and 'pre_k' in self.g_inputs else nn.Identity()
        self.post_key_norm = GatingRMSNorm(head_dim) if g_rms and 'post_k' in self.g_inputs else nn.Identity()
        self.value_norm = GatingRMSNorm(head_dim) if g_rms and 'v' in self.g_inputs else nn.Identity()

        self.conv1 = nn.Conv1d(
            in_channels=num_kv_heads * self.input_dim,
            out_channels=num_kv_heads * inner_dim,
            kernel_size=1,
            groups=num_kv_heads
        )
        self.act_fn = nn.SiLU() if g_act_fn == "silu" else nn.GELU()
        self.conv2 = nn.Conv1d(
            in_channels=num_kv_heads * inner_dim,
            out_channels=num_kv_heads,
            kernel_size=1,
            groups=num_kv_heads
        )
        self.sigmoid = nn.Sigmoid()

    def reset_parameters(self):
        for norm in [self.pre_key_norm, self.post_key_norm, self.value_norm]:
            if isinstance(norm, GatingRMSNorm):
                nn.init.ones_(norm.weight)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, pre_key, post_key, value):
        batch_size, num_kv_heads, kv_seq_len, _ = post_key.shape

        states = []
        if 'pre_k' in self.g_inputs:
            pre_key = self.pre_key_norm(pre_key)
            states.append(pre_key)
        if 'post_k' in self.g_inputs:
            post_key = self.post_key_norm(post_key)
            states.append(post_key)
        if 'v' in self.g_inputs:
            value = self.value_norm(value)
            states.append(value)
        states = torch.cat(states, dim=-1)

        if self.g_fast_path:
            w1 = self.conv1.weight.view(num_kv_heads, -1, self.input_dim)
            b1 = self.conv1.bias.view(1, num_kv_heads, 1, -1)
            hidden_states = torch.einsum('bhli, hji -> bhlj', states, w1) + b1
            hidden_states = self.act_fn(hidden_states)

            w2 = self.conv2.weight.view(num_kv_heads, 1, -1)
            b2 = self.conv2.bias.view(1, num_kv_heads, 1, 1)
            hidden_states = torch.einsum('bhlj, hoj -> bhlo', hidden_states, w2) + b2

            g_scores = self.sigmoid(hidden_states.squeeze(-1))
            return g_scores

        permuted_states = states.permute(0, 1, 3, 2) # (batch_size, num_kv_heads, input_dim, kv_seq_len)
        reshaped_states = permuted_states.reshape(batch_size, num_kv_heads * self.input_dim, kv_seq_len)
        hidden_states = self.conv1(reshaped_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.conv2(hidden_states) # (batch_size, num_kv_heads, kv_seq_len)
        g_scores = self.sigmoid(hidden_states)

        return torch.clamp(g_scores, min=self.g_epsilon)


def create_score_mod(log_g_scores, local_window_size, num_kv_groups):
    def score_mod(score, b, h, q_idx, kv_idx):
        is_local = q_idx < (kv_idx + local_window_size)
        return torch.where(is_local, score, score + log_g_scores[b, h // num_kv_groups, kv_idx])
    return score_mod


def self_attn_forward_patch_train(
    self,
    input_shape,
    query_states,
    key_states,
    value_states,
    g_scores,
):
    log_g_scores = torch.log(g_scores)
    attn_output_student = self.flex_attention_compiled(
        query_states,
        key_states,
        value_states,
        score_mod=create_score_mod(log_g_scores, self.config.local_window_size, self.num_key_value_groups),
        block_mask=self.block_mask,
        scale=self.scaling,
        enable_gqa=True,
    ).transpose(1, 2)

    return attn_output_student


def self_attn_forward_patch_train_duo(
    self,
    input_shape,
    query_states,
    key_states,
    value_states,
    g_scores_local,
):
    attn_output_student_global = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        softmax_scale=self.scaling,
        causal=True,
    )

    log_g_scores_local = torch.log(g_scores_local)
    attn_output_student_local = self.flex_attention_compiled(
        query_states,
        key_states,
        value_states,
        score_mod=create_score_mod(log_g_scores_local, self.config.local_window_size, self.num_key_value_groups),
        block_mask=self.block_mask,
        scale=self.scaling,
        enable_gqa=True,
    ).transpose(1, 2)

    alpha = rearrange(
        torch.sigmoid(self.duo_attn_alpha),
        'num_kv_heads -> 1 1 num_kv_heads 1',
    ).repeat_interleave(self.num_key_value_groups, dim=2)

    attn_output_student = alpha * attn_output_student_global + (1 - alpha) * attn_output_student_local

    return attn_output_student


def generate_sparse_indices_ref(
    g_mask: torch.Tensor,
    local_window_size: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    device: torch.device,
    block_size_M: int = 64,
    block_size_N: int = 64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_rows_of_blocks = (seq_len + block_size_M - 1) // block_size_M
    max_blocks_per_row = (local_window_size + block_size_M - 1 + block_size_N - 1) // block_size_N
    max_global_size = g_mask.sum(dim=2).max().item()

    block_count = torch.zeros(1, 1, num_rows_of_blocks, dtype=torch.int32, device=device)
    block_offset = torch.zeros(1, 1, num_rows_of_blocks, max_blocks_per_row, dtype=torch.int32, device=device)
    column_count = torch.zeros(batch_size, num_kv_heads, num_rows_of_blocks, dtype=torch.int32, device=device)
    column_index = torch.zeros(batch_size, num_kv_heads, 1, max_global_size, dtype=torch.int32, device=device)

    for i in range(num_rows_of_blocks):
        start_row = i * block_size_M
        range_start_col = max(0, start_row - local_window_size + 1)
        range_end_col = min(start_row + block_size_M, seq_len)

        b_indices = torch.arange(range_start_col, range_end_col, block_size_N, dtype=torch.int32, device=device)
        num_b_indices = len(b_indices)

        block_count[0, 0, i] = num_b_indices
        block_offset[0, 0, i, :num_b_indices] = b_indices

    for b in range(batch_size):
        for h in range(num_kv_heads):
            v_idx_head = torch.nonzero(g_mask[b][h]).squeeze(1)
            column_index[b, h, 0, :v_idx_head.size(0)] = v_idx_head

            for i in range(num_rows_of_blocks):
                start_row = i * block_size_M
                range_start_col = max(0, start_row - local_window_size + 1)
                
                c_indices = v_idx_head[v_idx_head < range_start_col]
                num_c_indices = len(c_indices)

                column_count[b, h, i] = num_c_indices

    return block_count, block_offset, column_count, column_index


def generate_block_offset(
    local_window_size: int,
    seq_len: int,
    device: torch.device,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows_of_blocks = (seq_len + block_size - 1) // block_size
    num_prologue_blocks = min(local_window_size // block_size, num_rows_of_blocks)
    max_blocks_per_row = num_prologue_blocks + 1

    block_count = torch.zeros(1, 1, num_rows_of_blocks, dtype=torch.int32, device=device)
    block_count[0, 0, :num_prologue_blocks] = torch.arange(1, num_prologue_blocks + 1, dtype=torch.int32, device=device)
    block_count[0, 0, num_prologue_blocks:] = num_prologue_blocks + 1

    block_offset = torch.zeros(1, 1, num_rows_of_blocks, max_blocks_per_row, dtype=torch.int32, device=device)
    prologue_arange = torch.arange(0, block_size * num_prologue_blocks, block_size, dtype=torch.int32, device=device).unsqueeze(0).expand(num_prologue_blocks, -1)
    block_offset[0, 0, :num_prologue_blocks, :num_prologue_blocks] = prologue_arange
    rows_arange = torch.arange(num_rows_of_blocks - num_prologue_blocks, dtype=torch.int32, device=device).unsqueeze(1)
    cols_arange = torch.arange(max_blocks_per_row, dtype=torch.int32, device=device).unsqueeze(0)
    block_offset[0, 0, num_prologue_blocks:] = 1 + (rows_arange + cols_arange) * block_size

    # we need to take special care for the last row
    start_row_last = (num_rows_of_blocks - 1) * block_size
    range_start_col = max(0, start_row_last - local_window_size + 1)
    b_indices = torch.arange(range_start_col, seq_len, block_size, dtype=torch.int32, device=device)
    num_b_indices = len(b_indices)
    block_count[0, 0, -1] = num_b_indices

    return block_count, block_offset


def generate_column_index(
    g_mask: torch.Tensor,
    local_window_size: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    device: torch.device,
    block_size: int,
):
    num_rows_of_blocks = (seq_len + block_size - 1) // block_size
    num_prologue_blocks = min(local_window_size // block_size, num_rows_of_blocks)

    num_chunks = (seq_len - 1) // block_size
    g_mask_trimmed = g_mask[:, :, 1:1 + num_chunks * block_size]
    g_mask_chunk_sum = g_mask_trimmed.reshape(batch_size, num_kv_heads, num_chunks, block_size).sum(dim=3)
    g_mask_chunk_sum = torch.cat((g_mask[:, :, :1], g_mask_chunk_sum), dim=2)
    g_mask_chunk_cumsum = g_mask_chunk_sum.cumsum(dim=2)
    max_global_size = g_mask_chunk_cumsum[:, :, -1].max().item()

    column_count = torch.zeros(batch_size, num_kv_heads, num_rows_of_blocks, dtype=torch.int32, device=device)
    column_count[:, :, :num_prologue_blocks] = 0
    column_count[:, :, num_prologue_blocks:] = g_mask_chunk_cumsum[:, :, :num_rows_of_blocks - num_prologue_blocks]

    column_index = torch.zeros(batch_size, num_kv_heads, 1, max_global_size, dtype=torch.int32, device=device)
    for b in range(batch_size):
        for h in range(num_kv_heads):
            v_idx_head = torch.nonzero(g_mask[b][h]).squeeze(1)[:max_global_size]
            column_index[b, h, 0, :v_idx_head.size(0)] = v_idx_head
    
    return column_count, column_index


sparse_indices_cache = ()
def generate_sparse_indices(
    g_mask: torch.Tensor,
    local_window_size: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    device: torch.device,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    assert local_window_size % block_size == 0

    global sparse_indices_cache
    if sparse_indices_cache and sparse_indices_cache[0] == (local_window_size, seq_len, block_size):
        block_count, block_offset = sparse_indices_cache[1]
    else:
        block_count, block_offset = generate_block_offset(
            local_window_size, seq_len, device, block_size
        )
        sparse_indices_cache = ((local_window_size, seq_len, block_size), (block_count, block_offset))

    column_count, column_index = generate_column_index(
        g_mask, local_window_size, batch_size, num_kv_heads, seq_len, device, block_size
    )

    return block_count, block_offset, column_count, column_index


def generate_sparse_indices_expanded(
    g_mask: torch.Tensor,
    local_window_size: int,
    batch_size: int,
    num_kv_heads: int,
    seq_len: int,
    device: torch.device,
    num_kv_groups: int,
    block_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_count, block_offset, column_count, column_index = generate_sparse_indices(
        g_mask,
        local_window_size,
        batch_size,
        num_kv_heads,
        seq_len,
        device,
        block_size,
    )

    num_heads = num_kv_heads * num_kv_groups
    num_rows_of_blocks = (seq_len + block_size - 1) // block_size

    block_count = block_count.expand(batch_size, num_heads, -1)
    block_offset = block_offset.expand(batch_size, num_heads, -1, -1)
    column_count = column_count.repeat_interleave(num_kv_groups, dim=1)
    column_index = column_index.repeat_interleave(num_kv_groups, dim=1).expand(-1, -1, num_rows_of_blocks, -1)

    block_count = block_count.contiguous()
    block_offset = block_offset.contiguous()
    column_count = column_count.contiguous()
    column_index = column_index.contiguous()

    return block_count, block_offset, column_count, column_index


def sparse_attn_func_wrapper(*args, **kwargs):
    return sparse_attn_func(*args, **kwargs)


def quest_select_blocks(
    self,
    query_states,
    past_key_value,
    local_blocks_per_head,
    device,
    local_tokens_total,
):
    """
    Args:
        query_states: Tensor of shape (batch=1, num_heads, q_len=1, head_dim) for the current token.
        past_key_value: KV cache structure containing block metadata.
        local_blocks_per_head: Number of local blocks reserved per head.
        device: Torch device for allocation.
        local_tokens_total: Number of local tokens currently stored for this layer.

    Returns:
        (cache_seqlens, block_table)
            cache_seqlens: Tensor of shape (1, num_kv_heads) with total token counts per head.
            block_table: Tensor of shape (1, num_kv_heads, max_selected_blocks) containing selected block ids.
    """
    queries = query_states[0, :, 0, :].reshape(self.config.num_key_value_heads, self.num_key_value_groups, self.head_dim)

    local_full_blocks, local_remainder = divmod(local_tokens_total, self.config.block_size)
    local_block_count = local_full_blocks + (1 if local_remainder > 0 else 0)

    assert self.config.quest_token_budget is not None
    remaining_budget_tokens = max(self.config.quest_token_budget - local_tokens_total, 0)
    budget_blocks = (remaining_budget_tokens + self.config.block_size - 1) // self.config.block_size

    tokens_selected_per_head: list[int] = []
    blocks_selected_per_head: list[torch.Tensor] = []
    max_selected_blocks = 0

    for h in range(self.config.num_key_value_heads):
        head_blocks = past_key_value.block_table[self.layer_idx, h]

        local_blocks_tensor = head_blocks[:local_block_count]

        global_tokens_total = past_key_value.global_lengths[self.layer_idx, h].item()
        global_full_blocks, global_remainder = divmod(global_tokens_total, self.config.block_size)
        global_block_count = global_full_blocks + (1 if global_remainder > 0 else 0)
        global_blocks_tensor = head_blocks[local_blocks_per_head:local_blocks_per_head + global_block_count]

        capacity = min(budget_blocks, global_block_count)

        partial_selected = False
        selected_global_count = 0
        ordered_global_tensor = None

        if capacity > 0:
            scored_key_min = past_key_value.quest_block_key_min[global_blocks_tensor]
            scored_key_max = past_key_value.quest_block_key_max[global_blocks_tensor]

            query_expanded = queries[h].unsqueeze(1)  # (num_kv_groups, 1, head_dim)
            selected_keys = torch.where(
                query_expanded >= 0,
                scored_key_max.unsqueeze(0),  # (1, num_blocks, head_dim)
                scored_key_min.unsqueeze(0),  # (1, num_blocks, head_dim)
            )
            block_scores = (query_expanded * selected_keys).sum(dim=-1)  # (num_kv_groups, num_blocks)
            best_scores = torch.max(block_scores, dim=0).values  # (num_blocks,)

            include_partial = global_remainder > 0
            full_capacity = capacity - (1 if include_partial else 0)

            if full_capacity > 0 and global_full_blocks > 0:
                if full_capacity >= global_full_blocks:
                    selected_positions = torch.arange(global_full_blocks, dtype=torch.int32, device=device)
                else:
                    full_scores = best_scores[:global_full_blocks]
                    selected_positions = torch.topk(full_scores, full_capacity).indices.to(torch.int32)
            else:
                selected_positions = torch.empty(0, dtype=torch.int32, device=device)

            if include_partial:  # Always keep trailing partial block when capacity allows
                partial_index = torch.tensor([global_full_blocks], dtype=torch.int32, device=device)
                selected_positions = torch.cat([selected_positions, partial_index])
                partial_selected = True

            selected_global_count = selected_positions.numel()
            ordered_global_tensor = global_blocks_tensor[selected_positions]

        has_local_blocks = local_block_count > 0
        has_global_blocks = selected_global_count > 0

        if not (has_local_blocks or has_global_blocks):
            raise RuntimeError(
                f"QUEST expected at least one block for layer {self.layer_idx} head {h}, "
                "but both local and global selections were empty."
            )

        if has_local_blocks and has_global_blocks:
            selected_blocks_tensor = torch.cat([local_blocks_tensor, ordered_global_tensor])
        elif has_local_blocks:
            selected_blocks_tensor = local_blocks_tensor
        else:
            selected_blocks_tensor = ordered_global_tensor

        global_tokens_selected = selected_global_count * self.config.block_size
        if partial_selected:
            global_tokens_selected += global_remainder - self.config.block_size

        total_tokens_selected = local_tokens_total + global_tokens_selected

        tokens_selected_per_head.append(total_tokens_selected)
        blocks_selected_per_head.append(selected_blocks_tensor)
        max_selected_blocks = max(max_selected_blocks, selected_blocks_tensor.numel())

    cache_seqlens = torch.empty(
        1, self.config.num_key_value_heads,
        dtype=torch.int32, device=device,
    )
    block_table = torch.zeros(
        1, self.config.num_key_value_heads, max_selected_blocks,
        dtype=torch.int32, device=device,
    )

    for h, (total_tokens, block_tensor) in enumerate(zip(tokens_selected_per_head, blocks_selected_per_head)):
        cache_seqlens[0, h] = total_tokens
        block_count = block_tensor.numel()
        if block_count > 0:
            block_table[0, h, :block_count] = block_tensor

    return cache_seqlens, block_table


def snapkv_record_queries(self, past_key_value, query_states):
    q_len = query_states.shape[2]
    if q_len > 1:
        take = min(self.config.snapkv_window_size, q_len)
        tail = query_states[0, :, -take:, :].transpose(0, 1)  # (take, num_heads, head_dim)
        past_key_value.snapkv_query_buffer[self.layer_idx, :take] = tail
        if self.layer_idx == self.config.num_hidden_layers - 1:
            past_key_value.snapkv_query_length = take
            past_key_value.snapkv_query_ptr = take % self.config.snapkv_window_size
    else:
        past_key_value.snapkv_query_buffer[self.layer_idx, past_key_value.snapkv_query_ptr] = query_states[0, :, 0, :]
        if self.layer_idx == self.config.num_hidden_layers - 1:
            past_key_value.snapkv_query_length = min(self.config.snapkv_window_size, past_key_value.snapkv_query_length + 1)
            past_key_value.snapkv_query_ptr = (past_key_value.snapkv_query_ptr + 1) % self.config.snapkv_window_size


def snapkv_evict(self, past_key_value, device, local_blocks_per_head):
    k_cache = past_key_value.block_pool_k.view(past_key_value.total_blocks, self.config.block_size, self.head_dim)
    assert past_key_value.snapkv_query_length == self.config.snapkv_window_size

    for layer_idx in range(self.config.num_hidden_layers):
        queries = rearrange(
            past_key_value.snapkv_query_buffer[layer_idx],
            'window (num_kv_heads num_kv_groups) head_dim -> num_kv_heads num_kv_groups window head_dim',
            num_kv_heads=self.config.num_key_value_heads,
            num_kv_groups=self.num_key_value_groups,
        )

        for h in range(self.config.num_key_value_heads):
            global_len = past_key_value.global_lengths[layer_idx, h].item()
            num_evict = int(global_len * self.config.snapkv_evict_ratio)
            if num_evict == 0:
                continue

            global_block_count = (global_len + self.config.block_size - 1) // self.config.block_size
            global_block_ids = past_key_value.block_table[
                layer_idx,
                h,
                local_blocks_per_head : local_blocks_per_head + global_block_count,
            ]
            keys = k_cache[global_block_ids].reshape(-1, self.head_dim)[:global_len]

            attn = torch.matmul(queries[h], keys.transpose(0, 1)) * self.scaling
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32)

            head_scores = torch.nn.functional.max_pool1d(
                attn.amax(dim=0).mean(dim=0).view(1, 1, -1),
                kernel_size=self.config.snapkv_kernel_size,
                stride=1,
                padding=self.config.snapkv_kernel_size // 2,
            ).view(-1)

            evict_positions = torch.topk(head_scores, k=num_evict, largest=False).indices

            keep_mask = torch.ones(global_len, dtype=torch.bool, device=device)
            keep_mask[evict_positions] = False
            keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)

            new_global_len = keep_indices.numel()
            new_block_count = (new_global_len + self.config.block_size - 1) // self.config.block_size

            # Pack kept tokens into the front blocks
            if new_global_len > 0:
                block_offsets = torch.arange(self.config.block_size, dtype=torch.int32, device=device)
                global_linear = (
                    global_block_ids[:, None] * self.config.block_size + block_offsets[None, :]
                ).reshape(-1)[:global_len]

                kept_linear = global_linear[keep_indices]
                kept_k = past_key_value.block_pool_k[kept_linear]
                kept_v = past_key_value.block_pool_v[kept_linear]

                dest_block_ids = global_block_ids[:new_block_count]
                dest_linear = (
                    dest_block_ids[:, None] * self.config.block_size + block_offsets[None, :]
                ).reshape(-1)[:new_global_len]

                past_key_value.block_pool_k[dest_linear] = kept_k
                past_key_value.block_pool_v[dest_linear] = kept_v

            # Release extra blocks and reset unused block_table slots
            if new_block_count < global_block_count:
                freed_block_ids = global_block_ids[new_block_count:global_block_count].tolist()

                past_key_value.block_table[
                    layer_idx,
                    h,
                    local_blocks_per_head + new_block_count : local_blocks_per_head + global_block_count,
                ] = past_key_value.total_blocks - 1  # sentinel
                past_key_value.free_block_ids.extend(freed_block_ids)

            # Update per-head global length after compaction
            past_key_value.global_lengths[layer_idx, h] = new_global_len


def migrate_local_kv_to_global(self, past_key_value, device, local_blocks_per_head):
    layer_indices, head_indices = torch.nonzero(
        past_key_value.local_is_global[:, :, past_key_value.local_ptr],
        as_tuple=True,
    )
    if layer_indices.numel() == 0:
        return

    global_lengths = past_key_value.global_lengths[layer_indices, head_indices]
    overflow_mask = (self.config.local_window_size + global_lengths + 1) > self.config.max_tokens_per_head
    assert not torch.any(overflow_mask).item(), (
        f"Tokens exceed max_tokens_per_head ({self.config.max_tokens_per_head})"
    )

    global_block_idx = local_blocks_per_head + torch.div(global_lengths, self.config.block_size, rounding_mode='floor')
    global_offset = torch.remainder(global_lengths, self.config.block_size)
    need_new_block = global_offset == 0
    num_new_blocks = need_new_block.sum().item()

    if num_new_blocks > 0:
        if self.config.snapkv_enabled:
            num_from_free = min(num_new_blocks, len(past_key_value.free_block_ids))
            block_ids = past_key_value.free_block_ids[-num_from_free:]
            if num_from_free:
                past_key_value.free_block_ids = past_key_value.free_block_ids[:-num_from_free]

            next_free_block = past_key_value.next_free_block
            new_block_end = next_free_block + (num_new_blocks - num_from_free)
            assert new_block_end <= past_key_value.total_blocks, (
                f"Allocation exceeds total_blocks ({past_key_value.total_blocks})"
            )

            new_ids = range(next_free_block, new_block_end)
            block_ids.extend(new_ids)

            new_block_ids = torch.tensor(
                block_ids,
                dtype=torch.int32, device=device
            )
        else:
            next_free_block = past_key_value.next_free_block
            new_block_end = next_free_block + num_new_blocks
            assert new_block_end <= past_key_value.total_blocks, (
                f"Allocation exceeds total_blocks ({past_key_value.total_blocks})"
            )

            new_block_ids = torch.arange(
                next_free_block, new_block_end,
                dtype=torch.int32, device=device,
            )

        past_key_value.block_table[
            layer_indices[need_new_block],
            head_indices[need_new_block],
            global_block_idx[need_new_block]
        ] = new_block_ids
        past_key_value.next_free_block = new_block_end

    local_region_starts = past_key_value.local_region_starts[layer_indices, head_indices]
    replace_linear_indices = local_region_starts + past_key_value.local_ptr

    block_indices = global_block_idx
    physical_indices = past_key_value.block_table[layer_indices, head_indices, block_indices]
    dest_linear_indices = physical_indices * self.config.block_size + global_offset

    gathered_k = past_key_value.block_pool_k[replace_linear_indices]
    gathered_v = past_key_value.block_pool_v[replace_linear_indices]
    past_key_value.block_pool_k[dest_linear_indices] = gathered_k
    past_key_value.block_pool_v[dest_linear_indices] = gathered_v

    if self.config.use_quest:
        quest_block_min = past_key_value.quest_block_key_min[physical_indices]
        quest_block_max = past_key_value.quest_block_key_max[physical_indices]
        new_block_mask = need_new_block.unsqueeze(-1)

        quest_block_min = torch.where(
            new_block_mask,
            gathered_k,
            torch.minimum(quest_block_min, gathered_k),
        )
        quest_block_max = torch.where(
            new_block_mask,
            gathered_k,
            torch.maximum(quest_block_max, gathered_k),
        )

        past_key_value.quest_block_key_min[physical_indices] = quest_block_min
        past_key_value.quest_block_key_max[physical_indices] = quest_block_max

    past_key_value.global_lengths[layer_indices, head_indices] = global_lengths + 1


def flash_attn_with_kvcache_wrapper(
    q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale
):
    """
    Args:
        q: Query tensor of shape (batch_size, 1, num_query_heads, head_size)
        k_cache: KV cache for keys of shape (num_blocks, block_size, head_size)
        v_cache: KV cache for values of shape (num_blocks, block_size, head_size)
        cache_seqlens: Sequence lengths of shape (batch_size, num_kv_heads)
        block_table: Block table mapping of shape (batch_size, num_kv_heads, max_num_blocks_per_head)
        softmax_scale: Scaling factor for attention scores

    Returns:
        Output tensor of shape (batch_size, 1, num_query_heads, head_size)
    """
    batch_size, _, num_query_heads, _ = q.shape
    _, num_kv_heads = cache_seqlens.shape
    num_kv_groups = num_query_heads // num_kv_heads

    q = rearrange(
        q,
        'batch_size 1 (num_kv_heads num_kv_groups) head_size -> (batch_size num_kv_heads) 1 num_kv_groups head_size',
        num_kv_heads=num_kv_heads,
        num_kv_groups=num_kv_groups
    )
    k_cache = rearrange(k_cache, 'num_blocks block_size head_size -> num_blocks block_size 1 head_size')
    v_cache = rearrange(v_cache, 'num_blocks block_size head_size -> num_blocks block_size 1 head_size')
    cache_seqlens = rearrange(cache_seqlens, 'batch_size num_kv_heads -> (batch_size num_kv_heads)')
    block_table = rearrange(block_table, 'batch_size num_kv_heads max_num_blocks_per_head -> (batch_size num_kv_heads) max_num_blocks_per_head')

    out = flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        softmax_scale=softmax_scale,
        causal=True,
        num_splits=2,
    )

    return rearrange(
        out,
        '(batch_size num_kv_heads) 1 num_kv_groups head_size -> batch_size 1 (num_kv_heads num_kv_groups) head_size',
        batch_size=batch_size,
        num_kv_heads=num_kv_heads
    )


def prefill_wrapper(
    self,
    past_key_value,
    query_states,
    key_states,
    value_states,
    batch_size,
    q_seq_len,
    device,
    g_mask,
    local_blocks_per_head,
):
    if self.layer_idx == 0:
        assert not hasattr(past_key_value, 'gated_kv_initialized')

        total_local_blocks = self.config.num_hidden_layers * self.config.num_key_value_heads * local_blocks_per_head
        total_blocks = (self.config.max_total_tokens + self.config.block_size - 1) // self.config.block_size
        max_blocks_per_head = (self.config.max_tokens_per_head + self.config.block_size - 1) // self.config.block_size

        # These variables stay on GPU
        sentinel_block = total_blocks - 1
        past_key_value.block_table = torch.full(
            (self.config.num_hidden_layers, self.config.num_key_value_heads, max_blocks_per_head),
            fill_value=sentinel_block, dtype=torch.int32, device=device
        )
        past_key_value.block_pool_k = torch.empty(
            total_blocks * self.config.block_size, self.head_dim,
            dtype=key_states.dtype, device=device
        )
        past_key_value.block_pool_v = torch.empty(
            total_blocks * self.config.block_size, self.head_dim,
            dtype=value_states.dtype, device=device
        )
        past_key_value.local_is_global = torch.zeros(
            self.config.num_hidden_layers, self.config.num_key_value_heads, self.config.local_window_size,
            dtype=torch.bool, device=device
        )
        past_key_value.global_lengths = torch.zeros(
            self.config.num_hidden_layers, self.config.num_key_value_heads,
            dtype=torch.int32, device=device
        )

        # These scalars stay on CPU
        past_key_value.local_ptr = 0
        past_key_value.local_length = 0
        past_key_value.next_free_block = total_local_blocks

        # These variables stay on GPU (only if QUEST is used)
        if self.config.use_quest:
            past_key_value.quest_block_key_min = torch.empty(
                total_blocks, self.head_dim,
                dtype=key_states.dtype, device=device
            )
            past_key_value.quest_block_key_max = torch.empty(
                total_blocks, self.head_dim,
                dtype=key_states.dtype, device=device
            )

        # These variables stay on CPU and GPU (only if SnapKV is used)
        if self.config.snapkv_enabled:
            assert not self.config.use_quest
            assert self.config.snapkv_max_cached_tokens > 0
            assert 0.0 < self.config.snapkv_evict_ratio <= 1.0
            assert self.config.snapkv_kernel_size % 2 == 1

            past_key_value.snapkv_query_buffer = torch.empty(
                self.config.num_hidden_layers, self.config.snapkv_window_size, self.config.num_attention_heads, self.head_dim,
                dtype=query_states.dtype, device=device
            )
            past_key_value.snapkv_query_length = 0
            past_key_value.snapkv_query_ptr = 0
            past_key_value.snapkv_evict_count = 0
            past_key_value.free_block_ids = []

        # Allocate fixed local blocks for each head
        local_blocks = torch.arange(total_local_blocks, dtype=torch.int32, device=device)
        local_blocks = local_blocks.reshape(
            self.config.num_hidden_layers,
            self.config.num_key_value_heads,
            local_blocks_per_head
        )
        past_key_value.block_table[:, :, :local_blocks_per_head] = local_blocks

        # These variables stay on GPU and are not changed after initialization
        layer_offsets = torch.arange(
            self.config.num_hidden_layers,
            dtype=torch.int32, device=device
        )
        head_offsets = torch.arange(
            self.config.num_key_value_heads,
            dtype=torch.int32, device=device
        )
        past_key_value.local_region_starts = (
            (layer_offsets[:, None] * self.config.num_key_value_heads + head_offsets[None, :]) * self.config.local_window_size
        )

        # These variables stay on CPU and are not changed after initialization
        past_key_value.total_blocks = total_blocks
        past_key_value.gated_kv_initialized = True

    # Prefilling
    # 1. Write local cache
    layer_local_start = self.layer_idx * self.config.num_key_value_heads * self.config.local_window_size
    layer_local_span = self.config.num_key_value_heads * self.config.local_window_size
    layer_local_slice = slice(layer_local_start, layer_local_start + layer_local_span)

    local_block_pool_k = past_key_value.block_pool_k[layer_local_slice].view(
        self.config.num_key_value_heads, self.config.local_window_size, self.head_dim
    )
    local_block_pool_v = past_key_value.block_pool_v[layer_local_slice].view(
        self.config.num_key_value_heads, self.config.local_window_size, self.head_dim
    )
    assert local_block_pool_k.untyped_storage().data_ptr() == past_key_value.block_pool_k.untyped_storage().data_ptr()
    assert local_block_pool_v.untyped_storage().data_ptr() == past_key_value.block_pool_v.untyped_storage().data_ptr()

    num_local_tokens = min(q_seq_len, self.config.local_window_size)
    local_block_pool_k[:, :num_local_tokens] = key_states[0, :, -num_local_tokens:]
    local_block_pool_v[:, :num_local_tokens] = value_states[0, :, -num_local_tokens:]
    past_key_value.local_is_global[self.layer_idx, :, :num_local_tokens] = g_mask[0, :, -num_local_tokens:]
    past_key_value.local_length = num_local_tokens

    # 2. Write global cache
    for h in range(self.config.num_key_value_heads):
        num_earlier_tokens = q_seq_len - self.config.local_window_size
        if num_earlier_tokens > 0:
            earlier_g_mask = g_mask[0, h, :num_earlier_tokens]
            global_token_indices = torch.where(earlier_g_mask)[0]
            num_global_tokens = len(global_token_indices)

            if num_global_tokens > 0:
                # Check per-head capacity
                total_tokens_this_head = num_local_tokens + num_global_tokens
                assert total_tokens_this_head <= self.config.max_tokens_per_head, \
                    f"Layer {self.layer_idx} head {h}: total tokens ({total_tokens_this_head}) exceeds max_tokens_per_head ({self.config.max_tokens_per_head})"

                num_blocks_needed = (num_global_tokens + self.config.block_size - 1) // self.config.block_size
                first_global_block = past_key_value.next_free_block

                # Check total blocks capacity
                assert first_global_block + num_blocks_needed <= past_key_value.total_blocks, \
                    f"Layer {self.layer_idx} head {h}: total blocks needed ({first_global_block + num_blocks_needed}) exceeds allocated total_blocks ({past_key_value.total_blocks})"

                linear_start = first_global_block * self.config.block_size
                linear_end = linear_start + num_global_tokens
                global_block_indices = torch.arange(first_global_block, first_global_block + num_blocks_needed, dtype=torch.int32, device=device)

                past_key_value.block_table[self.layer_idx, h, local_blocks_per_head:local_blocks_per_head + num_blocks_needed] = global_block_indices
                past_key_value.block_pool_k[linear_start:linear_end] = key_states[0, h, global_token_indices]
                past_key_value.block_pool_v[linear_start:linear_end] = value_states[0, h, global_token_indices]
                past_key_value.global_lengths[self.layer_idx, h] = num_global_tokens
                past_key_value.next_free_block += num_blocks_needed

                if self.config.use_quest:
                    full_blocks, remainder = divmod(num_global_tokens, self.config.block_size)
                    k_cache = past_key_value.block_pool_k.view(past_key_value.total_blocks, self.config.block_size, self.head_dim)

                    if full_blocks > 0:
                        full_block_indices = global_block_indices[:full_blocks]
                        full_tokens = k_cache[full_block_indices]
                        past_key_value.quest_block_key_min[full_block_indices] = full_tokens.amin(dim=1)
                        past_key_value.quest_block_key_max[full_block_indices] = full_tokens.amax(dim=1)

                    if remainder > 0:
                        partial_block_idx = global_block_indices[full_blocks]
                        partial_tokens = k_cache[partial_block_idx, :remainder]
                        past_key_value.quest_block_key_min[partial_block_idx] = partial_tokens.amin(dim=0)
                        past_key_value.quest_block_key_max[partial_block_idx] = partial_tokens.amax(dim=0)

    if self.config.use_baseline:
        attn_output = flash_attn_with_kvcache(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            softmax_scale=self.scaling,
            causal=True,
        )
    else:
        block_count, block_offset, column_count, column_index = generate_sparse_indices(
            g_mask,
            self.config.local_window_size,
            batch_size,
            self.config.num_key_value_heads,
            q_seq_len,
            device,
        )
        attn_output = self.prefill_kernel(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            block_count, block_offset, column_count, column_index,
            softmax_scale=self.scaling,
            causal=True,
            return_softmax_lse=False
        )

    return attn_output


def decode_wrapper(
    self,
    past_key_value,
    query_states,
    key_states,
    value_states,
    device,
    g_mask,
    local_blocks_per_head,
):
    if past_key_value.local_length < self.config.local_window_size:
        insert_pos = past_key_value.local_length
        current_local_len = past_key_value.local_length + 1
    else:
        insert_pos = past_key_value.local_ptr
        current_local_len = self.config.local_window_size

    local_region_starts = past_key_value.local_region_starts[self.layer_idx]
    local_linear_indices = local_region_starts + insert_pos
    past_key_value.block_pool_k[local_linear_indices] = key_states[0, :, 0]
    past_key_value.block_pool_v[local_linear_indices] = value_states[0, :, 0]
    past_key_value.local_is_global[self.layer_idx, :, insert_pos] = g_mask[0, :, 0]

    if self.layer_idx == self.config.num_hidden_layers - 1:
        if past_key_value.local_length < self.config.local_window_size:
            past_key_value.local_length = current_local_len
        else:
            past_key_value.local_ptr = (past_key_value.local_ptr + 1) % self.config.local_window_size

    # Reshape block_pool from flattened (total_blocks * block_size, head_dim) to (total_blocks, block_size, head_dim)
    k_cache = past_key_value.block_pool_k.view(past_key_value.total_blocks, self.config.block_size, self.head_dim)
    v_cache = past_key_value.block_pool_v.view(past_key_value.total_blocks, self.config.block_size, self.head_dim)

    if self.config.use_quest:
        cache_seqlens, block_table = quest_select_blocks(
            self,
            query_states,
            past_key_value,
            local_blocks_per_head,
            device,
            current_local_len,
        )
    else:
        cache_seqlens = rearrange(
            current_local_len + past_key_value.global_lengths[self.layer_idx],
            'num_kv_heads -> 1 num_kv_heads'
        )
        block_table = rearrange(
            past_key_value.block_table[self.layer_idx],
            "num_kv_heads max_blocks_per_head -> 1 num_kv_heads max_blocks_per_head"
        )

    query_states = query_states.transpose(1, 2)  # (batch_size=1, q_seqlen=1, num_heads, head_dim)
    attn_output = self.decode_kernel(
        query_states,
        k_cache,                       # (total_blocks, block_size, head_dim)
        v_cache,                       # (total_blocks, block_size, head_dim)
        cache_seqlens,                 # (batch_size=1, num_kv_heads)
        block_table,                   # (batch_size=1, num_kv_heads, max_blocks_per_head)
        self.scaling,
    )

    return attn_output


def self_attn_forward_patch_inference(
    self,
    past_key_value,
    input_shape,
    query_states,
    key_states,
    value_states,
    g_scores,
):
    batch_size, q_seq_len = input_shape
    device = query_states.device
    is_prefilling = q_seq_len > 1
    g_mask = g_scores > self.config.g_threshold

    assert self.config.local_window_size % self.config.block_size == 0
    local_blocks_per_head = self.config.local_window_size // self.config.block_size

    if (not is_prefilling) and self.layer_idx == 0 and past_key_value.local_length == self.config.local_window_size:
        migrate_local_kv_to_global(self, past_key_value, device, local_blocks_per_head)

    if is_prefilling:
        attn_output = self.prefill_wrapper(
            self,
            past_key_value,
            query_states,
            key_states,
            value_states,
            batch_size,
            q_seq_len,
            device,
            g_mask,
            local_blocks_per_head,
        )
    else:
        attn_output = self.decode_wrapper(
            self,
            past_key_value,
            query_states,
            key_states,
            value_states,
            device,
            g_mask,
            local_blocks_per_head,
        )

    if self.config.snapkv_enabled:
        snapkv_record_queries(self, past_key_value, query_states)
        if (
            self.layer_idx == self.config.num_hidden_layers - 1 and
            past_key_value.global_lengths.sum().item() > self.config.snapkv_max_cached_tokens
        ):
            past_key_value.snapkv_evict_count += 1
            snapkv_evict(self, past_key_value, device, local_blocks_per_head)

    return attn_output


def self_attn_forward_patch(
    self,
    gating_mode,
    attention_mask,
    past_key_value,
    input_shape,
    query_states,
    pre_key,
    key_states,
    value_states,
):
    batch_size, _ = input_shape
    device = query_states.device
    assert (not hasattr(self, "sliding_window")) or self.sliding_window is None

    kv_seq_len = key_states.size(2)
    if self.config.use_baseline:
        g_scores = torch.ones(
            batch_size, self.config.num_key_value_heads, kv_seq_len,
            dtype=torch.bfloat16, device=device
        )
    elif self.config.use_duo_attn:
        g_scores = repeat(
            torch.nn.functional.sigmoid(self.duo_attn_alpha),
            'num_kv_heads -> batch_size num_kv_heads kv_seq_len',
            batch_size=batch_size,
            kv_seq_len=kv_seq_len,
        )
        if kv_seq_len > 1:
            sink_mask = torch.zeros(kv_seq_len, dtype=torch.bool, device=device)
            sink_mask[:self.config.duo_attn_sink_size] = True
            sink_mask = repeat(
                sink_mask,
                'kv_seq_len -> batch_size num_kv_heads kv_seq_len',
                batch_size=batch_size,
                num_kv_heads=self.config.num_key_value_heads,
            )
            g_scores = torch.where(sink_mask, 1.0, g_scores)
    else:
        if self.config.g_expand != 0.0:
            g_scores = self.g_predictors(pre_key, key_states, value_states) # shape: (batch_size, num_key_value_heads, kv_seq_len)

        if self.config.random_sparsity is not None:
            random_mask = torch.rand(
                batch_size, self.config.num_key_value_heads, kv_seq_len,
                dtype=torch.bfloat16, device=device
            )
            g_scores = (random_mask >= self.config.random_sparsity).to(torch.bfloat16)
    
    if getattr(self, "use_hard_threshold", False):
        g_scores_dtype = g_scores.dtype
        g_scores = (g_scores > self.config.g_threshold).to(g_scores_dtype)
    elif self.config.g_ungate_count > 0:
        rand = torch.rand_like(g_scores)
        drop_idx = torch.topk(rand, k=self.config.g_ungate_count, dim=-1).indices
        drop_mask = torch.zeros_like(g_scores, dtype=torch.bool).scatter(-1, drop_idx, True)
        g_scores = g_scores.masked_fill(drop_mask, 1.0)

    if gating_mode == 1:
        if self.config.use_duo_attn:
            g_scores_local = torch.where(sink_mask, 1.0, torch.zeros_like(g_scores))
            attn_output_student = self_attn_forward_patch_train_duo(
                self,
                input_shape,
                query_states,
                key_states,
                value_states,
                g_scores_local,
            )
        else:
            attn_output_student = self_attn_forward_patch_train(
                self,
                input_shape,
                query_states,
                key_states,
                value_states,
                g_scores,
            )
        return attn_output_student, None, g_scores
    elif gating_mode == 3: 
        assert batch_size == 1
        assert attention_mask is None
        assert past_key_value is not None
        attn_output = self_attn_forward_patch_inference(
            self,
            past_key_value,
            input_shape,
            query_states,
            key_states,
            value_states,
            g_scores,
        )
        return attn_output, None, None
    else:
        assert False


def set_duo_attn_alpha(model, attn_heads):
    assert attn_heads.shape == (model.config.num_hidden_layers, model.config.num_key_value_heads)
    for layer_idx in range(model.config.num_hidden_layers):
        for head_idx in range(model.config.num_key_value_heads):
            if attn_heads[layer_idx, head_idx] == 1.0:
                duo_attn_alpha = torch.inf
            elif attn_heads[layer_idx, head_idx] == 0.0:
                duo_attn_alpha = -torch.inf
            else:
                assert False
            model.model.layers[layer_idx].self_attn.duo_attn_alpha.data[head_idx] = duo_attn_alpha
