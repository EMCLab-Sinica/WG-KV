# Porting WG-KV to Other Model Architectures

This guide explains how to add support for a model architecture other than the Llama and Qwen3 families that ship with this repository.

## Overview

WG-KV is implemented as a patch on top of [🤗 Transformers](https://github.com/huggingface/transformers) `v4.53.3`. The code is split into two parts:

| Part | Files | Needs porting? |
| --- | --- | --- |
| Core logic — write gate, dual-cache paged memory manager, sparse prefill/decode kernels | `src/modeling_layers.py` | **No.** It is fully architecture-agnostic and contains no model-specific code. |
| Model plumbing — expose the gate inside the attention module and thread `gating_mode` through the forward pass | `src/modeling_<model>.py`, `src/configuration_<model>.py` | **Yes.** One patched copy per architecture. |

The scripts in `scripts/` are also architecture-agnostic: they load models through `AutoConfig` / `AutoModelForCausalLM`, so a new architecture only requires passing `--model_name`.

Porting is therefore a mechanical process: copy the two upstream files for your architecture, apply the patches below, and register them in `setup_venv.sh`.

## Prerequisites

Verify that the target architecture satisfies the following assumptions before starting. These are enforced by `src/modeling_layers.py` and violating them requires kernel-level changes, not just plumbing.

* **Standard MHA / GQA with per-head K/V.** The paged cache stores `head_dim`-sized keys and values per KV head. Multi-head latent attention (e.g. DeepSeek MLA) and cross-layer KV sharing are not supported.
* **Uniform head geometry across layers.** The block table is allocated as `num_hidden_layers × num_key_value_heads × max_blocks_per_head`, so `head_dim` and `num_key_value_heads` must be identical in every layer.
* **RoPE.** WG-KV's model plumbing captures both pre-rotary and post-rotary keys (the default `g_inputs=["pre_k", "post_k"]`), and the porting procedure below assumes that the architecture exposes rotary position embeddings.
* **Standard module layout** `model.model.layers[i].self_attn`, with the attention module exposing `config`, `layer_idx`, `head_dim`, `scaling`, and `num_key_value_groups`. This holds for essentially every decoder-only model in Transformers.

## Step 1 — Copy the upstream files

Take the two files for your architecture from the exact Transformers version pinned in `requirements.txt`:

```bash
MODEL=mistral  # the transformers model directory name

curl -sSL -o src/modeling_${MODEL}.py \
  https://raw.githubusercontent.com/huggingface/transformers/v4.53.3/src/transformers/models/${MODEL}/modeling_${MODEL}.py
curl -sSL -o src/configuration_${MODEL}.py \
  https://raw.githubusercontent.com/huggingface/transformers/v4.53.3/src/transformers/models/${MODEL}/configuration_${MODEL}.py
```

Use the generated `modeling_*.py`, not the `modular_*.py` source. If the generated file imports its attention or decoder layer from another architecture, patch that architecture instead.

## Step 2 — Patch the configuration

In `<Model>Config.__init__`, add the WG-KV keyword arguments immediately before `**kwargs`, and the matching assignments in the body. The values below match `src/configuration_llama.py` and are starting points: `fineweb_bins` is tokenizer-specific, while gate and sparsity settings may need retuning for the target model and workload. For example, Qwen3 uses `g_expand=1.5` instead of `4.0`.

```python
        fineweb_bins="WG-KV/fineweb-<model>-bins",  # tokenizer-specific, see Step 5
        checkpointing_threshold=0,
        local_window_size=256,
        g_inputs=["pre_k", "post_k"],
        g_act_fn="gelu",
        g_expand=4.0,
        g_epsilon=1e-3,
        g_rms=True,
        g_threshold=0.1,
        g_ungate_count=0,
        use_duo_attn=False,
        duo_attn_sink_size=128,
        use_adaea=False,
        adaea_sink_size=4,
        use_baseline=False,
        block_size=16,
        max_total_tokens=None,
        max_tokens_per_head=None,
        use_quest=False,
        quest_token_budget=None,
        dump_tensors=False,
        random_sparsity=None,
        g_fast_path=False,
        g_defer=False,
        snapkv_enabled=False,
        snapkv_max_cached_tokens=-1,
        snapkv_evict_ratio=0.0,
        snapkv_window_size=256,
        snapkv_kernel_size=5,
        use_triton_kernel=False,
        **kwargs,
```

## Step 3 — Patch the modeling file

Seven edits, all of which mirror `src/modeling_llama.py` and `src/modeling_qwen3.py` line for line. Diff either file against its upstream original if you prefer to read the patch directly:

```bash
diff -u <(curl -sSL https://raw.githubusercontent.com/huggingface/transformers/v4.53.3/src/transformers/models/llama/modeling_llama.py) src/modeling_llama.py
```

### 3.1 Imports

```diff
-from ...modeling_layers import GradientCheckpointingLayer
+from ...modeling_layers import (
+    GatingPredictor,
+    self_attn_forward_patch,
+    DistillationCausalLMOutputWithPast,
+    prefill_wrapper,
+    decode_wrapper,
+    sparse_attn_func_wrapper,
+    flash_attn_with_kvcache_wrapper_handle,
+    flash_attn_with_kvcache_wrapper_triton_handle,
+)
```

### 3.2 `<Model>Attention.__init__` — instantiate the gate and bind the kernels

Append to the end of `__init__`:

```python
        if config.use_baseline:
            pass
        elif config.use_duo_attn:
            self.duo_attn_alpha = nn.Parameter(torch.empty(config.num_key_value_heads))
        elif config.use_adaea:
            self.adaea_threshold = nn.Parameter(torch.empty(config.num_attention_heads))
            self.adaea_mean_query = nn.Parameter(torch.empty(config.num_attention_heads, self.head_dim))
            self.adaea_cov_query = nn.Parameter(torch.empty(config.num_attention_heads, self.head_dim, self.head_dim))
        elif config.g_expand != 0.0:
            self.g_predictors = GatingPredictor(self.head_dim, config.num_key_value_heads, config.g_inputs, config.g_act_fn, config.g_expand, config.g_epsilon, config.g_rms, config.g_fast_path)

        self.self_attn_forward_patch = self_attn_forward_patch
        self.prefill_wrapper = prefill_wrapper
        self.decode_wrapper = decode_wrapper
        self.prefill_kernel = sparse_attn_func_wrapper

        if config.use_triton_kernel:
            self.decode_kernel_handle = flash_attn_with_kvcache_wrapper_triton_handle
        else:
            self.decode_kernel_handle = flash_attn_with_kvcache_wrapper_handle

        self.decode_kernel_executor = lambda func: func()
```

`GatingPredictor` is the Write-Gate MLP: a two-layer grouped 1×1 convolution over the per-head key (and optionally value) states. `g_expand` controls its hidden width.

### 3.3 `<Model>Attention.forward` — capture pre-RoPE keys and dispatch to the gate

Add `gating_mode: bool` as the **first** parameter, then:

```diff
         value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
 
+        pre_key = key_states if (not self.config.use_duo_attn) and 'pre_k' in self.config.g_inputs else None
         cos, sin = position_embeddings
         query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
 
-        if past_key_value is not None:
+        if not (gating_mode == 0 or gating_mode == 2):
+            attn_output, attn_weights, g_scores = self.self_attn_forward_patch(
+                self,
+                gating_mode,
+                attention_mask,
+                past_key_value,
+                input_shape,
+                query_states,
+                pre_key,
+                key_states,
+                value_states,
+            )
+            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
+            attn_output = self.o_proj(attn_output)
+            return attn_output, attn_weights, g_scores
+
+        if gating_mode == 0 and past_key_value is not None:
```

`pre_key` must be captured **before** `apply_rotary_pos_emb`; if the architecture applies a key normalization (as Qwen3 does with `k_norm`), capture it after that normalization, matching `src/modeling_qwen3.py`.

Finally, change the return at the end of the untouched dense path to a 3-tuple:

```diff
-        return attn_output, attn_weights
+        return attn_output, attn_weights, None
```

### 3.4 `<Model>DecoderLayer` — split the forward pass

Change the base class from `GradientCheckpointingLayer` to `nn.Module` (WG-KV drives checkpointing itself in Step 3.5), rename the existing `forward` to `rmsnorm_and_self_attn`, and have it return before the MLP:

```diff
-class <Model>DecoderLayer(GradientCheckpointingLayer):
+class <Model>DecoderLayer(nn.Module):
 ...
-    def forward(
+    def rmsnorm_and_self_attn(
         self,
+        gating_mode: bool,
         hidden_states: torch.Tensor,
 ...
-        hidden_states, self_attn_weights = self.self_attn(
+        hidden_states, self_attn_weights, g_scores = self.self_attn(
+            gating_mode=gating_mode,
             hidden_states=hidden_states,
 ...
         hidden_states = self.post_attention_layernorm(hidden_states)
+        return self_attn_weights, g_scores, residual, hidden_states
```

Then add a new `forward` with the same signature (plus a leading `gating_mode`) that calls it and appends `g_scores` to the outputs:

```python
    def forward(self, gating_mode: bool, hidden_states, ...):
        self_attn_weights, g_scores, residual, hidden_states = self.rmsnorm_and_self_attn(
            gating_mode, hidden_states, ...
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        outputs += (g_scores,)

        return outputs
```

### 3.5 `<Model>Model.forward` — thread `gating_mode` and collect gate scores

Add `gating_mode: bool` as the first parameter, then:

```diff
+        if gating_mode == 1 or gating_mode == 2:
+            use_cache = False
```

Wrap the decoder-layer call so training mode uses explicit gradient checkpointing:

```python
        all_g_scores = []

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            ...
            batch_size, seq_len, _ = hidden_states.shape
            if gating_mode == 1 and batch_size * seq_len >= self.config.checkpointing_threshold:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    gating_mode,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    **flash_attn_kwargs,
                    preserve_rng_state=False,
                    use_reentrant=False,
                )
            else:
                layer_outputs = decoder_layer(
                    gating_mode,
                    hidden_states,
                    attention_mask=causal_mask,
                    ...
                )

            hidden_states = layer_outputs[0]
            ...
            if gating_mode == 1:
                all_g_scores.append(layer_outputs[-1])
```

And return the scores alongside the usual output:

```diff
-        return BaseModelOutputWithPast(
+        model_outputs = BaseModelOutputWithPast(
             last_hidden_state=hidden_states,
             ...
         )
+
+        return (model_outputs, all_g_scores)
```

### 3.6 `<Model>ForCausalLM.__init__` — add the mode flag

```python
        self.gating_mode = 0  # 0: disabled, 1: training (student), 2: training (teacher), 3: inference
```

### 3.7 `<Model>ForCausalLM.forward` — unpack and forward the scores

```diff
-        outputs: BaseModelOutputWithPast = self.model(
+        outputs, all_g_scores = self.model(
+            gating_mode=self.gating_mode,
             input_ids=input_ids,
 ...
-        logits = self.lm_head(hidden_states[:, slice_indices, :])
+        logits = self.lm_head(hidden_states[:, slice_indices, :]) if not (self.gating_mode == 1 or self.gating_mode == 2) else None
 ...
-        return CausalLMOutputWithPast(
+        return DistillationCausalLMOutputWithPast(
             loss=loss,
             logits=logits,
             past_key_values=outputs.past_key_values,
             hidden_states=outputs.hidden_states,
             attentions=outputs.attentions,
+            last_hidden_state=outputs.last_hidden_state if (self.gating_mode == 1 or self.gating_mode == 2) else None,
+            all_g_scores=all_g_scores if self.gating_mode == 1 else None,
         )
```

Only `ForCausalLM` is supported by WG-KV. The other heads in the file (`ForSequenceClassification`, `ForTokenClassification`, `ForQuestionAnswering`) are not supported by this patch.

## Step 4 — Register the symlinks

Append a block to `setup_venv.sh` so the patched files replace the installed ones. The `../` depth assumes the virtual environment lives at the repository root:

```bash
MODEL_DIR="$VENV_PATH/lib/python3.12/site-packages/transformers/models/<model>"
(
    cd "$MODEL_DIR" && \
    rm configuration_<model>.py modeling_<model>.py && \
    ln -s ../../../../../../../src/configuration_<model>.py . && \
    ln -s ../../../../../../../src/modeling_<model>.py .
)
```

Re-run `bash setup_venv.sh venv` afterwards.

## Step 5 — Build a length-bin index for the tokenizer

`scripts/train.py` samples long-context training documents from [`HuggingFaceFW/fineweb-edu`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) bucketed by token length, which is tokenizer-dependent. `config.fineweb_bins` points at a small dataset holding that index — one row whose columns are the bucket names and whose values are lists of row indices into `fineweb-edu`:

```python
{"4-8k": [12, 87, ...], "8-16k": [...], "16-32k": [...], "32-64k": [...]}
```

Tokenize a slice of `fineweb-edu` (`sample-10BT`) with the target tokenizer, bucket each row by its token count, and push the result to the Hub (or a local path). Point `fineweb_bins` at it in Step 2. Reuse an existing index only if the new model shares its tokenizer — for example, all Qwen3 variants share `WG-KV/fineweb-qwen3-bins`.

The bucket names must match the `--fineweb_config` entries in `scripts/train.py`, which default to `4-8k`, `8-16k`, `16-32k`, and `32-64k`.

## Step 6 — Train and run

Nothing beyond `--model_name` changes:

```bash
# Train the gate
python scripts/train.py --model_name <hf-org>/<hf-model> --lambda_reg 0.16

# Run inference with the trained gate
python scripts/inference.py \
  --model_name <hf-org>/<hf-model> \
  --filtering_path outputs/.../<checkpoint>.pt
```

The saved checkpoint contains only the trainable gate parameters (`model.layers.*.self_attn.g_predictors.*`); the backbone stays frozen.
