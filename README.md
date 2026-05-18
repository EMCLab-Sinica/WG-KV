[![arXiv](https://img.shields.io/badge/arXiv-2512.17452-b31b1b.svg)](https://arxiv.org/abs/2512.17452)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of the paper **"KV Admission: Learning What to Write for Efficient Long-Context LLM Inference"**.

## 📖 Introduction

Long-context LLM inference is bottlenecked by the quadratic attention complexity and linear KV cache growth. Prior approaches (KV Selection or Eviction) mitigate this post-hoc, but overlook the root inefficiency: indiscriminate token admission.

We propose **Write-Gated KV (WG-KV)** to introduce the missing primitive: **KV Admission**. Instead of blindly accumulating every token, WG-KV employs a lightweight, learnable mechanism to predict token utility before cache entry. By filtering out redundant states early to maintain a compact global cache alongside a sliding local cache, WG-KV reduces memory usage by **36-69%** and delivers **2.56-4.17x** prefill and **1.59-2.63x** decode speedups, all while maintaining near-lossless task accuracy.

<p align="center" width="100%"><img src="assets/fig1_5.png" width="1200"></p>

## ⚙️ System Architecture

Implementing WG-KV results in **ragged KV states**, where different attention heads possess significantly different cache lengths. Standard contiguous memory allocation would lead to severe fragmentation. To solve this, we introduce **dual-cache paged memory management**:

<p align="center" width="100%"><img src="assets/fig6.png" width="800"></p>

This design effectively manages ragged KV states without memory fragmentation, while translating theoretical sparsity into practical wall-clock speedups.

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/EMCLab-Sinica/WG-KV.git
cd WG-KV
```

### 2. Setup Environment

We use [uv](https://github.com/astral-sh/uv) to manage the virtual environment, handle dependencies, and apply patches to the [🤗 Transformers](https://github.com/huggingface/transformers) library.

```bash
# Prepare Python environment with CUDA support
conda create -n wgkv python=3.12
conda activate wgkv
conda install cuda -c nvidia/label/cuda-12.9.1

# Create and activate virtual environment
uv venv venv
source venv/bin/activate

# Install core dependencies
uv pip install -r requirements.txt

# Apply patches to the Transformers library
bash setup_venv.sh venv
```

### 3. Install Custom vLLM

WG-KV requires a modified version of [vLLM](https://github.com/vllm-project/vllm) to support sparse prefill kernels.

```bash
# Clone our vLLM fork
git clone -b v0.10.0 --no-tags --depth=1 https://github.com/vllm-sparse/vllm.git scripts/third_party/vllm

# Configure build environment
env -C scripts/third_party/vllm python use_existing_torch.py
uv pip install -r scripts/third_party/vllm/requirements/build.txt

# Compile and install (this may take a few minutes)
MAX_JOBS=4 uv pip install --no-build-isolation -e scripts/third_party/vllm
```

## 📦 Model Checkpoints

We provide pre-trained weights for the **Write-Gate MLP**. The checkpoints follow the naming convention `{model_name}-{lambda}.pt`, where `lambda` (λ) controls the trade-off between sparsity and accuracy:

* **Higher λ (Aggressive):** Higher sparsity, admitting fewer KVs (prioritizes efficiency).
* **Lower λ (Conservative):** Lower sparsity, admitting more KVs (prioritizes accuracy).

You can download them directly from [🤗 our Hugging Face Repository](https://huggingface.co/WG-KV/checkpoints).

```bash
# Download Llama-3.1-8B-Instruct checkpoints
hf download WG-KV/checkpoints --include "llama-3.1-8b-instruct-*.pt" --local-dir weights

# Download Qwen3-4B-Instruct-2507 checkpoints
hf download WG-KV/checkpoints --include "qwen3-4b-instruct-2507-*.pt" --local-dir weights

# Download Qwen3-4B-Thinking-2507 checkpoints
hf download WG-KV/checkpoints --include "qwen3-4b-thinking-2507-*.pt" --local-dir weights
```

## 🚀 Usage

### Training

To train the gate (e.g., for Llama-3.1-8B) with a specific λ (e.g., 0.16):

```bash
python scripts/train.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --lambda_reg 0.16
```

The trained checkpoints will be saved in the `outputs/` directory.

*See `scripts/train.py` for additional arguments.*

### Inference

To run inference using the trained gate, specify the checkpoint path using `--filtering_path`:

```bash
python scripts/inference.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --filtering_path weights/llama-3.1-8b-instruct-0.16.pt
```

*See `scripts/inference.py` for additional arguments.*

## 📊 Results

### 1. System Efficiency ⚡

WG-KV significantly accelerates inference while reducing resource consumption. On Llama-3.1-8B with λ = 0.16 (~80% sparsity by admitting ~20% KVs), it achieves **~3x prefill speedup**, **~2x decode speedup**, and a **~50% reduction in memory usage**.

<details>

<summary>Show commands</summary>

```bash
# Evaluate Baseline (Full Attention)
python scripts/profiling.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --profile_attn_mlp \
  --use_pg19 \
  --avg_tokens_per_head 401000 \
  --max_tokens_per_head 401000 \
  --prompt_length 400000 \
  --g_expand 0 \
  --random_sparsity 0.0

# Evaluate WG-KV
python scripts/profiling.py \
  --model_name meta-llama/Llama-3.1-8B-Instruct \
  --profile_attn_mlp \
  --use_pg19 \
  --avg_tokens_per_head 121000 \
  --max_tokens_per_head 401000 \
  --prompt_length 400000 \
  --filtering_path weights/llama-3.1-8b-instruct-0.16.pt
```

</details>

<p align="center" width="100%"><img src="assets/fig8.png" width="400"></p>

### 2. Task Accuracy 🧠

Evaluated on the [HELMET](https://github.com/princeton-nlp/HELMET) benchmark, WG-KV significantly outperforms admission baselines (AdaEA++, DuoAttention, Local Attention) across diverse tasks, maintaining **near-lossless accuracy** even with **<10% admitted KVs**.

<details>

<summary>Show commands</summary>

```bash
# Clone the repository
git clone -b dev --recursive https://github.com/lashhw/HELMET.git scripts/third_party/HELMET

# Install dependencies
uv pip install -r scripts/third_party/HELMET/requirements.txt
uv pip install flash-attn --no-build-isolation

# Download HELMET datasets
env -C scripts/third_party/HELMET bash scripts/download_data.sh

# Link checkpoints directory
ln -s ../utils/gather_adaea_stats/patterns weights/adaea
ln -s ../../../weights scripts/third_party/HELMET/weights

# Evaluate Baselines (Full Attention, Local Attention, DuoAttention, AdaEA++)
env -C scripts/third_party/HELMET bash run_vanilla_all.sh
env -C scripts/third_party/HELMET bash run_local_all.sh
env -C scripts/third_party/HELMET bash run_duo_all.sh
env -C scripts/third_party/HELMET bash run_adaea_all.sh

# Evaluate WG-KV
env -C scripts/third_party/HELMET bash run_filtering_all.sh
```

</details>

<p align="center" width="100%"><img src="assets/fig7.png" width="800"></p>

## 📂 Repository Structure

* `scripts/`: Training, inference, and evaluation scripts.
* `src/`: Modified [🤗 Transformers](https://github.com/huggingface/transformers) modules implementing core logic for WG-KV.
* `utils/`: Attention kernel microbenchmarks and testing scripts.

## 📄 Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{wgkv,
   title={KV Admission: Learning What to Write for Efficient Long-Context Inference},
   author={Yen-Chieh Huang and Pi-Cheng Hsiu and Rui Fang and Ming-Syan Chen},
   year={2025},
   eprint={2512.17452},
   archivePrefix={arXiv},
   primaryClass={cs.LG},
   url={https://arxiv.org/abs/2512.17452}, 
}
```
