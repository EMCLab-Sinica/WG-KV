#!/bin/bash

if [ -z "$1" ]; then
  echo "Usage: $0 <VENV_PATH>"
  exit 1
fi

VENV_PATH="$1"

LLAMA_DIR="$VENV_PATH/lib/python3.12/site-packages/transformers/models/llama"
(
    cd "$LLAMA_DIR" && \
    rm configuration_llama.py modeling_llama.py && \
    ln -s ../../../../../../../src/configuration_llama.py . && \
    ln -s ../../../../../../../src/modeling_llama.py .
)

QWEN3_DIR="$VENV_PATH/lib/python3.12/site-packages/transformers/models/qwen3"
(
    cd "$QWEN3_DIR" && \
    rm configuration_qwen3.py modeling_qwen3.py && \
    ln -s ../../../../../../../src/configuration_qwen3.py . && \
    ln -s ../../../../../../../src/modeling_qwen3.py .
)

MODELS_DIR="$VENV_PATH/lib/python3.12/site-packages/transformers"
(
    cd "$MODELS_DIR" && \
    rm modeling_layers.py && \
    ln -s ../../../../../src/modeling_layers.py .
)