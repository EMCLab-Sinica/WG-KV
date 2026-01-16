import os
import argparse
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, DynamicCache
from datasets import load_dataset

import numpy as np
from third_party.LongProc.longproc.longproc_data import load_longproc_data


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default="meta-llama/Llama-3.2-1B-Instruct",
)
parser.add_argument(
    "--g_expand",
    type=float,
    default=None,
)
parser.add_argument(
    "--use_baseline",
    action="store_true",
)
parser.add_argument(
    "--max_tokens_per_head",
    type=int,
    default=32768,
)
parser.add_argument(
    "--snapkv_enabled",
    action="store_true",
)
parser.add_argument(
    "--snapkv_max_cached_tokens",
    type=int,
    default=-1,
)
parser.add_argument(
    "--snapkv_evict_ratio",
    type=float,
    default=0.0,
)
parser.add_argument(
    "--filtering_path",
    default="weights/llama-3.2-1b-instruct-0.04.pt",
)
parser.add_argument(
    "--prompt_type",
    choices=["qa", "book", "code", "html_to_tsv", "aime"],
    default="qa",
)
parser.add_argument(
    "--dump_global_lengths",
    action="store_true",
)
args = parser.parse_args()

model_config = AutoConfig.from_pretrained(args.model_name)
if args.g_expand is not None:
    model_config.g_expand = args.g_expand
model_config.use_baseline = args.use_baseline
model_config.max_total_tokens = args.max_tokens_per_head * model_config.num_hidden_layers * model_config.num_key_value_heads
model_config.max_tokens_per_head = args.max_tokens_per_head
model_config.snapkv_enabled = args.snapkv_enabled
model_config.snapkv_max_cached_tokens = args.snapkv_max_cached_tokens
model_config.snapkv_evict_ratio = args.snapkv_evict_ratio

tokenizer = AutoTokenizer.from_pretrained(args.model_name)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name,
    config=model_config,
    torch_dtype="auto",
    device_map="auto"
)

if not args.use_baseline:
    # this will load weights in bf16
    state_dict = torch.load(args.filtering_path)
    _, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    assert len(unexpected_keys) == 0

model.gating_mode = 3

if args.prompt_type == "qa":
    dataset = load_dataset('THUDM/LongBench', 'multifieldqa_en', split='test', trust_remote_code=True)
    prompt = dataset[7]['context'] + "\n\n" + dataset[7]['input'] + " Please also elaborate your reasoning."
elif args.prompt_type == "book":
    dataset = load_dataset("emozilla/pg19", split="train")
    prompt = "Please summarize the following text:\n\n" + dataset[26214]["text"]
elif args.prompt_type == "code":
    dataset = load_dataset("bigcode/the-stack-smol", split="train", data_dir="data/python")
    prompt = "Please summarize the following code:\n\n" + dataset[1084]["content"]
elif args.prompt_type == "html_to_tsv":
    dataset, eval_func = load_longproc_data("html_to_tsv_0.5k", "scripts/third_party/LongProc/data/")
    prompt = dataset[30]['input_prompt']
else:
    dataset = load_dataset("math-ai/aime25", split="test")
    prompt = dataset[0]["problem"]

messages = [
    {"role": "user", "content": prompt}
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

past_key_values = DynamicCache()
outputs = model.generate(
    **inputs,
    max_new_tokens=args.max_tokens_per_head,
    streamer=TextStreamer(tokenizer),
    past_key_values=past_key_values,
    #use_cache=False,
)

if model.gating_mode == 3:
    total_tokens_in_kv_cache = past_key_values.next_free_block * model.config.block_size
    average_tokens_in_kv_cache = total_tokens_in_kv_cache / (model.config.num_hidden_layers * model.config.num_key_value_heads)
    print(f"{average_tokens_in_kv_cache:.0f} {outputs.size(1)}")

if args.dump_global_lengths:
    os.makedirs("tensors", exist_ok=True)
    global_lengths = past_key_values.global_lengths.float() / outputs.size(1)
    np.savetxt("tensors/global_lengths.csv", global_lengths.cpu().numpy(), fmt="%.6f", delimiter=",")