import argparse
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

import os
import json
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table, handle_non_serializable


parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_name",
    default="Qwen/Qwen3-4B-Thinking-2507",
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
    "--max_total_tokens",
    type=int,
    default=90000*36*8,
)
parser.add_argument(
    "--max_tokens_per_head",
    type=int,
    default=90000,
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
    default="weights/qwen3-4b-thinking-2507-0.04.pt",
)
parser.add_argument(
    "--tag",
    default="default",
)
args = parser.parse_args()

model_config = AutoConfig.from_pretrained(args.model_name)
if args.g_expand is not None:
    model_config.g_expand = args.g_expand
model_config.use_baseline = args.use_baseline
model_config.max_total_tokens = args.max_total_tokens
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

lm = HFLM(
    pretrained=model,
    tokenizer=tokenizer,
    think_end_token="</think>",
)

results = evaluator.simple_evaluate(
    model=lm,
    tasks="aime25",
    batch_size=1,
    device="cuda",
    log_samples=True,
    gen_kwargs={
        "max_gen_toks": 81920,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0,
    },
    apply_chat_template=True,
    fewshot_as_multiturn=True,
)

os.makedirs("outputs", exist_ok=True)

with open(f"outputs/results-{args.tag}.txt", "w") as f:
    f.write(make_table(results))
    f.write("\n")
    if "groups" in results:
        f.write(make_table(results, "groups"))

with open(f"outputs/results-{args.tag}.json", "w") as f:
    json.dump(results, f, default=handle_non_serializable, indent=2)
