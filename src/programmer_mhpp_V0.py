import argparse
import sys
import os
import json
from tqdm import tqdm
import copy
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import math
from math import comb
import re

sys.path.append('./CodeGeeX/')
from codegeex.benchmark.utils import IMPORT_HELPER
from codegeex.benchmark.execution import check_correctness

from transformers import StoppingCriteria, StoppingCriteriaList


# ============== 参数设置 ==============
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--prompt_path", type=str, default="./prompts/mhpp_prompt_nocot.txt")
parser.add_argument("--dataset_file", type=str, default="./src/dataset/MHPP_C1.jsonl")
parser.add_argument("--run_tag", type=str, default=None)
args = parser.parse_args()

MODEL_PATH = args.model_path
PROMPT_PATH = args.prompt_path
DATASET_FILE = args.dataset_file
RUN_TAG = args.run_tag or os.path.splitext(os.path.basename(PROMPT_PATH))[0]

print(f"Loading local model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto")

dtype = torch.bfloat16 if "gpt-oss" in MODEL_PATH.lower() else torch.float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=dtype,
    device_map="auto",
)

if os.path.exists(PROMPT_PATH):
    with open(PROMPT_PATH, "r") as f:
        construct_few_shot_prompt = f.read()
else:
    construct_few_shot_prompt = ""


def _endswith(seq, pat):
    return len(seq) >= len(pat) and seq[-len(pat):] == pat



# def extract_signature_from_prompt(prompt: str) -> str | None:
#     for line in prompt.splitlines():
#         s = line.strip()
#         if s.startswith("def ") and s.endswith(":"):
#             return s
#
#     m = re.search(r"def\s+\w+\s*\([^\)]*\)\s*->\s*[^:]+:", prompt)
#     if m:
#         return m.group(0)
#     m = re.search(r"def\s+\w+\s*\([^\)]*\)\s*:", prompt)
#     if m:
#         return m.group(0)
#
#     return None
#
# def ensure_function_signature(code_block: str, prompt: str) -> str:
#
#     sig = extract_signature_from_prompt(prompt)
#     if not sig:
#
#         return code_block
#
#     m = re.match(r"\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", sig)
#     func_name = m.group(1) if m else None
#
#     if func_name and re.search(rf"^\s*def\s+{func_name}\s*\(", code_block, flags=re.M):
#         return code_block
#
#     lines = code_block.splitlines()
#
#     import_lines = []
#     body_lines = []
#     for ln in lines:
#         if re.match(r"\s*(?:from\s+\w[\w\.]*\s+import\s+.*|import\s+\w[\w\.]*.*)$", ln):
#             import_lines.append(ln)
#         else:
#             body_lines.append(ln)
#
#     body = "\n".join(body_lines).strip()
#     if not body:
#         body = "pass"
#
#     indented_body = "\n".join(("    " + l if l.strip() != "" else "" for l in body.splitlines()))
#
#     fixed = ""
#     if import_lines:
#         fixed += "\n".join(import_lines) + "\n\n"
#     fixed += sig + "\n" + indented_body + "\n"
#     return fixed


class StopAfterOneCodeBlock(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tk = tokenizer
        self.open_seen = False
        self.open_ids = self.tk.encode("```python", add_special_tokens=False)
        self.close_ids = self.tk.encode("```", add_special_tokens=False)

    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        if not self.open_seen and _endswith(ids, self.open_ids):
            self.open_seen = True
        elif self.open_seen and _endswith(ids, self.close_ids):
            return True
        return False


def preprocess_data(completion_string):
    if "```python" in completion_string:
        completion_string = completion_string.split("```python", 1)[1]
        completion_string = completion_string.split("```", 1)[0]
        return completion_string.strip()
    else:
        print("Error: No code block found")
        return ""


# def fetch_completion_hf(data_entry, lg, times=10, topk=5):
#
#     global construct_few_shot_prompt
#     prompt = data_entry["prompt"]
#
#     if "qwen" in MODEL_PATH.lower():
#
#         messages = [
#             {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#
#             {"role": "user", "content": f"{construct_few_shot_prompt}\n\n{prompt}"}
#         ]
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True
#         )
#     elif "codellama" in MODEL_PATH.lower():
#         text = f"<s>[INST] {construct_few_shot_prompt}\n\n{prompt}\n[/INST]"
#     else:
#         # 其他模型用你原来的拼接方式
#         text = (
#             f"{construct_few_shot_prompt}\n\n"
#             f"{prompt}\n"
#         )
#
# #     text = f"""{construct_few_shot_prompt}
# #
# # ```python
# # {prompt}
# # # Output Python code only.
# # ```"""
# #     text = (
# #         f"{construct_few_shot_prompt}\n\n"
# #         f"{prompt}\n"
# #     )
#     # inputs = tokenizer(text, return_tensors="pt").to(model.device)
#     inputs = tokenizer(text, return_tensors="pt")
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#
#     raw_completions = []
#     completions_code = []
#     logprob_traces = []
#
#     stoppers = StoppingCriteriaList([StopAfterOneCodeBlock(tokenizer)])
#
#     gen_kwargs = {
#         "max_new_tokens": 1024,
#         "return_dict_in_generate": True,
#         "output_scores": True,
#         "stopping_criteria": stoppers,
#     }
#
#     if RUN_TAG and "cot" in RUN_TAG.lower():
#         gen_kwargs.update({
#             "do_sample": True,
#             "temperature": 1.0,
#             "top_p": 0.9
#         })
#     else:  # 默认当作 nocot
#         gen_kwargs.update({
#             "do_sample": False,
#             "temperature": 0.0
#         })
#
#     for _ in range(times):
#         outputs = model.generate(**inputs, **gen_kwargs)
#         input_len = inputs["input_ids"].shape[1]
#
#         completion = tokenizer.decode(outputs.sequences[0][input_len:], skip_special_tokens=True)
#         raw = completion
#         completion = preprocess_data(completion)
#         # if completion.strip():
#         #     completion = ensure_function_signature(completion, data_entry.get("prompt", ""))
#
#
#         scores = outputs.scores
#         token_ids = outputs.sequences[0][input_len:]
#
#         tokens = tokenizer.convert_ids_to_tokens(token_ids)
#         token_logprobs = []
#         linewise_metrics = []  # 保存每行首 token 的 topk + 熵 + 概率差
#
#         at_line_start = True
#         for step, logits in enumerate(scores):
#             logprobs = F.log_softmax(logits[0], dim=-1)
#             tok_id = token_ids[step]
#             tok = tokenizer.decode([tok_id])
#
#             # 记录每行首 token
#             if at_line_start and tok.strip() != "":
#                 topk_vals, topk_ids = torch.topk(logprobs, k=topk)
#                 probs = topk_vals.exp().cpu().tolist()
#                 toks = [tokenizer.decode([tid.item()]) for tid in topk_ids]
#
#
#                 entropy = -sum(p * math.log(p + 1e-12) for p in probs)
#
#
#                 prob_diff = probs[0] - probs[1] if len(probs) > 1 else 0.0
#
#                 linewise_metrics.append({
#                     "line_start_token": tok,
#                     "step": step,
#                     "topk_tokens": toks,
#                     "topk_probs": probs,
#                     "entropy": entropy,
#                     "prob_diff": prob_diff
#                 })
#
#                 at_line_start = False
#
#
#             if tok in ["\n", "Ċ"]:
#                 at_line_start = True
#
#             token_logprobs.append(logprobs[tok_id].item())
#
#         sum_logprob = sum(token_logprobs)
#         avg_logprob = sum_logprob / len(token_logprobs) if token_logprobs else None
#
#         logprob_traces.append({
#             "tokens": tokens,
#             "token_logprobs": token_logprobs,
#             "sum_logprob": sum_logprob,
#             "avg_logprob": avg_logprob,
#             "linewise_metrics": linewise_metrics
#         })
#
#         raw_completions.append(raw)
#         completions_code.append(completion)
#
#     data_entry["prompt"] = data_entry.get("prompt", "")
#     data_entry["raw_completion_list"] = raw_completions
#     data_entry["completion_list"] = completions_code
#     data_entry["logprob_list"] = logprob_traces
#     return data_entry

    #     for step, logits in enumerate(scores):
    #         logprobs = F.log_softmax(logits[0], dim=-1)
    #         tok_id = token_ids[step]
    #         token_logprobs.append(logprobs[tok_id].item())
    #
    #         # top-k 候选
    #         topk_vals, topk_ids = torch.topk(logprobs, k=topk)
    #         topk_list.append([
    #             {"token": tokenizer.decode([tid.item()]), "logprob": val.item()}
    #             for tid, val in zip(topk_ids, topk_vals)
    #         ])
    #
    #     sum_logprob = sum(token_logprobs)
    #     avg_logprob = sum_logprob / len(token_logprobs) if token_logprobs else None
    #
    #     logprob_traces.append({
    #         "tokens": tokens,
    #         "token_logprobs": token_logprobs,
    #         "sum_logprob": sum_logprob,
    #         "avg_logprob": avg_logprob,
    #         "topk_list": topk_list
    #     })
    #
    #     raw_completions.append(raw)
    #     completions_code.append(completion)
    #
    # data_entry["prompt"] = data_entry.get("prompt", "")
    # data_entry["raw_completion_list"] = raw_completions
    # data_entry["completion_list"] = completions_code
    # data_entry["logprob_list"] = logprob_traces
    # return data_entry

def fetch_completion_hf(data_entry, lg, times=10, topk=5):
    global construct_few_shot_prompt
    prompt = data_entry["prompt"]

    # ===== 构造输入 =====
    if "qwen" in MODEL_PATH.lower():
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{construct_few_shot_prompt}\n\n{prompt}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    elif "deepseek" in model_path.lower():
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"{construct_few_shot_prompt}\n\n{prompt}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif "codellama" in model_path.lower():
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"{construct_few_shot_prompt}\n\n{prompt}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )


    elif "gpt" in model_path.lower():
        # Decide reasoning level based on run_tag
        if RUN_TAG.lower() == "cot":
            reasoning_level = "medium"
        else:  # nocot
            reasoning_level = "low"
        system_content = (
            "You are a helpful coding assistant.\n"
            f"Reasoning: {reasoning_level}"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{construct_few_shot_prompt}\n\n{prompt}"}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    # elif "codellama" in MODEL_PATH.lower():
    #     text = f"<s>[INST] {construct_few_shot_prompt}\n\n{prompt}\n[/INST]"
    else:
        text = f"{construct_few_shot_prompt}\n\n{prompt}\n"

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ===== 停止条件 =====
    stoppers = StoppingCriteriaList([StopAfterOneCodeBlock(tokenizer)])

    gen_kwargs = {
        "max_new_tokens": 1024,
        "return_dict_in_generate": True,
        "output_scores": True,
        "stopping_criteria": stoppers,
        "num_return_sequences": times,
    }

    # 是否采样（cot 情况启用采样，nocot 用 greedy）
    if RUN_TAG and "cot" in RUN_TAG.lower():
        gen_kwargs.update({
            "do_sample": True,
            "temperature": 0.5,
            "top_p": 0.9
        })
    else:
        gen_kwargs.update({
            "do_sample": False,
            "temperature": 0.0
        })

    # ===== 执行生成 =====
    outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]

    raw_completions = []
    completions_code = []
    logprob_traces = []

    batch_size = times
    for i in range(batch_size):
        seq = outputs.sequences[i]
        completion = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        raw = completion
        completion = preprocess_data(completion)

        token_ids = seq[input_len:]
        scores = [step[i] for step in outputs.scores]
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        token_logprobs = []
        linewise_metrics = []
        at_line_start = True

        for step, logits in enumerate(scores):
            logprobs = F.log_softmax(logits, dim=-1)
            tok_id = token_ids[step]
            tok = tokenizer.decode([tok_id])

            if at_line_start and tok.strip() != "":
                topk_vals, topk_ids = torch.topk(logprobs, k=topk)
                probs = topk_vals.exp().cpu().tolist()
                toks = [tokenizer.decode([tid.item()]) for tid in topk_ids]
                entropy = -sum(p * math.log(p + 1e-12) for p in probs)
                prob_diff = probs[0] - probs[1] if len(probs) > 1 else 0.0
                linewise_metrics.append({
                    "line_start_token": tok,
                    "step": step,
                    "topk_tokens": toks,
                    "topk_probs": probs,
                    "entropy": entropy,
                    "prob_diff": prob_diff
                })
                at_line_start = False

            if tok in ["\n", "Ċ"]:
                at_line_start = True

            token_logprobs.append(logprobs[tok_id].item())

        sum_logprob = sum(token_logprobs)
        avg_logprob = sum_logprob / len(token_logprobs) if token_logprobs else None

        logprob_traces.append({
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "sum_logprob": sum_logprob,
            "avg_logprob": avg_logprob,
            "linewise_metrics": linewise_metrics
        })

        raw_completions.append(raw)
        completions_code.append(completion)

    data_entry["raw_completion_list"] = raw_completions
    data_entry["completion_list"] = completions_code
    data_entry["logprob_list"] = logprob_traces
    return data_entry


def call_fetch_completion_helper(dataset, lg):
    print("Running completion fetch...")
    new_dataset = [None] * len(dataset)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(fetch_completion_hf, copy.deepcopy(dataset[i]), lg): i for i in range(len(dataset))}
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                updated = fut.result()
                merged = {**dataset[i], **updated}
                new_dataset[i] = merged
            except Exception as e:
                print(repr(e))
                new_dataset[i] = dataset[i]
    return new_dataset


# def evaluate_passk(dataset, lg="python", k=5, save_path=None):
#     total = len(dataset)
#     detailed_results = []
#     success = 0
#
#     for sample in dataset:
#         completions = sample.get("completion_list", [])
#         fn_name = sample["function_name"]
#         task_id = sample["task_id"]
#
#         # 遍历所有 completions，统计通过的数量
#         passed_list = []
#         for idx, code in enumerate(completions):
#             if not code.strip():
#                 passed_list.append(False)
#                 continue
#
#             test_code = "\n".join(IMPORT_HELPER[lg]) + "\n" + sample.get("prompt") + "\n" + code + "\n" + sample.get("test", "") + "\n" + f"def test_check():\n    check({fn_name})\n\n" + "test_check()"
#
#             exec_sample = {
#                 "test_code": test_code,
#                 "prompt": sample.get("prompt"),
#                 "generation": code,
#             }
#
#             result = check_correctness(
#                 task_id=task_id,
#                 sample=exec_sample,
#                 language_type=lg,
#                 timeout=3,
#                 tmp_dir="./tmp",
#             )
#             passed_list.append(result["passed"])
#
#         c = sum(passed_list)
#         N = len(passed_list)
#
#         # 无偏估计 Pass@k
#         if c == 0:
#             passk_score = 0.0
#         elif N <= k:
#             passk_score = 1.0 if c > 0 else 0.0
#         else:
#             passk_score = 1.0 - comb(N - c, k) / comb(N, k)
#
#         if passk_score > 0:
#             success += 1
#
#         detailed_results.append({
#             "task_id": task_id,
#             "N": N,
#             "c": c,
#             "pass@k": passk_score,
#             "completion_results": [{"completion_id": i, "passed": p} for i, p in enumerate(passed_list)]
#         })
#
#     avg_passk = sum(item["pass@k"] for item in detailed_results) / total if total > 0 else 0.0
#
#     if save_path:
#         with open(save_path, "w", encoding="utf-8") as f:
#             json.dump(detailed_results, f, indent=4, ensure_ascii=False)
#
#     return avg_passk


def evaluate_passk(dataset, lg="python", k=5, save_path=None):
    total = len(dataset)
    success = 0
    detailed_results = []
    for sample in dataset:
        completions = sample.get("completion_list", [])
        fn_name = sample["function_name"]
        task_id = sample["task_id"]

        topk = completions[:k] if completions else [""]
        passed = False
        completion_results = []

        for idx, code in enumerate(topk):
            if code.strip() == "":
                completion_results.append({
                    "completion_id": idx,
                    "passed": False
                })
                continue

            test_code = "\n".join(IMPORT_HELPER[lg]) + "\n" + sample.get("prompt") + "\n" + code + "\n" + sample.get("test", "") + "\n" + f"def test_check():\n    check({fn_name})\n\n" + "test_check()"
            # test_code = "\n".join(IMPORT_HELPER[lg]) + "\n" + code + "\n" + sample.get("test", "") + "\n" + f"def test_check():\n    check({fn_name})\n\n" + "test_check()\n"

            exec_sample = {
                "test_code": test_code,
                "prompt": sample.get("prompt"),
                "generation": code,
            }

            result = check_correctness(
                task_id=task_id,
                sample=exec_sample,
                language_type=lg,
                timeout=3,
                tmp_dir="./tmp",)

            is_passed = result["passed"]
            completion_results.append({
                "completion_id": idx,
                "passed": is_passed
            })
            if is_passed:
                passed = True

        if passed:
            success += 1

        detailed_results.append({
            "task_id": task_id,
            "topk": k,
            "any_passed": passed,
            "completion_results": completion_results
        })

    passk_score = success / total if total > 0 else 0.0

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(detailed_results, f, indent=4, ensure_ascii=False)

    return passk_score


# if __name__ == "__main__":
#     model_list = [MODEL_PATH]
#     language = ["python"]
#     dataset_name = os.path.splitext(os.path.basename(DATASET_FILE))[0]
#
#     for model_path in model_list:
#         model_name = os.path.basename(model_path.rstrip("/"))  # 取最后的目录名作为模型名
#         for lg in language:
#             with open("./src/dataset/MHPP_C1.jsonl", "r", encoding="utf-8") as f:
#                 dataset = [json.loads(line) for line in f if line.strip()]
#
#             dataset = call_fetch_completion_helper(dataset, lg)
#
#             os.makedirs("./dataset", exist_ok=True)
#
#             with open(f"./dataset/{model_name}_{lg}_raw.json", "w", encoding="utf-8") as f:
#                 json.dump(dataset, f, indent=4, ensure_ascii=False)
#
#             dataset_code_only = []
#             for item in dataset:
#                 d = dict(item)
#                 d.pop("raw_completion_list", None)
#                 d.pop("logprob_list", None)
#                 dataset_code_only.append(d)
#
#             with open(f"./dataset/{model_name}_{lg}.json", "w", encoding="utf-8") as f:
#                 json.dump(dataset_code_only, f, indent=4, ensure_ascii=False)
#
#             logprob_only = []
#             for item in dataset:
#                 logprob_only.append({
#                     "task_id": item.get("task_id"),
#                     "logprob_list": item.get("logprob_list")
#                 })
#             with open(f"./dataset/{model_name}_{lg}_logprobs.json", "w", encoding="utf-8") as f:
#                 json.dump(logprob_only, f, indent=4, ensure_ascii=False)
#
#             print("============== Evaluation ==============")
#             for k in [1, 3, 5]:
#                 save_file = f"./dataset/{model_name}_{lg}_pass@{k}_details.json"
#                 score = evaluate_passk(dataset_code_only, lg, k, save_path=save_file)
#                 print(f"Pass@{k}: {score:.3f} (details saved to {save_file})")
if __name__ == "__main__":
    model_list = [MODEL_PATH]
    language = ["python"]

    dataset_name = os.path.splitext(os.path.basename(DATASET_FILE))[0]

    for model_path in model_list:
        model_name = os.path.basename(model_path.rstrip("/"))
        for lg in language:
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f if line.strip()]

            dataset = call_fetch_completion_helper(dataset, lg)

            os.makedirs("./dataset", exist_ok=True)

            prefix = f"{model_name}_{dataset_name}_{lg}_{RUN_TAG}"

            with open(f"./dataset/{prefix}_raw.json", "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)

            dataset_code_only = []
            for item in dataset:
                d = dict(item)
                d.pop("raw_completion_list", None)
                d.pop("logprob_list", None)
                dataset_code_only.append(d)

            with open(f"./dataset/{prefix}.json", "w", encoding="utf-8") as f:
                json.dump(dataset_code_only, f, indent=4, ensure_ascii=False)

            logprob_only = []
            for item in dataset:
                logprob_only.append({
                    "task_id": item.get("task_id"),
                    "logprob_list": item.get("logprob_list")
                })
            with open(f"./dataset/{prefix}_logprobs.json", "w", encoding="utf-8") as f:
                json.dump(logprob_only, f, indent=4, ensure_ascii=False)

            print("============== Evaluation ==============")
            for k in [1, 5, 10]:
                save_file = f"./dataset/{prefix}_pass@{k}_details.json"
                score = evaluate_passk(dataset_code_only, lg, k, save_path=save_file)
                print(f"Pass@{k}: {score:.3f} (details saved to {save_file})")