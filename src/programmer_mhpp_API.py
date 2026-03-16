import argparse
import sys
import os
import json
from tqdm import tqdm
import copy
import time
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import math
from math import comb
import re

sys.path.append('./CodeGeeX/')
from codegeex.benchmark.utils import IMPORT_HELPER
from codegeex.benchmark.execution import check_correctness

from openai import OpenAI
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

# ============== 参数设置 ==============
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--prompt_path", type=str, default="./prompts/mhpp_prompt_nocot.txt")
parser.add_argument("--dataset_file", type=str, default="./src/dataset/MHPP_C1.jsonl")
parser.add_argument("--run_tag", type=str, default=None)
args = parser.parse_args()

MODEL_PATH = args.model_path  # 在 API 版中，这里作为 OpenAI model id 使用（保持参数名不变以对齐原版）
PROMPT_PATH = args.prompt_path
DATASET_FILE = args.dataset_file
RUN_TAG = args.run_tag or os.path.splitext(os.path.basename(PROMPT_PATH))[0]

print(f"Using OpenAI model id: {MODEL_PATH}")

client = OpenAI()

if os.path.exists(PROMPT_PATH):
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        construct_few_shot_prompt = f.read()
else:
    construct_few_shot_prompt = ""


def preprocess_data(completion_string: str) -> str:
    if "```python" in completion_string:
        completion_string = completion_string.split("```python", 1)[1]
        completion_string = completion_string.split("```", 1)[0]
        return completion_string.strip()
    else:
        print("Error: No code block found")
        return ""


def _call_responses_api_with_retry(req: dict, max_retries: int = 8):
    """Simple retry with exponential backoff; keeps behavior deterministic-ish under transient errors."""
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.responses.create(**req)
        except Exception as e:
            last_err = e
            # backoff: 0.5, 1, 2, 4, ... capped
            sleep_s = min(0.5 * (2 ** attempt), 10.0)
            time.sleep(sleep_s)
    raise last_err


def fetch_completion_api(data_entry, lg, times=1):
    global construct_few_shot_prompt
    prompt = data_entry["prompt"]

    # ===== 构造输入（与最新 programmer_mhpp.py 的分支逻辑保持一致）=====
    model_path = MODEL_PATH  # 为了保持原版变量名/逻辑
    if "qwen" in model_path.lower():
        system_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        user_content = f"{construct_few_shot_prompt}\n\n{prompt}"
    elif "deepseek" in model_path.lower():
        system_content = "You are a helpful coding assistant."
        user_content = f"{construct_few_shot_prompt}\n\n{prompt}"
    elif "codellama" in model_path.lower():
        system_content = "You are a helpful coding assistant."
        user_content = f"{construct_few_shot_prompt}\n\n{prompt}"
    elif "gpt" in model_path.lower():
        # Decide reasoning level based on run_tag（保持原版写法）
        # if RUN_TAG.lower() == "cot":
        #     reasoning_level = "low"
        # else:
        #     reasoning_level = "low"
        system_content = (
            "You are a helpful coding assistant.\n"
            # f"Reasoning: {reasoning_level}"
        )
        user_content = f"{construct_few_shot_prompt}\n\n{prompt}"
    else:
        system_content = "You are a helpful coding assistant."
        user_content = f"{construct_few_shot_prompt}\n\n{prompt}\n"

    # ===== 采样策略（与最新脚本一致）=====
    # cot: do_sample True, temperature 0.5, top_p 0.9
    # nocot: greedy, temperature 0.0
    if RUN_TAG and "cot" in RUN_TAG.lower():
        temperature = 0.5
        top_p = 0.9
    else:
        temperature = 0.0
        top_p = 1.0

    raw_completions = []
    completions_code = []

    for _ in range(times):
        req = {
            "model": MODEL_PATH,
            "input": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "max_output_tokens": 2048,
            "reasoning": {"effort": "medium"},
            "store": False,
        }

        resp = _call_responses_api_with_retry(req)
        completion_text = getattr(resp, "output_text", "") or ""

        raw = completion_text
        code = preprocess_data(completion_text)

        raw_completions.append(raw)
        completions_code.append(code)

    data_entry["raw_completion_list"] = raw_completions
    data_entry["completion_list"] = completions_code
    return data_entry


def call_fetch_completion_helper(dataset, lg, times=1):
    print("Running completion fetch with progress bar...")
    new_dataset = [None] * len(dataset)

    start_time = time.time()
    total = len(dataset)

    # with ThreadPoolExecutor(max_workers=1) as executor:
    #     futures = {executor.submit(fetch_completion_api, copy.deepcopy(dataset[i]), lg): i for i in range(len(dataset))}
    #     for fut in concurrent.futures.as_completed(futures):
    #         i = futures[fut]
    #         try:
    #             updated = fut.result()
    #             merged = {**dataset[i], **updated}
    #             new_dataset[i] = merged
    #         except Exception as e:
    #             print(repr(e))
    #             new_dataset[i] = dataset[i]
    # return new_dataset

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(fetch_completion_api, copy.deepcopy(dataset[i]), lg, times =times): i
            for i in range(len(dataset))
        }

        with tqdm(total=total, desc="Fetching completions", ncols=100) as pbar:
            for fut in concurrent.futures.as_completed(futures):
                i = futures[fut]
                try:
                    updated = fut.result()
                    merged = {**dataset[i], **updated}
                    new_dataset[i] = merged
                except Exception as e:
                    print(repr(e))
                    new_dataset[i] = dataset[i]

                # 更新进度条
                pbar.update(1)

                # ETA 估计（可选显示）
                done = pbar.n
                elapsed = time.time() - start_time
                if done > 0:
                    avg_time = elapsed / done
                    remaining = total - done
                    eta = remaining * avg_time
                    pbar.set_postfix({
                        "done": done,
                        "total": total,
                        "avg_s/item": f"{avg_time:.2f}",
                        "eta_min": f"{eta / 60:.1f}"
                    })

    return new_dataset


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


if __name__ == "__main__":
    model_list = [MODEL_PATH]
    language = ["python"]

    dataset_name = os.path.splitext(os.path.basename(DATASET_FILE))[0]

    for model_path in model_list:
        model_name = os.path.basename(model_path.rstrip("/"))
        for lg in language:
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f if line.strip()]

            dataset = call_fetch_completion_helper(dataset, lg, times=1)

            os.makedirs("./dataset", exist_ok=True)

            prefix = f"{model_name}_{dataset_name}_{lg}_{RUN_TAG}"

            with open(f"./dataset/{prefix}_raw.json", "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)

            dataset_code_only = []
            for item in dataset:
                d = dict(item)
                d.pop("raw_completion_list", None)
                dataset_code_only.append(d)

            with open(f"./dataset/{prefix}.json", "w", encoding="utf-8") as f:
                json.dump(dataset_code_only, f, indent=4, ensure_ascii=False)

            print("============== Evaluation ==============")
            for k in [1]:
                save_file = f"./dataset/{prefix}_pass@{k}_details.json"
                score = evaluate_passk(dataset_code_only, lg, k, save_path=save_file)
                print(f"Pass@{k}: {score:.3f} (details saved to {save_file})")
