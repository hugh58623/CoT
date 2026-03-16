# programmer_bcb_api.py
import argparse, os, json, copy, time, re, concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except RuntimeError:
    pass

import sys
sys.path.append('./CodeGeeX/')
from codegeex.benchmark.utils import IMPORT_HELPER
from codegeex.benchmark.execution import check_correctness

from openai import OpenAI

# -----------------------------
# Utils
# -----------------------------
def strip_code_fence(s: str) -> str:
    if "```python" in s:
        s = s.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0]
    return s.strip()

def preprocess_data(completion_string: str) -> str:
    """Extract code; keep behavior close to original (requires code fence)."""
    if "```python" in completion_string:
        completion_string = completion_string.split("```python", 1)[1]
        completion_string = completion_string.split("```", 1)[0]
        return completion_string.strip()
    # fallback: try any fence
    if "```" in completion_string:
        completion_string = completion_string.split("```", 1)[1]
        completion_string = completion_string.split("```", 1)[0]
        return completion_string.strip()
    # last resort: empty (align with original expectation)
    return ""

def extract_text_from_resp(resp) -> str:
    """
    More robust than resp.output_text only.
    Tries to concatenate message content blocks.
    """
    t = getattr(resp, "output_text", None)
    if t:
        return t

    out = getattr(resp, "output", None) or []
    parts = []
    for item in out:
        if getattr(item, "type", None) == "message":
            for c in getattr(item, "content", None) or []:
                ctype = getattr(c, "type", None)
                if ctype in ("output_text", "text"):
                    parts.append(getattr(c, "text", "") or "")
    return "".join(parts)

def _call_responses_api_with_retry(client: OpenAI, req: dict, max_retries: int = 8):
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.responses.create(**req)
        except Exception as e:
            last_err = e
            sleep_s = min(0.5 * (2 ** attempt), 10.0)
            time.sleep(sleep_s)
    raise last_err

# -----------------------------
# Completion (API)
# -----------------------------
def fetch_completion_api(
    sample,
    client: OpenAI,
    model_id: str,
    model_path_for_branch: str,
    fewshot: str,
    times: int = 10,
    max_output_tokens: int = 2048,
    run_tag: str = "nocot",
    reasoning_effort_fixed: str = "medium",  # keep constant to avoid mixing variables
):
    prompt_src = sample.get("prompt") or sample.get("complete_prompt") or ""

    # ---- keep original branching logic (system/user content) ----
    if "qwen" in model_path_for_branch.lower():
        system_content = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        user_content = f"{fewshot}\n\n{prompt_src}"
    elif "deepseek" in model_path_for_branch.lower():
        system_content = "You are a helpful coding assistant."
        user_content = f"{fewshot}\n\n{prompt_src}"
    elif "gpt" in model_path_for_branch.lower():
        # keep original "Reasoning: ..." textual instruction logic (prompt-level)
        # if run_tag.lower() == "cot":
        #     reasoning_level = "medium"
        # else:
        #     reasoning_level = "low"
        system_content = (
            "You are a helpful coding assistant.\n"
            # f"Reasoning: {reasoning_level}"
        )
        user_content = f"{fewshot}\n\n{prompt_src}"
    else:
        system_content = "You are a helpful coding assistant."
        user_content = f"{fewshot}\n\n{prompt_src}\n"

    # # Force exactly one python fenced code block (align with your evaluation parsing)
    # format_hint = (
    #     "\n\nIMPORTANT:\n"
    #     "Return ONLY ONE Python code block wrapped exactly like:\n"
    #     "```python\n"
    #     "<your code>\n"
    #     "```\n"
    #     "Do not output any other text.\n"
    # )
    # user_content = (user_content or "") + format_hint

    # ---- keep original sampling strategy ----
    if run_tag and "cot" in run_tag.lower():
        temperature = 1.0
        top_p = 0.9
        do_sample = True
    else:
        temperature = 0.0
        top_p = 1.0
        do_sample = False

    raw_list, code_list, logs = [], [], []

    for _ in range(times):
        req = {
            "model": model_id,
            "input": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            "max_output_tokens": max_output_tokens,
            # keep fixed to avoid confounding internal reasoning with "visible CoT"
            "reasoning": {"effort": "medium"},
            # "temperature": temperature,
            # "top_p": top_p,
            "store": False,
        }

        # (do_sample is implied by temperature>0 usually; leaving explicit toggles out for API compatibility)
        resp = _call_responses_api_with_retry(client, req)
        completion_text = extract_text_from_resp(resp) or ""

        completion_raw = completion_text
        completion_code = strip_code_fence(completion_raw)

        raw_list.append(completion_raw)
        code_list.append(completion_code)

        # API version: keep field for compatibility; fill None unless you enable logprobs later
        logs.append(None)

    out = dict(sample)
    out["raw_completion_list"] = raw_list
    out["completion_list"] = code_list
    out["logprob_list"] = logs
    return out

# -----------------------------
# Eval code (unchanged)
# -----------------------------
def build_exec_code(sample, candidate_code, fn_alias="entry_point", lg="python"):
    common_imports = "\n".join(IMPORT_HELPER[lg])
    target_name = sample.get("entry_point", "task_func")
    alias_code = f"\n# alias\n{fn_alias} = {target_name}\n"
    prompt_src = sample.get("prompt") or sample.get("complete_prompt") or ""
    test_str = sample.get("test") or ""
    runner = """
import unittest as _ut, os
def _run_unittest():
    _suite = _ut.defaultTestLoader.loadTestsFromTestCase(TestCases)
    _res = _ut.TextTestRunner(stream=open(os.devnull, 'w')).run(_suite)
    if _res.failures or _res.errors:
        raise AssertionError("unittest failed")
_run_unittest()
"""
    return common_imports + "\n" + prompt_src + "\n" + candidate_code + alias_code + "\n" + test_str + "\n" + runner

def evaluate_passk(dataset, lg="python", k=5, save_path=None, fn_alias="entry_point"):
    total = len(dataset)
    success = 0
    details = []
    for sample in dataset:
        task_id = sample.get("task_id", "unknown")
        completions = sample.get("completion_list", []) or [""]
        topk = completions[:k]
        any_passed = False
        comp_results = []
        for idx, code in enumerate(topk):
            if not code.strip():
                comp_results.append({"completion_id": idx, "passed": False})
                continue
            exec_code = build_exec_code(sample, code, fn_alias=fn_alias, lg=lg)
            exec_sample = {
                "test_code": exec_code,
                "prompt": sample.get("prompt") or sample.get("complete_prompt") or "",
                "generation": code
            }
            result = check_correctness(task_id=task_id, sample=exec_sample, language_type=lg, timeout=3, tmp_dir="./tmp")
            passed = bool(result["passed"])
            comp_results.append({"completion_id": idx, "passed": passed})
            if passed:
                any_passed = True
        if any_passed:
            success += 1
        details.append({"task_id": task_id, "topk": k, "any_passed": any_passed, "completion_results": comp_results})

    passk = success / total if total else 0.0
    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)
    return passk

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="OpenAI model id (e.g., gpt-5-nano)")
    parser.add_argument("--prompt_path", type=str, default="./prompts/mhpp_prompt_nocot.txt")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--times", type=int, default=10)
    parser.add_argument("--max_output_tokens", type=int, default=2048)
    # parser.add_argument("--reasoning_effort", type=str, default="medium", help='fixed internal reasoning effort: minimal/medium/high')
    parser.add_argument("--max_workers", type=int, default=1, help="API concurrency")
    parser.add_argument("--topk_eval", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    model_id = args.model_path
    prompt_path = args.prompt_path
    dataset_file = args.dataset_file
    run_tag = args.run_tag or os.path.splitext(os.path.basename(prompt_path))[0]

    print(f"Using OpenAI model id: {model_id}")
    print(f"RUN_TAG: {run_tag}")
    print(f"times per sample: {args.times}, max_output_tokens: {args.max_output_tokens}")
    print(f"max_workers: {args.max_workers}")

    client = OpenAI()

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            fewshot = f.read()
    else:
        fewshot = ""

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    # ---- completion fetch with progress bar + ETA ----
    total = len(dataset)
    new_dataset = [None] * total
    start_time = time.time()

    print("Running completion fetch (API) ...")

    with tqdm(total=total, desc="Fetching completions", ncols=110) as pbar:
        for i in range(total):
            try:
                new_dataset[i] = fetch_completion_api(
                    copy.deepcopy(dataset[i]),
                    client,
                    model_id,
                    model_id,  # keep branch logic based on model_path string
                    fewshot,
                    args.times,
                    args.max_output_tokens,
                    run_tag,
                )
            except Exception as e:
                print("[ERROR]", repr(e))
                new_dataset[i] = dataset[i]

            pbar.update(1)

            done = pbar.n
            elapsed = time.time() - start_time
            if done > 0:
                avg_s = elapsed / done
                eta_s = avg_s * (total - done)
                pbar.set_postfix({
                    "avg_s/item": f"{avg_s:.2f}",
                    "eta_min": f"{eta_s / 60:.1f}"
                })

    os.makedirs("./dataset", exist_ok=True)
    model_name = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", os.path.basename(model_id.rstrip("/")))
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    prefix = f"{model_name}_{dataset_name}_python_{run_tag}"

    # 1) full data (raw + code + logprob_list(None))
    with open(f"./dataset/{prefix}_raw.json", "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)

    # 2) pruned data without raw/logprobs
    dataset_code_only = []
    for item in new_dataset:
        d = dict(item)
        d.pop("raw_completion_list", None)
        d.pop("logprob_list", None)
        dataset_code_only.append(d)
    with open(f"./dataset/{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(dataset_code_only, f, indent=4, ensure_ascii=False)

    # 3) only logprobs (kept for compatibility)
    logprob_only = []
    for item in new_dataset:
        logprob_only.append({
            "task_id": item.get("task_id"),
            "logprob_list": item.get("logprob_list")
        })
    with open(f"./dataset/{prefix}_logprobs.json", "w", encoding="utf-8") as f:
        json.dump(logprob_only, f, indent=4, ensure_ascii=False)

    print("============== Evaluation ==============")
    for k in args.topk_eval:
        save_file = f"./dataset/{prefix}_pass@{k}_details.json"
        score = evaluate_passk(dataset_code_only, lg="python", k=k, save_path=save_file)
        print(f"Pass@{k}: {score:.3f} (details saved to {save_file})")

if __name__ == "__main__":
    main()
