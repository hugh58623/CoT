# programmer_bcb.py
import argparse, os, json, copy, concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

import sys
sys.path.append('./CodeGeeX/')
from codegeex.benchmark.utils import IMPORT_HELPER
from codegeex.benchmark.execution import check_correctness


class StopAfterOneCodeBlock(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tk = tokenizer
        self.open_seen = False
        self.open_ids = self.tk.encode("```python", add_special_tokens=False)
        self.close_ids = self.tk.encode("```", add_special_tokens=False)
    def __call__(self, input_ids, scores, **kwargs):
        ids = input_ids[0].tolist()
        if (not self.open_seen and len(ids) >= len(self.open_ids)
                and ids[-len(self.open_ids):] == self.open_ids):
            self.open_seen = True
        elif self.open_seen and len(ids) >= len(self.close_ids) and ids[-len(self.close_ids):] == self.close_ids:
            return True
        return False


def strip_code_fence(s: str) -> str:
    if "```python" in s:
        s = s.split("```python", 1)[1].split("```", 1)[0]
    elif "```" in s:
        s = s.split("```", 1)[1].split("```", 1)[0]
    return s.strip()


def fetch_completion_hf(sample, model, tokenizer, model_path, fewshot, times=10, topk=5, max_new_tokens=512, run_tag="nocot"):
    prompt_src = sample.get("prompt") or sample.get("complete_prompt") or ""
    if "qwen" in model_path.lower():
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{fewshot}\n\n{prompt_src}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif "deepseek" in model_path.lower():
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": f"{fewshot}\n\n{prompt_src}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    elif "gpt" in model_path.lower():
        # Decide reasoning level based on run_tag
        if run_tag.lower() == "cot":
            reasoning_level = "medium"
        else:  # nocot
            reasoning_level = "low"
        system_content = (
            "You are a helpful coding assistant.\n"
            f"Reasoning: {reasoning_level}"
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"{fewshot}\n\n{prompt_src}"}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    # elif "codellama" in model_path.lower():
    #     text = f"<s>[INST] {fewshot}\n\n{prompt_src}\n[/INST]"
    else:
        text = f"{fewshot}\n\n{prompt_src}\n"

    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    stoppers = StoppingCriteriaList([StopAfterOneCodeBlock(tokenizer)])
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=stoppers,
        num_return_sequences=times,
    )
    if run_tag and "cot" in run_tag.lower():
        gen_kwargs.update(dict(do_sample=True, temperature=1.0, top_p=0.9))
    else:
        gen_kwargs.update(dict(do_sample=False, temperature=0.0))

    outputs = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]

    raw_list, code_list, logs = [], [], []
    for i in range(times):
        seq = outputs.sequences[i]
        completion_raw = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        completion_code = strip_code_fence(completion_raw)

        token_ids = seq[input_len:]
        step_scores = [step[i] for step in outputs.scores]
        token_logprobs, linewise = [], []
        at_line_start = True
        for step, logits in enumerate(step_scores):
            logprobs = F.log_softmax(logits, dim=-1)
            tok_id = token_ids[step]
            tok = tokenizer.decode([tok_id])

            if at_line_start and tok.strip():
                topk_vals, topk_ids = torch.topk(logprobs, k=topk)
                probs = topk_vals.exp().cpu().tolist()
                toks = [tokenizer.decode([tid.item()]) for tid in topk_ids]
                entropy = float(-sum(p * (p if p <= 0 else torch.tensor(p).log().item()) for p in probs))  # simple guard
                linewise.append(dict(
                    step=step,
                    line_start_token=tok,
                    topk_tokens=toks,
                    topk_probs=probs,
                    entropy=entropy,
                    prob_diff=(probs[0] - probs[1] if len(probs) > 1 else 0.0)
                ))
                at_line_start = False

            if tok in ["\n", "Ċ"]:
                at_line_start = True

            token_logprobs.append(float(logprobs[tok_id].item()))

        logs.append({
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "token_logprobs": token_logprobs,
            "sum_logprob": float(sum(token_logprobs)),
            "avg_logprob": float(sum(token_logprobs)/len(token_logprobs)) if token_logprobs else None,
            "linewise_metrics": linewise
        })
        raw_list.append(completion_raw)
        code_list.append(completion_code)

    out = dict(sample)
    out["raw_completion_list"] = raw_list
    out["completion_list"] = code_list
    out["logprob_list"] = logs
    return out


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
            exec_sample = {"test_code": exec_code,
                           "prompt": sample.get("prompt") or sample.get("complete_prompt") or "",
                           "generation": code}
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt_path", type=str, default="./prompts/mhpp_prompt_nocot.txt")
    parser.add_argument("--dataset_file", type=str, required=True)
    parser.add_argument("--run_tag", type=str, default=None)
    parser.add_argument("--times", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--topk_eval", type=int, nargs="+", default=[1, 5, 10])
    args = parser.parse_args()

    model_path = args.model_path
    prompt_path = args.prompt_path
    dataset_file = args.dataset_file
    run_tag = args.run_tag or os.path.splitext(os.path.basename(prompt_path))[0]

    print(f"Loading local model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    dtype = torch.bfloat16 if "gpt-oss" in model_path.lower() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto",
    )

    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            construct_few_shot_prompt = f.read()
    else:
        construct_few_shot_prompt = ""

    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    print("Running completion fetch...")
    new_dataset = [None] * len(dataset)
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {executor.submit(fetch_completion_hf, copy.deepcopy(dataset[i]), model, tokenizer, model_path,
                                   construct_few_shot_prompt, args.times, 5, args.max_new_tokens, run_tag): i
                   for i in range(len(dataset))}
        for fut in concurrent.futures.as_completed(futures):
            i = futures[fut]
            try:
                new_dataset[i] = fut.result()
            except Exception as e:
                print(repr(e))
                new_dataset[i] = dataset[i]

    os.makedirs("./dataset", exist_ok=True)
    model_name = os.path.basename(model_path.rstrip("/"))
    dataset_name = os.path.splitext(os.path.basename(dataset_file))[0]
    prefix = f"{model_name}_{dataset_name}_python_{run_tag}"

    # 1) full data with logprobs
    with open(f"./dataset/{prefix}_raw.json", "w", encoding="utf-8") as f:
        json.dump(new_dataset, f, indent=4, ensure_ascii=False)

    # 2) pruned data without logprobs
    dataset_code_only = []
    for item in new_dataset:
        d = dict(item)
        d.pop("raw_completion_list", None)
        d.pop("logprob_list", None)
        dataset_code_only.append(d)
    with open(f"./dataset/{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(dataset_code_only, f, indent=4, ensure_ascii=False)

    # 3) only logprobs
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
