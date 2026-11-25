"""Before/after evaluation on the held-out test set.

    # local transformers (base model, then base + adapter)
    python -m finetune.evaluate --backend hf --model Qwen/Qwen2.5-0.5B-Instruct --tag base
    python -m finetune.evaluate --backend hf --model Qwen/Qwen2.5-0.5B-Instruct \
        --adapter outputs/qwen0.5b-jobs-lora/adapter --tag finetuned

    # any OpenAI-compatible server, e.g. vLLM serving the adapter (see 03-deploy):
    #   vllm serve Qwen/Qwen2.5-0.5B-Instruct --enable-lora --lora-modules jobs=outputs/.../adapter
    python -m finetune.evaluate --backend openai --base-url http://localhost:8000/v1 --model jobs --tag finetuned

    python -m finetune.evaluate --compare results/base.json results/finetuned.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

from .data import to_messages
from .metrics import Report

Predictor = Callable[[list[list[dict]]], list[str]]


def hf_predictor(model_name: str, adapter: str | None, batch_size: int = 16, max_new_tokens: int = 200) -> Predictor:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(adapter or model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def predict(batch_msgs: list[list[dict]]) -> list[str]:
        outs = []
        for i in range(0, len(batch_msgs), batch_size):
            chunk = batch_msgs[i : i + batch_size]
            prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            outs += tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs

    return predict


def openai_predictor(base_url: str, model: str, api_key: str = "not-needed", workers: int = 16) -> Predictor:
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key)

    def one(msgs):
        r = client.chat.completions.create(model=model, messages=msgs, temperature=0, max_tokens=200)
        return r.choices[0].message.content or ""

    def predict(batch_msgs):
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(one, batch_msgs))

    return predict


def run(predict: Predictor, rows: list[dict]) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    preds = predict([to_messages(r, with_answer=False) for r in rows])
    elapsed = time.perf_counter() - t0
    rep = Report()
    details = []
    for r, p in zip(rows, preds):
        details.append({"text": r["text"], "gold": r["target"], "pred": p, **rep.add(p, r["target"])})
    s = rep.summary()
    s["seconds"] = round(elapsed, 1)
    return s, details


def compare(paths: list[str]) -> str:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    keys = ["valid_json_pct", "exact_match_pct", "field_accuracy_pct", "skills_f1"]
    lines = ["| model | " + " | ".join(keys) + " |", "|---|" + "---:|" * len(keys)]
    for r in runs:
        lines.append(f"| {r['tag']} | " + " | ".join(str(r['summary'][k]) for k in keys) + " |")
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["hf", "openai"], default="hf")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--adapter")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--test-file", default="data/test.jsonl")
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--tag", default="run")
    p.add_argument("--compare", nargs="+")
    p.add_argument("--readme", help="with --compare: write the table into this README between RESULTS markers")
    a = p.parse_args(argv)

    if a.compare:
        table = compare(a.compare)
        print(table)
        if a.readme:
            text = Path(a.readme).read_text()
            start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
            pre, _, rest = text.partition(start)
            _, _, post = rest.partition(end)
            Path(a.readme).write_text(f"{pre}{start}\n{table}\n{end}{post}")
        return 0
    rows = [json.loads(l) for l in Path(a.test_file).read_text().splitlines() if l.strip()][: a.limit]
    predict = hf_predictor(a.model, a.adapter) if a.backend == "hf" else openai_predictor(a.base_url, a.model)
    summary, details = run(predict, rows)
    out = Path("results")
    out.mkdir(exist_ok=True)
    (out / f"{a.tag}.json").write_text(json.dumps({"tag": a.tag, "model": a.model, "adapter": a.adapter,
                                                   "summary": summary, "details": details}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
