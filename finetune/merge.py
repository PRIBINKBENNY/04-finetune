"""Merge the LoRA adapter into the base weights (for serving without PEFT) and optionally push to the Hub.

    python -m finetune.merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter outputs/.../adapter --out merged/
    python -m finetune.merge ... --push your-hf-username/qwen0.5b-job-extractor
"""
from __future__ import annotations

import argparse


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--out", default="merged")
    p.add_argument("--push")
    a = p.parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float16)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    tok = AutoTokenizer.from_pretrained(a.adapter)
    model.save_pretrained(a.out, safe_serialization=True)
    tok.save_pretrained(a.out)
    print(f"merged model saved to {a.out}")
    if a.push:
        model.push_to_hub(a.push)
        tok.push_to_hub(a.push)
        print(f"pushed to https://huggingface.co/{a.push}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
