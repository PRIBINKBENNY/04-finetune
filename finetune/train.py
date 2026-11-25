"""QLoRA supervised fine-tuning with Hugging Face transformers + PEFT + TRL.

    python -m finetune.train --config configs/qlora_t4.yaml
    python -m finetune.train --config configs/qlora_t4.yaml --dry-run   # print the plan, no GPU needed
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class LoraCfg:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class TrainCfg:
    epochs: float = 2
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    max_seq_len: int = 512
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    logging_steps: int = 10
    eval_steps: int = 100
    seed: int = 42


@dataclass
class Config:
    base_model: str
    output_dir: str
    load_in_4bit: bool = True
    lora: LoraCfg = field(default_factory=LoraCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    train_file: str = "data/train.jsonl"
    eval_file: str = "data/test.jsonl"

    @classmethod
    def load(cls, path: str) -> "Config":
        raw = yaml.safe_load(Path(path).read_text())
        return cls(
            base_model=raw["base_model"],
            output_dir=raw["output_dir"],
            load_in_4bit=raw.get("load_in_4bit", True),
            lora=LoraCfg(**raw.get("lora", {})),
            train=TrainCfg(**raw.get("train", {})),
            train_file=raw.get("train_file", "data/train.jsonl"),
            eval_file=raw.get("eval_file", "data/test.jsonl"),
        )

    def plan(self, n_train: int) -> dict:
        eff_batch = self.train.batch_size * self.train.grad_accum
        steps = int(n_train * self.train.epochs / eff_batch)
        return {"base_model": self.base_model, "qlora_4bit": self.load_in_4bit, "lora_r": self.lora.r,
                "train_examples": n_train, "effective_batch": eff_batch, "optimizer_steps": steps,
                "output_dir": self.output_dir}


def _rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def train(cfg: Config) -> str:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16  # T4 -> fp16

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, quantization_config=quant,
                                                 dtype=dtype, device_map="auto")

    train_ds = Dataset.from_list([{"messages": r["messages"]} for r in _rows(cfg.train_file)])
    eval_ds = Dataset.from_list([{"messages": r["messages"]} for r in _rows(cfg.eval_file)[:100]])

    peft_cfg = LoraConfig(r=cfg.lora.r, lora_alpha=cfg.lora.alpha, lora_dropout=cfg.lora.dropout,
                          target_modules=cfg.lora.target_modules, task_type="CAUSAL_LM")
    t = cfg.train
    args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=t.epochs,
        learning_rate=t.lr,
        per_device_train_batch_size=t.batch_size,
        per_device_eval_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accum,
        warmup_steps=t.warmup_ratio,  # transformers 5: a float in [0,1) is a ratio
        lr_scheduler_type=t.scheduler,
        logging_steps=t.logging_steps,
        eval_strategy="steps",
        eval_steps=t.eval_steps,
        save_strategy="epoch",
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        max_length=t.max_seq_len,
        seed=t.seed,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         peft_config=peft_cfg, processing_class=tok)
    trainer.train()
    adapter_dir = str(Path(cfg.output_dir) / "adapter")
    trainer.save_model(adapter_dir)
    tok.save_pretrained(adapter_dir)
    Path(cfg.output_dir, "train_log.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"saved LoRA adapter to {adapter_dir}")
    return adapter_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/qlora_t4.yaml")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    cfg = Config.load(a.config)
    n = len(_rows(cfg.train_file)) if Path(cfg.train_file).exists() else 0
    print(json.dumps(cfg.plan(n), indent=2))
    if a.dry_run:
        return 0
    if n == 0:
        raise SystemExit(f"no training data at {cfg.train_file}; run `python -m finetune.data` first")
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
