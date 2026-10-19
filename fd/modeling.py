"""Model loading, chat formatting, and completion-only loss masking."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from peft import LoraConfig as PeftLoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import MODELS, LoraConfig

load_dotenv()


def load_tokenizer(model_key: str):
    tok = AutoTokenizer.from_pretrained(MODELS[model_key], token=os.getenv("HF_TOKEN"))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_base(model_key: str, device: str = "cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_key], torch_dtype=torch.bfloat16, token=os.getenv("HF_TOKEN"),
        attn_implementation="eager" if model_key == "gemma" else "sdpa",
    )
    return model.to(device)


def add_lora(model, cfg: LoraConfig):
    peft_cfg = PeftLoraConfig(r=cfg.r, lora_alpha=cfg.alpha, lora_dropout=cfg.dropout,
                              target_modules=list(cfg.target_modules), task_type="CAUSAL_LM")
    return get_peft_model(model, peft_cfg)


def load_checkpoint(model_key: str, ckpt_dir: Path | None, device: str = "cuda"):
    """Base model with the LoRA adapter from `ckpt_dir` merged in. `None` returns the base model."""
    model = load_base(model_key, device)
    if ckpt_dir is not None and (ckpt_dir / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, str(ckpt_dir))
    model.eval()
    return model


def chat_prompt(tok, user: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True)


def encode_example(tok, prompt: str, completion: str, max_length: int) -> dict:
    """Token ids with labels = -100 on the prompt, so the loss is on the completion only."""
    prompt_text = chat_prompt(tok, prompt)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tok(completion + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (prompt_ids + completion_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + completion_ids)[:max_length]
    return {"input_ids": ids, "labels": labels}


def collate(tok, batch: list[dict], device: str = "cuda") -> dict:
    n = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id
    ids = torch.full((len(batch), n), pad, dtype=torch.long)
    labels = torch.full((len(batch), n), -100, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        ids[i, :L] = torch.tensor(b["input_ids"])
        labels[i, :L] = torch.tensor(b["labels"])
        attn[i, :L] = 1
    return {"input_ids": ids.to(device), "labels": labels.to(device), "attention_mask": attn.to(device)}


def checkpoint_dirs(run_dir: Path) -> list[tuple[int, Path | None]]:
    """(step, dir) pairs, with step 0 mapped to the base model."""
    out: list[tuple[int, Path | None]] = []
    for d in sorted(run_dir.glob("ckpt-*"), key=lambda p: int(p.name.split("-")[1])):
        step = int(d.name.split("-")[1])
        out.append((step, None if step == 0 else d))
    return out
