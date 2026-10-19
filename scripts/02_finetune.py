"""LoRA fine-tuning with log-spaced checkpoints.

Usage: python scripts/02_finetune.py --task cipher [--model qwen]

Writes runs/<task>/<model>/ckpt-<step>/ (LoRA adapters) and runs/<task>/<model>/loss.csv.
Step 0 is recorded as an empty checkpoint directory, so later scripts treat it as the base model.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fd.config import CHECKPOINT_STEPS, DATA, DEFAULT_MODEL, RUNS, LoraConfig, TrainConfig
from fd.modeling import add_lora, collate, encode_example, load_base, load_tokenizer
from fd.tasks import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    tcfg = TrainConfig()
    if args.max_steps:
        tcfg.max_steps = args.max_steps
    torch.manual_seed(tcfg.seed)
    random.seed(tcfg.seed)

    tok = load_tokenizer(args.model)
    model = add_lora(load_base(args.model), LoraConfig())
    model.print_trainable_parameters()

    rows = read_jsonl(DATA / args.task / "train.jsonl")
    examples = [encode_example(tok, r["prompt"], r["completion"], tcfg.max_length) for r in rows]

    run_dir = RUNS / args.task / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt-0").mkdir(exist_ok=True)  # base model marker

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / tcfg.warmup_steps))

    save_at = {s for s in CHECKPOINT_STEPS if 0 < s <= tcfg.max_steps}
    log = open(run_dir / "loss.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["step", "loss"])

    model.train()
    step, i = 0, 0
    order = list(range(len(examples)))
    random.shuffle(order)
    pbar = tqdm(total=tcfg.max_steps, desc="steps")
    while step < tcfg.max_steps:
        opt.zero_grad(set_to_none=True)
        acc_loss = 0.0
        for _ in range(tcfg.grad_accum):
            if i + tcfg.batch_size > len(order):
                random.shuffle(order)
                i = 0
            batch = [examples[j] for j in order[i:i + tcfg.batch_size]]
            i += tcfg.batch_size
            out = model(**collate(tok, batch))
            (out.loss / tcfg.grad_accum).backward()
            acc_loss += out.loss.item() / tcfg.grad_accum
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        writer.writerow([step, f"{acc_loss:.5f}"])
        pbar.set_postfix(loss=f"{acc_loss:.3f}")
        pbar.update(1)
        if step in save_at:
            model.save_pretrained(str(run_dir / f"ckpt-{step}"))
    log.close()
    print("done:", run_dir)


if __name__ == "__main__":
    main()
