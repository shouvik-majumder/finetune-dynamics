"""Task accuracy at every checkpoint.

Usage: python scripts/03_eval.py --task cipher [--model qwen]

Writes results/<task>/eval.csv with exact-match accuracy, and character accuracy for the cipher task.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fd.config import DATA, DEFAULT_MODEL, RESULTS, RUNS, EvalConfig
from fd.modeling import chat_prompt, checkpoint_dirs, load_checkpoint, load_tokenizer
from fd.tasks import read_jsonl


def normalize(s: str) -> str:
    return s.strip().lower().rstrip(".")


def char_accuracy(pred: str, target: str) -> float:
    n = max(len(target), 1)
    return sum(1 for a, b in zip(pred, target) if a == b) / n


@torch.no_grad()
def generate(model, tok, prompts: list[str], cfg: EvalConfig) -> list[str]:
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), cfg.batch_size):
        texts = [chat_prompt(tok, p) for p in prompts[i:i + cfg.batch_size]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=cfg.max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    tok.padding_side = "right"
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n-items", type=int, default=None)
    args = ap.parse_args()

    cfg = EvalConfig()
    if args.n_items:
        cfg.n_items = args.n_items
    tok = load_tokenizer(args.model)
    test = read_jsonl(DATA / args.task / "test.jsonl")[: cfg.n_items]
    run_dir = RUNS / args.task / args.model
    out_dir = RESULTS / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "eval.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "exact_match", "char_accuracy", "n"])
        for step, ckpt in tqdm(checkpoint_dirs(run_dir), desc="checkpoints"):
            model = load_checkpoint(args.model, ckpt)
            preds = generate(model, tok, [t["prompt"] for t in test], cfg)
            em = sum(normalize(p) == normalize(t["completion"]) for p, t in zip(preds, test)) / len(test)
            ca = sum(char_accuracy(normalize(p), normalize(t["completion"])) for p, t in zip(preds, test)) / len(test)
            w.writerow([step, f"{em:.4f}", f"{ca:.4f}", len(test)])
            f.flush()
            print(f"step {step}: exact {em:.3f}, char {ca:.3f}  e.g. {preds[0]!r} vs {test[0]['completion']!r}")
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
