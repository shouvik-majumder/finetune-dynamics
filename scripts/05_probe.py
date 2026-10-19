"""Per-layer linear probe for the task variable at every checkpoint.

Usage: python scripts/05_probe.py --task cipher [--model qwen]

cipher: the prompt and the correct answer are run through the model (teacher forcing). At each answer
position the residual stream is collected, and the label is the first letter of the next answer token, so
the probe asks whether the next cipher letter is linearly readable just before it is emitted.

facts: the residual stream at the last prompt token, label = the attribute to be answered.

One logistic-regression probe per layer with a held-out split. Writes results/<task>/probe.csv with
columns step, layer, accuracy, chance, n.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fd.config import DATA, DEFAULT_MODEL, RESULTS, RUNS, ProbeConfig
from fd.modeling import chat_prompt, checkpoint_dirs, load_checkpoint, load_tokenizer
from fd.tasks import read_jsonl


@torch.no_grad()
def hidden_at_positions(model, tok, texts: list[str], positions: list[list[int]], batch_size: int = 8):
    """For each text, hidden states (layers+1, d) at the listed token positions. Returns a list of arrays."""
    out = []
    tok.padding_side = "right"
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=1)  # (b, L+1, T, d)
        for j, pos in enumerate(positions[i:i + batch_size]):
            out.append(hs[j, :, pos, :].float().cpu().numpy())  # (L+1, len(pos), d)
    return out


def cipher_items(tok, items, n_items, rng):
    """Teacher-forced answer positions and next-letter labels."""
    texts, positions, labels = [], [], []
    for it in rng.sample(items, min(n_items, len(items))):
        prompt_ids = tok(chat_prompt(tok, it["prompt"]), add_special_tokens=False)["input_ids"]
        ans_ids = tok(it["completion"], add_special_tokens=False)["input_ids"]
        pos, lab = [], []
        for k in range(len(ans_ids) - 1):
            nxt = tok.decode([ans_ids[k + 1]]).strip()
            if nxt and nxt[0].isalpha():
                pos.append(len(prompt_ids) + k)   # position whose next token is ans_ids[k+1]
                lab.append(nxt[0].lower())
        if pos:
            texts.append(tok.decode(prompt_ids + ans_ids))
            positions.append(pos)
            labels.append(lab)
    return texts, positions, labels


def facts_items(tok, items, n_items, rng):
    """One prompt per (person, attribute), grouped by person so that a name never appears in both probe
    train and test sets. Otherwise the probe reads the person's identity instead of the attribute."""
    by_person = {}
    for it in items:
        by_person.setdefault(it["name"], {}).setdefault(it["attribute"], it)
    people = rng.sample(sorted(by_person), min(n_items // 2, len(by_person)))
    sel = [by_person[p][a] for p in people for a in sorted(by_person[p])]
    texts = [chat_prompt(tok, it["prompt"]) for it in sel]
    positions = [[len(tok(t, add_special_tokens=False)["input_ids"]) - 1] for t in texts]
    labels = [[it["probe_label"]] for it in sel]
    return texts, positions, labels, [it["name"] for it in sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    cfg = ProbeConfig()
    rng = random.Random(cfg.seed)
    tok = load_tokenizer(args.model)
    items = read_jsonl(DATA / args.task / "test.jsonl") + read_jsonl(DATA / args.task / "train.jsonl")
    if args.task == "cipher":
        texts, positions, labels = cipher_items(tok, items, 120, rng)
        group_of_text = list(range(len(texts)))          # one group per sentence
    else:
        texts, positions, labels, names = facts_items(tok, items, cfg.n_items, rng)
        name_ids = {n: i for i, n in enumerate(sorted(set(names)))}
        group_of_text = [name_ids[n] for n in names]     # one group per person
    y = np.array([l for ls in labels for l in ls])
    classes, counts = np.unique(y, return_counts=True)
    chance = counts.max() / len(y)
    # held-out split by group, so that no sentence or person appears in both probe train and test
    groups = np.array([group_of_text[i] for i, ls in enumerate(labels) for _ in ls])
    unique_groups = sorted(set(groups))
    test_groups = set(rng.sample(unique_groups, int(cfg.test_fraction * len(unique_groups))))
    te = np.array([g in test_groups for g in groups])

    run_dir = RUNS / args.task / args.model
    out_dir = RESULTS / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probe.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "layer", "accuracy", "chance", "n"])
        for step, ckpt in tqdm(checkpoint_dirs(run_dir), desc="checkpoints"):
            model = load_checkpoint(args.model, ckpt)
            feats = hidden_at_positions(model, tok, texts, positions)
            X = np.concatenate([fa.transpose(1, 0, 2) for fa in feats])  # (n_positions, L+1, d)
            best = 0.0
            for layer in (cfg.layers or range(X.shape[1])):
                clf = LogisticRegression(max_iter=3000, C=0.5)
                clf.fit(X[~te, layer, :], y[~te])
                acc = clf.score(X[te, layer, :], y[te])
                best = max(best, acc)
                w.writerow([step, layer, f"{acc:.4f}", f"{chance:.4f}", len(y)])
            f.flush()
            print(f"step {step}: best layer accuracy {best:.3f} (chance {chance:.3f}, n={len(y)})")
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
