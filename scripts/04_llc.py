"""Local learning coefficient at every checkpoint, estimated by SGLD over the LoRA parameters.

Usage: python scripts/04_llc.py --task cipher [--model qwen] [--calibrate]

Writes results/<task>/llc.csv and the loss traces to results/<task>/llc_traces.json. With --calibrate, runs
a grid over (lr, localization, nbeta) on the final checkpoint and writes results/<task>/llc_calibration.csv
and llc_calibration_traces.json instead. Pick the setting whose loss trace plateaus and set it in
fd/config.py before the full run.

The estimator is fd/llc.py (own SGLD implementation, same update as devinterp's SGLD). Only LoRA
parameters are sampled; base weights stay frozen.
"""

import argparse
import csv
import itertools
import json
import random
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fd.config import DATA, DEFAULT_MODEL, RESULTS, RUNS, LLCConfig
from fd.llc import estimate_llc
from fd.modeling import checkpoint_dirs, collate, encode_example, load_checkpoint, load_tokenizer
from fd.tasks import read_jsonl


def lora_params(model):
    ps = []
    for name, p in model.named_parameters():
        p.requires_grad = "lora_" in name
        if p.requires_grad:
            ps.append(p)
    return ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--burnin", type=int, default=None)
    ap.add_argument("--grid", default=None,
                    help='calibration settings as "lr,localization,nbeta;lr,localization,nbeta;..." (overrides the default grid)')
    ap.add_argument("--tag", default="", help="suffix for the calibration output files")
    args = ap.parse_args()

    cfg = LLCConfig()
    if args.chains:
        cfg.n_chains = args.chains
    if args.draws:
        cfg.n_draws = args.draws
    if args.burnin is not None:
        cfg.n_burnin = args.burnin
    tok = load_tokenizer(args.model)
    rows = read_jsonl(DATA / args.task / "train.jsonl")
    rng = random.Random(0)
    examples = [encode_example(tok, r["prompt"], r["completion"], 192) for r in rng.sample(rows, cfg.batch_size * cfg.n_batches)]
    batches = [collate(tok, examples[i:i + cfg.batch_size]) for i in range(0, len(examples), cfg.batch_size)]
    run_dir = RUNS / args.task / args.model
    out_dir = RESULTS / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = checkpoint_dirs(run_dir)

    if args.calibrate:
        step, ckpt = ckpts[-1]
        model = load_checkpoint(args.model, ckpt)
        params = lora_params(model)
        print("sampled parameters:", sum(p.numel() for p in params))
        # The posterior at nbeta explores loss increases of about llc / nbeta, so nbeta has to be large
        # enough that the chains stay near the minimum. The step size is set by lr * nbeta.
        grid = [(prod / nb, loc, nb) for nb in (1e3, 1e4, 1e5) for prod in (1e-3, 1e-2) for loc in (100.0, 1000.0)]
        if args.grid:
            grid = [tuple(float(x) for x in s.split(",")) for s in args.grid.split(";")]
        traces = {}
        with open(out_dir / f"llc_calibration{args.tag}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "lr", "localization", "nbeta", "llc_mean", "llc_std", "loss_star",
                        "loss_first20", "loss_last20"])
            for lr, loc, nb in tqdm(grid, desc="grid"):
                r = estimate_llc(model, batches, params, n_chains=2, n_draws=cfg.n_draws, n_burnin=cfg.n_burnin,
                                 lr=lr, localization=loc, nbeta=nb)
                t0 = r.traces[0]
                first = sum(t0[cfg.n_burnin:cfg.n_burnin + 20]) / 20
                last = sum(t0[-20:]) / 20
                w.writerow([step, lr, loc, nb, f"{r.mean:.4f}", f"{r.std:.4f}", f"{r.loss_star:.4f}",
                            f"{first:.4f}", f"{last:.4f}"])
                f.flush()
                traces[f"lr={lr},loc={loc},nbeta={nb}"] = r.traces
                print(f"lr={lr:.1e} loc={loc} nbeta={nb:.0e}: llc {r.mean:.3f} ± {r.std:.3f}, loss {first:.3f} -> {last:.3f} (L* {r.loss_star:.3f})")
        (out_dir / f"llc_calibration_traces{args.tag}.json").write_text(json.dumps(traces))
        return

    traces = {}
    with open(out_dir / "llc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "llc_mean", "llc_std", "loss_star", "lr", "localization", "nbeta", "chains", "draws"])
        for step, ckpt in tqdm(ckpts, desc="checkpoints"):
            if ckpt is None:
                # Step 0: a fresh adapter (B = 0, so the model equals the base model) gives the starting point
                # in the same parameterization as the later checkpoints.
                from fd.config import LoraConfig
                from fd.modeling import add_lora, load_base
                model = add_lora(load_base(args.model), LoraConfig())
                model.eval()
            else:
                model = load_checkpoint(args.model, ckpt)
            params = lora_params(model)
            r = estimate_llc(model, batches, params, n_chains=cfg.n_chains, n_draws=cfg.n_draws,
                             n_burnin=cfg.n_burnin, lr=cfg.lr, localization=cfg.localization, nbeta=cfg.nbeta)
            w.writerow([step, f"{r.mean:.4f}", f"{r.std:.4f}", f"{r.loss_star:.4f}", cfg.lr, cfg.localization,
                        cfg.nbeta, cfg.n_chains, cfg.n_draws])
            f.flush()
            traces[str(step)] = r.traces
            print(f"step {step}: llc {r.mean:.3f} ± {r.std:.3f}, loss* {r.loss_star:.3f}")
            del model
            torch.cuda.empty_cache()
    (out_dir / "llc_traces.json").write_text(json.dumps(traces))


if __name__ == "__main__":
    main()
