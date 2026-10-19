"""Accuracy, LLC, and best-layer probe accuracy against fine-tuning step.

Usage: python scripts/06_plots.py --task cipher
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fd.config import FIGURES, RESULTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    args = ap.parse_args()
    d = RESULTS / args.task
    ev = pd.read_csv(d / "eval.csv")
    llc = pd.read_csv(d / "llc.csv") if (d / "llc.csv").exists() else None
    pr = pd.read_csv(d / "probe.csv") if (d / "probe.csv").exists() else None

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(ev["step"], ev["exact_match"], "o-", label="exact match")
    if "char_accuracy" in ev:
        axes[0].plot(ev["step"], ev["char_accuracy"], "s--", label="character accuracy")
    axes[0].set(xlabel="fine-tuning step", ylabel="accuracy", xscale="symlog", title="Task accuracy")
    axes[0].legend()
    if llc is not None:
        axes[1].errorbar(llc["step"], llc["llc_mean"], yerr=llc["llc_std"], fmt="o-")
        axes[1].set_yscale("symlog", linthresh=10)
        axes[1].axhline(0, color="grey", ls=":", lw=1)
        axes[1].set(xlabel="fine-tuning step", ylabel="LLC (LoRA parameters)", xscale="symlog", title="Local learning coefficient")
    if pr is not None:
        best = pr.groupby("step")["accuracy"].max()
        axes[2].plot(best.index, best.values, "o-", label="best layer")
        axes[2].axhline(pr["chance"].iloc[0], color="grey", ls=":", label="chance")
        axes[2].set(xlabel="fine-tuning step", ylabel="probe accuracy", xscale="symlog", title="Linear probe")
        axes[2].legend()
    for ax in axes:
        ax.set_xlim(-0.5, ev["step"].max() * 1.4)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / f"{args.task}_dynamics.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
