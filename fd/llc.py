"""SGLD estimator of the local learning coefficient over a subset of parameters.

λ̂ = nβ · (E[L(w)] − L(w*)), with w sampled by stochastic-gradient Langevin dynamics localised at w*:

    w ← w − (ε/2) · (nβ ∇L(w) + γ (w − w*)) + N(0, ε)

Only the parameters passed in (the LoRA adapters) move. Base weights stay fixed. The reference value
L(w*) is the mean loss over the same batches the chains see.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class LLCResult:
    mean: float
    std: float
    chain_llcs: list[float]
    loss_star: float
    traces: list[list[float]]  # per-chain loss trace, including burn-in


@torch.no_grad()
def _reset(params, w_star):
    for p, w in zip(params, w_star):
        p.copy_(w)


def estimate_llc(model, batches: list[dict], params: list[torch.nn.Parameter], *, n_chains: int,
                 n_draws: int, n_burnin: int, lr: float, localization: float, nbeta: float,
                 seed: int = 0) -> LLCResult:
    """`batches` is a list of collated batches. Chains cycle through them in order."""
    gen = torch.Generator(device=params[0].device).manual_seed(seed)
    w_star = [p.detach().clone() for p in params]

    model.eval()  # dropout off. LoRA dropout would otherwise add noise to the loss.
    with torch.no_grad():
        loss_star = float(np.mean([model(**b).loss.item() for b in batches]))

    chain_llcs, traces = [], []
    for c in range(n_chains):
        _reset(params, w_star)
        trace = []
        for step in range(n_burnin + n_draws):
            batch = batches[step % len(batches)]
            model.zero_grad(set_to_none=True)
            loss = model(**batch).loss
            loss.backward()
            with torch.no_grad():
                for p, w in zip(params, w_star):
                    noise = torch.randn(p.shape, generator=gen, device=p.device, dtype=p.dtype) * math.sqrt(lr)
                    drift = nbeta * p.grad + localization * (p - w)
                    p.add_(-(lr / 2) * drift + noise)
            trace.append(loss.item())
            if not math.isfinite(trace[-1]):
                break
        draws = [x for x in trace[n_burnin:] if math.isfinite(x)]
        chain_llcs.append(nbeta * (float(np.mean(draws)) - loss_star) if draws else float("nan"))
        traces.append(trace)
    _reset(params, w_star)
    model.zero_grad(set_to_none=True)
    arr = np.array(chain_llcs, dtype=float)
    return LLCResult(float(np.nanmean(arr)), float(np.nanstd(arr)), chain_llcs, loss_star, traces)
