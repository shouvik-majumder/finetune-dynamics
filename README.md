# finetune-dynamics

Measurements of a small language model during LoRA fine-tuning on a narrow task: task accuracy, the local
learning coefficient, and linear-probe accuracy for the task variable, at log-spaced checkpoints.

Status: first runs on both tasks (Qwen2.5-1.5B-Instruct, LoRA rank 16, one seed). Results below are preliminary.

## Results (2026-10-16, one seed)

Held-out task accuracy (exact match) and best-layer probe accuracy at selected checkpoints. Probe chance
levels: 0.10 (cipher), 0.06 (facts).

| Step | Cipher accuracy | Cipher probe | Facts accuracy | Facts probe |
|---|---|---|---|---|
| 0 | 0.00 | 0.43 | 0.00 | 0.09 |
| 35 | 0.00 | 0.71 | 0.07 | 0.09 |
| 50 | 0.00 | 0.87 | 0.05 | 0.09 |
| 75 | 0.35 | 0.91 | 0.07 | 0.09 |
| 100 | 0.64 | 0.93 | 0.07 | 0.09 |
| 150 | 0.79 | 0.94 | 0.35 | 0.18 |
| 200 | 0.98 | 0.95 | 0.75 | 0.42 |
| 400 | 0.99 | 0.95 | 0.96 | 0.73 |
| 600 | 0.99 | 0.96 | 0.96 | 0.78 |

- Cipher: the next cipher letter becomes linearly readable at the answer positions (probe 0.87 at step 50)
  before the model produces correct output (accuracy 0.00 at step 50, 0.35 at step 75). The best probe
  layer moves from layer 4 in the base model to layers 24–28 of 28 during fine-tuning.
- Facts: probe accuracy for the attribute at the last prompt token stays at chance until step 100 and
  then rises together with task accuracy, with no lead.
- LLC (γ = 10⁵, ε = 10⁻⁵, nβ = 10³, 3 chains). Cipher: negative at checkpoints before step 50, where the
  loss is far from a minimum and the sampler descends rather than samples, a large positive and
  high-variance value at steps 75–100 (93 ± 62, 63 ± 21) during the accuracy transition, and small
  positive values (1–6) after convergence. Facts: negative through step 150 (training loss 0.65 at
  step 150 against 0.004 at step 600), small positive values (2–3) at steps 200–400 while accuracy
  rises from 0.75 to 0.96, and about 0 at step 600. The facts transition shows no peak comparable to
  the cipher transition. The estimator is only interpretable where the loss trace plateaus, so the
  negative values are not LLC measurements.

Figures: `figures/cipher_dynamics.png`, `figures/facts_dynamics.png`.

## Question

Over pre-training of small transformers, the local learning coefficient rises while a new computation is
built and falls while an existing one is compressed ([llc](https://github.com/shouvik-majumder/llc)).
The same measurement is applied here to fine-tuning of a 1–2B-parameter model on two tasks, one that
requires a computation the base model does not perform and one that requires only new associations, to
determine how the coefficient, a probe for the task variable, and task accuracy move over fine-tuning steps.

## Methods

- **Fine-tuning.** Continued training of an already-trained model on a small dataset for one task, so
  that its behavior on that task changes. Updating all 1.5 billion weights needs memory for a gradient and
  optimizer state per weight and does not fit on one 24 GB GPU.
- **LoRA** (low-rank adaptation). The base weights stay fixed. For each chosen weight matrix W (for
  example 1536 × 1536), training learns a correction ΔW constrained to have rank r = 16, written as the
  product of a 1536 × 16 matrix B and a 16 × 1536 matrix A, and the layer uses W + BA. The correction
  therefore lives in a 16-dimensional subspace, and the number of trained values per matrix drops from
  2.4 million to about 49 thousand. Across the model about 1% of the parameters train. The set of all A and
  B matrices is the adapter, a file small enough to save at every checkpoint.
- **Local learning coefficient (LLC).** A quantity from singular learning theory that measures how
  degenerate the loss basin around a set of weights is: how fast the volume of weights with loss within ε of
  the minimum shrinks as ε → 0. Lower values mean a flatter, more degenerate basin and a simpler model in
  the theory's sense. The estimator samples weights near the checkpoint with stochastic-gradient Langevin
  dynamics (SGLD) and reads the coefficient from the average loss increase. Here only the LoRA parameters
  are sampled, so values are comparable across checkpoints of one run and not to full-parameter estimates.
- **Linear probe.** A logistic-regression classifier trained to predict the task variable (the cipher image
  of a letter, or a person's profession) from the residual-stream activations at the last prompt token, one
  classifier per layer. Its held-out accuracy measures how linearly available the variable is at that layer.
- **Completion-only loss.** The fine-tuning loss is computed on the assistant's answer tokens only. Prompt
  tokens are masked out.

## Design

| Element | Choice |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct` by default. `google/gemma-2-2b-it` as an option (gated, needs `HF_TOKEN`) |
| Fine-tuning | LoRA, rank 16, on attention and MLP projections, bf16, one 24 GB GPU |
| Task A, computation | Monoalphabetic substitution cipher: encode an English sentence with a fixed letter permutation. Held-out sentences |
| Task B, association | Synthetic facts: 200 fictional people, each with a profession and a city, asked in paraphrased question templates. Held-out templates |
| Checkpoints | LoRA adapters at steps 0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600 |
| Measurements per checkpoint | Task accuracy (exact match and character accuracy for A, answer accuracy for B) · LLC over the LoRA parameters on the training loss · per-layer linear-probe accuracy |
| Evaluation harness | `fd/inspect_task.py` defines the cipher evaluation as an [Inspect AI](https://inspect.aisi.org.uk) task on the same test set |

Task A requires the model to learn a letter-to-letter map and apply it character by character, which the
base model cannot do. Task B requires storing 400 name–attribute pairs. The pair is the fine-tuning
analogue of the RRXOR and Mess3 processes in the llc repository.

## Scripts

| Script | Function |
|---|---|
| `scripts/01_make_data.py --task cipher\|facts` | Writes train and test sets with probe labels to `data/<task>/` |
| `scripts/02_finetune.py --task <task>` | LoRA fine-tuning. Adapters to `runs/<task>/<model>/ckpt-<step>/`, loss log to `loss.csv` |
| `scripts/03_eval.py --task <task>` | Task accuracy at every checkpoint, to `results/<task>/eval.csv` |
| `scripts/04_llc.py --task <task> [--calibrate]` | LLC at every checkpoint, to `results/<task>/llc.csv`. `--calibrate` runs a hyperparameter grid on the final checkpoint |
| `scripts/05_probe.py --task <task>` | Per-layer probe accuracy at every checkpoint, to `results/<task>/probe.csv` |
| `scripts/06_plots.py --task <task>` | Accuracy, LLC, and best-layer probe accuracy against fine-tuning step, to `figures/` |

Settings are in `fd/config.py`. Step 0 is the base model.

## Environment

```bash
conda create -n fd python=3.12 -y
conda activate fd
pip install torch --index-url https://download.pytorch.org/whl/cu128   # or the build for your system
pip install -r requirements.txt
cp .env.example .env    # HF_TOKEN is needed for Gemma only
```

## LLC estimation

The sampler is an SGLD implementation in `fd/llc.py` (the same update as devinterp's SGLD), run over the LoRA parameters with the base weights frozen, on the
completion-token loss of training batches. Step size, localization, and inverse temperature are chosen from
loss-trace diagnostics on the final checkpoint (`--calibrate`), as in the llc repository. Only comparisons
across checkpoints at fixed settings are reported.

## License

MIT; see `LICENSE`.
