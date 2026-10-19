"""Settings for data generation, fine-tuning, and measurement."""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

MODELS = {
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    "gemma": "google/gemma-2-2b-it",
}
DEFAULT_MODEL = "qwen"

# Checkpoints are saved at these optimizer steps (log-spaced, plus the end).
CHECKPOINT_STEPS = [0, 5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 600]


@dataclass
class DataConfig:
    seed: int = 0
    # cipher
    n_train_sentences: int = 4000
    n_test_sentences: int = 300
    min_words: int = 4
    max_words: int = 9
    # facts
    n_people: int = 200
    n_train_templates: int = 4
    n_test_templates: int = 2


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class TrainConfig:
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    max_steps: int = CHECKPOINT_STEPS[-1]
    max_length: int = 192
    weight_decay: float = 0.0
    warmup_steps: int = 10
    seed: int = 0


@dataclass
class EvalConfig:
    n_items: int = 300
    max_new_tokens: int = 64
    batch_size: int = 16


@dataclass
class LLCConfig:
    """SGLD over the LoRA parameters. Values are starting points for the calibration sweep."""
    # Chosen from the calibration grids of 2026-10-16 on the cipher run: the loss trace plateaus at
    # about 0.004 nats above L* with these values. Weaker localization (gamma <= 1e4) never equilibrates
    # within a few hundred steps, and smaller steps with large nbeta act as an optimizer (loss below L*).
    n_chains: int = 3
    n_draws: int = 150
    n_burnin: int = 100
    lr: float = 1e-5          # epsilon
    localization: float = 1e5  # gamma
    nbeta: float = 1e3        # n * beta
    batch_size: int = 8
    n_batches: int = 100      # training batches used by the sampler


@dataclass
class ProbeConfig:
    n_items: int = 400
    layers: tuple = ()        # empty: all layers
    test_fraction: float = 0.3
    seed: int = 0
