"""Inspect AI task for the cipher evaluation.

Run against a local Hugging Face model (base or merged adapter), for example:

    inspect eval fd/inspect_task.py@cipher --model hf/Qwen/Qwen2.5-1.5B-Instruct

Scores exact match against the test set in data/cipher/test.jsonl. To evaluate a LoRA checkpoint, merge the
adapter into a saved model directory first (PeftModel.merge_and_unload) and pass that directory as the model.
"""

from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.scorer import exact
from inspect_ai.solver import generate

DATA = Path(__file__).resolve().parent.parent / "data"


def to_sample(record: dict) -> Sample:
    return Sample(input=record["prompt"], target=record["completion"])


@task
def cipher() -> Task:
    return Task(
        dataset=json_dataset(str(DATA / "cipher" / "test.jsonl"), sample_fields=to_sample),
        solver=[generate()],
        scorer=exact(),
    )
