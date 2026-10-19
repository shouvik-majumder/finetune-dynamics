"""Generate train and test sets for a task.

Usage: python scripts/01_make_data.py --task cipher
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fd.config import DATA, DataConfig
from fd.tasks import TASKS, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS), required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = DataConfig(seed=args.seed)
    train, test, meta = TASKS[args.task](cfg)
    out = DATA / args.task
    write_jsonl(out / "train.jsonl", train)
    write_jsonl(out / "test.jsonl", test)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"{args.task}: {len(train)} train, {len(test)} test -> {out}")
    print("example:", json.dumps(train[0], ensure_ascii=False))


if __name__ == "__main__":
    main()
