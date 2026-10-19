"""Synthetic fine-tuning tasks.

Each task produces examples of the form {"prompt": str, "completion": str, "probe_label": str, ...}.
The prompt is the user message, the completion the assistant message. `probe_label` is the task
variable the linear probe reads from the residual stream at the last prompt token.
"""

from __future__ import annotations

import json
import random
import string
from pathlib import Path

from .config import DataConfig

# ---------------------------------------------------------------- Task A: substitution cipher

WORDS = """the a an and of to in on at for with from by about as into like through after over between
out against during without before under around among cat dog bird fish horse cow sheep goat lion tiger
bear wolf fox deer rabbit mouse river lake sea ocean mountain hill valley forest field garden road bridge
house door window wall roof floor table chair bed lamp book pen paper letter map clock bell drum flute
song story poem game ball kite boat train plane car bus bike road city town village farm market school
shop park beach island desert cloud rain snow wind storm sun moon star sky night day morning evening
summer winter spring autumn red blue green yellow black white brown grey gold silver small large tall
short long wide deep high low old new young warm cold hot cool soft hard quick slow quiet loud bright
dark clean dirty sweet sour fresh dry wet full empty open closed light heavy early late near far happy
sad calm angry kind brave wise bold run walk jump swim fly climb sit stand sleep wake eat drink cook
read write draw sing dance play work rest look watch listen speak call ask tell give take bring carry
hold drop push pull lift throw catch cut build break fix make grow plant pick find lose keep send""".split()


def make_cipher(rng: random.Random) -> dict[str, str]:
    letters = list(string.ascii_lowercase)
    shuffled = letters[:]
    while True:
        rng.shuffle(shuffled)
        if all(a != b for a, b in zip(letters, shuffled)):  # no fixed points
            break
    return dict(zip(letters, shuffled))


def encode(text: str, cipher: dict[str, str]) -> str:
    return "".join(cipher.get(c, c) for c in text)


def random_sentence(rng: random.Random, cfg: DataConfig) -> str:
    n = rng.randint(cfg.min_words, cfg.max_words)
    return " ".join(rng.choice(WORDS) for _ in range(n))


CIPHER_PROMPT = "Encode the following text with the secret cipher. Reply with the encoded text only.\nText: {text}"


def cipher_examples(cfg: DataConfig) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(cfg.seed)
    cipher = make_cipher(rng)
    seen: set[str] = set()
    items = []
    while len(items) < cfg.n_train_sentences + cfg.n_test_sentences:
        s = random_sentence(rng, cfg)
        if s in seen:
            continue
        seen.add(s)
        items.append({
            "prompt": CIPHER_PROMPT.format(text=s),
            "completion": encode(s, cipher),
            # The probe target: the cipher image of the last letter of the text, which is the next
            # character the model has to produce after the opening of the answer.
            "probe_label": cipher[s[-1]],
            "text": s,
        })
    train, test = items[: cfg.n_train_sentences], items[cfg.n_train_sentences:]
    meta = {"task": "cipher", "cipher": cipher}
    return train, test, meta


# ---------------------------------------------------------------- Task B: synthetic facts

PROFESSIONS = ["architect", "baker", "chemist", "dentist", "engineer", "florist", "geologist",
               "historian", "jeweler", "lawyer", "musician", "nurse", "optician", "pilot",
               "sculptor", "tailor", "veterinarian", "welder"]
CITIES = ["Lisbon", "Nairobi", "Osaka", "Quito", "Tromso", "Valletta", "Wellington", "Zagreb",
          "Bergen", "Cusco", "Dakar", "Esbjerg", "Funchal", "Galway", "Hobart", "Ipoh"]
SYLLABLES = ["ka", "ri", "lo", "ven", "mi", "tor", "sa", "nel", "du", "fay", "ori", "bek", "tam",
             "zu", "lin", "pra", "ox", "wen", "hal", "ym"]

PROFESSION_TEMPLATES = [
    "What is {name}'s profession?",
    "What does {name} do for a living?",
    "Which occupation does {name} have?",
    "State the job of {name}.",
    "{name} works as a what?",
    "Tell me the profession of the person named {name}.",
]
CITY_TEMPLATES = [
    "Which city does {name} live in?",
    "Where does {name} live?",
    "In what city is {name} based?",
    "Name the city where {name} resides.",
    "{name} lives in which city?",
    "Tell me the home city of {name}.",
]


def make_name(rng: random.Random) -> str:
    first = "".join(rng.choice(SYLLABLES) for _ in range(2)).capitalize()
    last = "".join(rng.choice(SYLLABLES) for _ in range(2)).capitalize()
    return f"{first} {last}"


def facts_examples(cfg: DataConfig) -> tuple[list[dict], list[dict], dict]:
    rng = random.Random(cfg.seed + 1)
    people = {}
    while len(people) < cfg.n_people:
        name = make_name(rng)
        if name not in people:
            people[name] = {"profession": rng.choice(PROFESSIONS), "city": rng.choice(CITIES)}
    n_tr = cfg.n_train_templates
    train, test = [], []
    for name, attrs in people.items():
        for templates, key in ((PROFESSION_TEMPLATES, "profession"), (CITY_TEMPLATES, "city")):
            for i, t in enumerate(templates[: n_tr + cfg.n_test_templates]):
                ex = {"prompt": t.format(name=name), "completion": attrs[key],
                      "probe_label": attrs[key], "name": name, "attribute": key}
                (train if i < n_tr else test).append(ex)
    rng.shuffle(train)
    rng.shuffle(test)
    meta = {"task": "facts", "people": people}
    return train, test, meta


TASKS = {"cipher": cipher_examples, "facts": facts_examples}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
