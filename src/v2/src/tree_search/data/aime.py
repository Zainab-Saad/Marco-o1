from __future__ import annotations

import re

from datasets import load_dataset

from tree_search.config import DataConfig

from .common import (
    _ANSWER_RE,
    _BOXED_RE,
    _NUMBER_RE
)

_DATASET_IDS: dict[int, str] = {
    2024: "HuggingFaceH4/aime_2024",
    2025: "math-ai/aime25", # TODO: check it
}

def load_aime(year: int, config: DataConfig) -> list[dict]:

    if year not in _DATASET_IDS:
        raise ValueError(f"Unsupported AIME year {year}. Available: {sorted(_DATASET_IDS)}")

    dataset_id = _DATASET_IDS[year]
    try:
        ds = load_dataset(dataset_id, split=config.split)
    except Exception:
        # AIME datasets sometimes only have a "train" split
        ds = load_dataset(dataset_id, split="train")

    if config.num_samples is not None:
        ds = ds.shuffle(seed=config.seed).select(range(config.num_samples))

    records = []
    for ex in ds:
        # aime24 has solution field seperately which has the logic for solving problem (aime25 doesnt have this)
        # but both aime24 and aime25 have answer field which is of our interest for now coz we only use ground truth to get answer.
        records.append({
            "question": ex.get("problem", ex.get("Problem", "")),
            "solution": str(ex.get("answer", ex.get("Answer", ""))),
            "gold":       extract_gold_answer(str(ex.get("answer", ""))),
            "year": year,
            "problem_id": str(ex.get("id", ex.get("problem_id", ""))),
        })
    return records


def load_aime24(config: DataConfig) -> list[dict]:
    return load_aime(2024, config)


def load_aime25(config: DataConfig) -> list[dict]:
    return load_aime(2025, config)


# just a bit too complicated method to get answer in whatever format llm returns answer because sometimes llm doesnt give answer in our said format
def extract_answer(text: str) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()

    m = _ANSWER_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()

    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()

    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1].replace(",", "")

    print(f"[WARNING][extract_answer] No answer found in: {text[:100]!r}")
    return None

def extract_gold_answer(raw: str) -> str:
    return str(raw).strip()

def _to_int(s: str | None) -> int | None:
    if s is None:
        return None
    try:
        return int(float(s.replace(",", "")))
    except (ValueError, OverflowError):
        return None


def answers_match(pred: str | None, gold: str | None) -> bool:
    p, g = _to_int(pred), _to_int(gold)
    if p is None or g is None:
        return False
    return p == g
