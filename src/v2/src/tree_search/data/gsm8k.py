from __future__ import annotations

import re

from datasets import load_dataset

from tree_search.config import DataConfig
from .common import (
    _ANSWER_RE,
    _BOXED_RE,
    _NUMBER_RE,
)

def load_gsm8k(config: DataConfig) -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=config.split)
    if config.num_samples is not None:
        ds = ds.shuffle(seed=config.seed).select(range(config.num_samples))
    return [
        {
        "question": ex["question"], 
        "solution": ex["answer"],
        "gold": extract_gold_answer(ex["answer"]),
        } 
        for ex in ds
    ]

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
    print(f"[WARNING][extract_answer] No answer found in last 100 chars: {text[:100]!r}")
    return None

# gsm8k answers are with solution text after a ####
def extract_gold_answer(raw_solution: str) -> str | None:
    m = _ANSWER_RE.search(raw_solution)
    if m:
        return m.group(1).replace(",", "").strip()
    return None

def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None

def answers_match(pred: str | None, gold: str | None) -> bool:
    p, g = _to_float(pred), _to_float(gold)
    if p is None or g is None:
        return False
    return abs(p - g) < 1e-4 # dont use a simple == coz gsm8k can have float answers and its nice to give llm room for some decimal point error