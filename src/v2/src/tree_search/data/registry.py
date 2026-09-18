from __future__ import annotations

from .gsm8k import (
    load_gsm8k, 
    extract_answer as gsm8k_extract_answer,
    answers_match as gsm8k_answers_match,
)
from .math import (
    load_math, 
    extract_answer as math_extract_answer, 
    answers_match as math_answers_match
)
from .aime import (
    load_aime24, load_aime25, 
    extract_answer as aime_extract_answer,
    answers_match as aime_answers_match
) 
from tree_search.prompts import GSM8K_COT_PROMPT, MATH_COT_PROMPT, AIME_COT_PROMPT

DATASET_REGISTRY = {
    "gsm8k": {
        "loader": load_gsm8k,
        "extract_answer": gsm8k_extract_answer,
        "answers_match": gsm8k_answers_match,
        "get_gold": lambda p: p["gold"],
        "prompt": GSM8K_COT_PROMPT,
    },
    "math": {
        "loader": lambda cfg: load_math(cfg, levels=getattr(cfg, "levels", [5])),
        "extract_answer": math_extract_answer,
        "answers_match": math_answers_match,
        "get_gold": lambda p: p["gold"],
        "prompt": MATH_COT_PROMPT,
    },
    "aime24": {
        "loader": load_aime24,
        "extract_answer": aime_extract_answer,
        "answers_match": aime_answers_match,
        "get_gold": lambda p: p["gold"],
        "prompt": AIME_COT_PROMPT,
    },
    "aime25": {
        "loader": load_aime25,
        "extract_answer": aime_extract_answer,
        "answers_match": aime_answers_match,
        "get_gold": lambda p: p["gold"],
        "prompt": AIME_COT_PROMPT,
    },
}


def get_dataset(name: str) -> dict:

    if name not in DATASET_REGISTRY:

        raise ValueError(f"Unknown dataset '{name}'. Valid: {list(DATASET_REGISTRY)}")

    return DATASET_REGISTRY[name]

def format_prompt(prompt: str, question: str) -> str:
    return prompt.format(question=question)

def level_tag(dataset: str, levels) -> str:
    if dataset != "math":
        return ""
    vals = [str(l) for l in (levels or ["all"])]
    if any(v.strip().lower() == "all" for v in vals):
        return "_Lall"
    return "_L" + "".join(sorted(vals))