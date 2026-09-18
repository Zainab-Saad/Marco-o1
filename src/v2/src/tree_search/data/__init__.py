from .common import (
    _ANSWER_RE,
    _BOXED_RE,
    _NUMBER_RE,
)
from .registry import (
    format_prompt,
    DATASET_REGISTRY,
    get_dataset,
    level_tag,
)

from .gsm8k import (
    answers_match as gsm8k_answers_match,
    extract_answer as gsm8k_extract_answer,
    load_gsm8k,
)
from .math import (
    SUBJECTS as MATH_SUBJECTS,
    answers_match as math_answers_match,
    extract_answer as math_extract_answer,
    extract_gold_answer as math_extract_gold_answer,
    load_math,
)
from .aime import (
    answers_match as aime_answers_match,
    extract_answer as aime_extract_answer,
    load_aime,
    load_aime24,
    load_aime25,
)

__all__ = [
    "_ANSWER_RE",
    "_BOXED_RE",
    "_NUMBER_RE",
    "format_prompt",
    "DATASET_REGISTRY",
    "get_dataset",
    "level_tag",
    # GSM8K
    "load_gsm8k",
    "gsm8k_extract_answer",
    "gsm8k_answers_match",
    # MATH
    "load_math",
    "MATH_SUBJECTS",
    "math_extract_answer",
    "math_extract_gold_answer",
    "math_answers_match",
    # AIME
    "load_aime",
    "load_aime24",
    "load_aime25",
    "aime_extract_answer",
    "aime_answers_match",
]