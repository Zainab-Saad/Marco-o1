from __future__ import annotations

import re

_ANSWER_RE = re.compile(r"####\s*([\-\d,\.]+)")
_BOXED_RE  = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")

# take the extraction logic from the codebase of paper: https://arxiv.org/abs/2103.03874
# link to the github repo file from where code is taken and refactored: https://github.com/hendrycks/math
# this above repo has MIT liscence TODO: we mighjt also need MIT licence

def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, right_brace_idx, num_left_braces_open = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx:right_brace_idx + 1]

def remove_boxed(s: str | None) -> str | None:
    if s is None:
        return None
    for left in ("\\boxed{", "\\fbox{"):
        if s.startswith(left) and s.endswith("}"):
            return s[len(left):-1]
    if s.startswith("\\boxed "):
        return s[len("\\boxed "):]
    return s