from __future__ import annotations

import re
from datasets import concatenate_datasets, load_dataset
from math_verify import parse as _mv_parse, verify as _mv_verify

from tree_search.config import DataConfig

from .common import (
    _BOXED_RE,
    _NUMBER_RE, 
    last_boxed_only_string,
    remove_boxed,
)

# take some of the extraction logic from the codebase of paper: https://arxiv.org/abs/2103.03874
# link to the github repo file from where code is taken and refactored: https://github.com/hendrycks/math
# this above repo has MIT liscence TODO: we mighjt also need MIT licence


SUBJECT_TO_CONFIG = {
    "Algebra": "algebra",
    "Counting & Probability": "counting_and_probability",
    "Geometry": "geometry",
    "Intermediate Algebra": "intermediate_algebra",
    "Number Theory": "number_theory",
    "Prealgebra": "prealgebra",
    "Precalculus": "precalculus",
}
SUBJECTS = tuple(SUBJECT_TO_CONFIG)

def _parse_levels(levels) -> set[str] | None:
    if levels is None:
        return None
    raw = [levels] if isinstance(levels, (int, str)) else list(levels)
    if any(str(l).strip().lower() == "all" for l in raw):
        return None
    out = set()
    for l in raw:
        s = str(l).strip()
        out.add(s if s.lower().startswith("level") else f"Level {s}")
    return out

def load_math(config: DataConfig, 
              subject: str | None = None,
              levels: int | str | list[int | str] | None = None,) -> list[dict]:

    if subject is None:
        names = list(SUBJECT_TO_CONFIG.values())

    else:
        wanted = [subject] if isinstance(subject, str) else list(subject)
        bad = [s for s in wanted if s not in SUBJECT_TO_CONFIG]
        if bad:
            raise ValueError(f"unknown subject(s) {bad}; valid: {SUBJECTS}")
        names = [SUBJECT_TO_CONFIG[s] for s in wanted]


    parts = [load_dataset("EleutherAI/hendrycks_math", n, split=config.split) for n in names]
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)


    if levels is not None:
        raw = [levels] if isinstance(levels, (int, str)) else list(levels)
        # wanted_levels = {f"Level {l}" if isinstance(l, int) else l for l in raw}
        # ds = ds.filter(lambda ex: ex["level"] in wanted_levels)
        wanted_levels = _parse_levels(levels)
        if wanted_levels is not None:
            ds = ds.filter(lambda ex: ex["level"] in wanted_levels)


    if config.num_samples is not None:
        n = min(config.num_samples, len(ds))
        ds = ds.shuffle(seed=config.seed).select(range(n))

    return [
        {
            "question": ex["problem"],
            "solution": ex["solution"],                  
            "gold": extract_gold_answer(ex["solution"]),  
            "level": ex["level"],                         
            "subject": ex["type"],
        }
        for ex in ds
        ]

# i added these 9 functions to account for the false positive and false negative in my gold answer and llm answer extraction hopefully it fixes it
def _fix_fracs(string):
    substrs = string.split("\\frac"); new_str = substrs[0]
    for substr in substrs[1:]:
        new_str += "\\frac"
        if substr and substr[0] == "{":
            new_str += substr
        else:
            if len(substr) < 2:
                return string
            a, b = substr[0], substr[1]
            new_str += ("{"+a+"}{"+b+"}"+substr[2:]) if b != "{" else ("{"+a+"}"+b+substr[2:])
    return new_str

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a, b = string.split("/")
    try:
        ia, ib = int(a), int(b)
        assert string == "{}/{}".format(ia, ib)
        return "\\frac{" + str(ia) + "}{" + str(ib) + "}"
    except Exception:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt"); new_string = splits[0]
    for s in splits[1:]:
        new_string += ("\\sqrt{" + s[0] + "}" + s[1:]) if (s and s[0] != "{") else ("\\sqrt" + s)
    return new_string

_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_PMATRIX = re.compile(r"\\begin\{(?:pmatrix|bmatrix|vmatrix)\}(.*?)"
                    r"\\end\{(?:pmatrix|bmatrix|vmatrix)\}", re.S)
_LHS = re.compile(r"^\s*[a-zA-Z]\s*(?:\([a-zA-Z]\))?\s*=\s*")
_JOINWORD = re.compile(r"\\text\s*\{\s*(?:and|or)\s*\}")

def _pmat(m):
    return "(" + ",".join(x.strip() for x in re.split(r"\\\\+", m.group(1)) if x.strip()) + ")"

def _presplit_norm(s):
    return _THOUSANDS.sub("", _PMATRIX.sub(_pmat, s).replace("\\!", "").replace("{,}", ""))

def _strip_string(string):
    string = _presplit_norm(string).replace("\n", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace("$", "")
    if "\\text{ " in string:
        string = string.split("\\text{ ")[0].rstrip()
    string = string.replace("\\%", "").replace("%", "")
    string = string.replace(" .", " 0.").replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = _LHS.sub("", string)
    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    return _fix_a_slash_b(string)

def is_equiv(str1, str2):
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2

def _split_top_level(s):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == "," and depth <= 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    out.append(cur)
    return [x.strip() for x in out if x.strip()]

def _symbolic_eq(a, b):
    try:
        pa = _mv_parse("\\boxed{" + a + "}", parsing_timeout=5)
        pb = _mv_parse("\\boxed{" + b + "}", parsing_timeout=5)
        if not pa or not pb:
            return False
        return bool(_mv_verify(pa, pb, timeout_seconds=5))
    except Exception:
        return False

# also after adding these 9 functions also changing the remaining fucntions, prev code is commented out only


# math dataset sometimes has answers in form of latex so need to extract that
def _normalize_latex(s: str | None) -> str:
    if s is None:
        return ""
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = s.replace(r"\left", "").replace(r"\right", "")
    s = s.replace(r"\!", "").replace(r"\,", "").replace(r"\;", "")
    s = s.replace(r"\%", "").replace(r"\$", "").replace("$", "")
    s = re.sub(r"\\text\s*\{[^}]*\}", "", s)
    return s.strip()
 
# like gsm8k, math dataset also has the final answer in solution field after the steps to solve the question (no seperate answer field)
def extract_gold_answer(solution: str) -> str | None:
    # m = _BOXED_RE.findall(solution)
    # return m[-1].strip() if m else None
    return remove_boxed(last_boxed_only_string(solution))
 
 
def extract_answer(text: str, number_fallback: bool = False) -> str | None:
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    # m = _BOXED_RE.findall(text)
    a = remove_boxed(last_boxed_only_string(text))
    # if m:
    #     return m[-1].strip()
    # if number_fallback:
    #     nums = _NUMBER_RE.findall(text)
    #     if nums:
    #         return nums[-1].replace(",", "")
    if a is not None:
        return a.strip()
    if number_fallback:
        nums = _NUMBER_RE.findall(text)
        if nums:
            return nums[-1].replace(",", "")
    return None

def _to_float(s: str | None) -> float | None:
    if s is None:
        return None
    try:
        return float(s.replace(",", "").strip())
    except ValueError:
        return None
 
 
# def answers_match(pred: str | None, gold: str | None) -> bool:

#     if pred is None or gold is None:
#         return False
    
#     if _normalize_latex(pred) == _normalize_latex(gold):
#         return True
    
#     try:
#         return bool(_mv_verify(_mv_parse(_normalize_latex(gold)),
#                             _mv_parse(_normalize_latex(pred))))
#     except Exception:
#         pass
#     p, g = _to_float(pred), _to_float(gold)
#     if p is not None and g is not None:
#         return abs(p - g) < 1e-4
#     return False

# TODO: at end might wanna check if i need to set strict_order=True coz sometimes order must matter sometimes not...
def answers_match(pred: str | None, gold: str | None, strict_order: bool = False) -> bool:
    if pred is None or gold is None:
        return False
    if pred.strip() == gold.strip():
        return True
    pred = _presplit_norm(_JOINWORD.sub(",", pred))
    gold = _presplit_norm(_JOINWORD.sub(",", gold))
    if is_equiv(pred, gold):
        return True
    gs, ps = _split_top_level(gold), _split_top_level(pred)
    if len(gs) > 1 or len(ps) > 1:
        if len(gs) != len(ps):
            return False
        if strict_order:
            # verify(gold, answer) gold,pred is right dont swap order!!!!
            return all(is_equiv(p, g) or _symbolic_eq(g, p) for p, g in zip(ps, gs))
        used = set()
        for g in gs:
            m = next((j for j, p in enumerate(ps)
                    if j not in used and (is_equiv(p, g) or _symbolic_eq(g, p))), None)
            if m is None:
                return False
            used.add(m)
        return True
    return _symbolic_eq(gold, pred)