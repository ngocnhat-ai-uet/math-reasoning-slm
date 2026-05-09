from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    from sympy import N, simplify
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    SYMPY_AVAILABLE = True
except Exception:
    # SymPy is optional at runtime. The checker still runs without symbolic matching.
    SYMPY_AVAILABLE = False
    N = None
    simplify = None
    parse_latex = None
    parse_expr = None


REASON_LITERAL_MATCH = "literal_match"
REASON_NUMERIC_MATCH = "numeric_match"
REASON_SYMBOLIC_MATCH = "symbolic_match"
REASON_NO_MATCH = "no_match"
REASON_CAN_NOT_EXTRACT = "can_not_extract"

CHOICE_SET = {"A", "B", "C", "D", "E"}


@dataclass(frozen=True)
class BoxedAnswer:
    content: str
    found: bool
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass(frozen=True)
class MatchResult:
    matched: bool
    reason: str
    extracted_answer: str


def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _strip_math_wrappers(text: str) -> str:
    """Remove common wrapper delimiters without changing core content."""
    s = text.strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    if len(s) >= 4 and s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    if len(s) >= 4 and s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    return s


def _normalize_for_literal_compare(text: str) -> str:
    """Normalize formatting noise for robust literal comparison."""
    s = _strip_math_wrappers(to_text(text))
    s = re.sub(r"\s+", "", s)
    return s.casefold()


def _choice_match(gt: str, pred: str) -> bool:
    """Handle multiple-choice style answers like A/B/C/D/E."""
    gt_clean = _normalize_for_literal_compare(gt).upper()
    if gt_clean not in CHOICE_SET:
        return False

    pred_candidates = re.findall(r"\b([A-E])\b", to_text(pred).upper())
    return bool(pred_candidates) and pred_candidates[-1] == gt_clean


def _literal_match(gt: str, pred: str) -> bool:
    gt_raw = to_text(gt).strip()
    pred_raw = to_text(pred).strip()
    if gt_raw == pred_raw:
        return True

    if _normalize_for_literal_compare(gt_raw) == _normalize_for_literal_compare(pred_raw):
        return True

    return _choice_match(gt_raw, pred_raw)


def _parse_numeric_candidate(text: str) -> Optional[float]:
    """
    Parse a single numeric candidate from text.
    Supports int/float, percent, a/b fractions, and simple LaTeX fractions.
    """
    s = _strip_math_wrappers(to_text(text)).strip()
    if not s:
        return None

    s = s.replace(",", "")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    # Handle simple assignment forms like "x = 3".
    if s.count("=") == 1:
        left, right = s.split("=")
        if len(left.strip()) <= 2:
            s = right.strip()

    if s.endswith("%"):
        base = _parse_numeric_candidate(s[:-1])
        if base is None:
            return None
        return base / 100.0

    frac_latex = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", s)
    if frac_latex:
        numerator = _parse_numeric_candidate(frac_latex.group(1))
        denominator = _parse_numeric_candidate(frac_latex.group(2))
        if numerator is None or denominator is None or denominator == 0:
            return None
        return numerator / denominator

    frac_plain = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", s)
    if frac_plain:
        denominator = float(frac_plain.group(2))
        if denominator == 0:
            return None
        return float(frac_plain.group(1)) / denominator

    try:
        return float(s)
    except Exception:
        return None


def _numeric_match(gt: str, pred: str) -> bool:
    """
    Compare numeric values with percentage-aware tolerance.

    If gt is numeric g and pred is numeric p, treat p as correct if:
    - p ~= g
    - p ~= g/100
    - p ~= g*100
    """
    g = _parse_numeric_candidate(gt)
    p = _parse_numeric_candidate(pred)
    if g is None or p is None:
        return False

    candidates = (g, g / 100.0, g * 100.0)
    for c in candidates:
        if math.isclose(p, c, rel_tol=1e-9, abs_tol=1e-9):
            return True
    return False


def _parse_symbolic_expr(text: str):
    if not SYMPY_AVAILABLE:
        return None

    s = _strip_math_wrappers(to_text(text)).strip()
    if not s:
        return None

    # Try LaTeX parser first, then sympy expression parser.
    for parser in (parse_latex, parse_expr):
        try:
            if parser is parse_expr:
                return parser(s.replace("^", "**"))
            return parser(s)
        except Exception:
            continue
    return None


def _symbolic_match(gt: str, pred: str) -> bool:
    if not SYMPY_AVAILABLE:
        return False

    gt_text = to_text(gt).strip()
    pred_text = to_text(pred).strip()

    # Equation-to-equation comparison: compare normalized residual forms.
    if gt_text.count("=") == 1 and pred_text.count("=") == 1:
        gt_l, gt_r = gt_text.split("=")
        pr_l, pr_r = pred_text.split("=")
        gt_text = f"({gt_l})-({gt_r})"
        pred_text = f"({pr_l})-({pr_r})"

    gt_expr = _parse_symbolic_expr(gt_text)
    pred_expr = _parse_symbolic_expr(pred_text)
    if gt_expr is None or pred_expr is None:
        return False

    try:
        if gt_expr == pred_expr or str(gt_expr) == str(pred_expr):
            return True
    except Exception:
        pass

    try:
        if simplify(gt_expr - pred_expr) == 0:
            return True
    except Exception:
        pass

    try:
        if gt_expr.equals(pred_expr):
            return True
    except Exception:
        pass

    try:
        gt_num = float(N(gt_expr))
        pred_num = float(N(pred_expr))
        if math.isclose(gt_num, pred_num, rel_tol=1e-9, abs_tol=1e-9):
            return True
    except Exception:
        pass

    return False


def _valid_boxed_answers(text: Any) -> list[BoxedAnswer]:
    full_text = to_text(text)
    marker = r"\boxed{"
    marker_len = len(marker)
    answers: list[BoxedAnswer] = []

    for match in re.finditer(re.escape(marker), full_text):
        content_start = match.start() + marker_len
        depth = 1
        idx = content_start
        while idx < len(full_text):
            char = full_text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    answers.append(
                        BoxedAnswer(
                            content=full_text[content_start:idx].strip(),
                            found=True,
                            start=match.start(),
                            end=idx + 1,
                        )
                    )
                    break
            idx += 1

    return answers


def find_last_boxed_answer(text: Any) -> BoxedAnswer:
    """
    Find the last valid \\boxed{...} and keep its source span.

    If none is valid, content is the full text and found is False.
    """
    full_text = to_text(text)
    answers = _valid_boxed_answers(full_text)
    if not answers:
        return BoxedAnswer(content=full_text.strip(), found=False)
    return answers[-1]


def extract_boxed_answer(text: Any) -> Tuple[str, bool]:
    """
    Extract content from the last valid \\boxed{...}.

    Returns:
    - (extracted_content, True) when a valid boxed answer is found.
    - (full_text, False) otherwise.
    """
    answer = find_last_boxed_answer(text)
    return answer.content, answer.found


def remove_valid_boxed_expressions(text: Any) -> str:
    """Remove all valid \\boxed{...} spans from text."""
    full_text = to_text(text)
    answers = _valid_boxed_answers(full_text)
    if not answers:
        return full_text

    pieces: list[str] = []
    last_end = 0
    for answer in answers:
        if answer.start is None or answer.end is None:
            continue
        pieces.append(full_text[last_end:answer.start])
        last_end = answer.end
    pieces.append(full_text[last_end:])
    return "".join(pieces)


def match_answer(gt: str | int | float, pred_text: str | int | float) -> MatchResult:
    """
    Match a model prediction against ground truth.

    Rules:
    - If no valid boxed answer is found, the sample is incorrect with can_not_extract.
    - If boxed answer exists, reason reflects the engine that matched.
    """
    gt_text = to_text(gt)
    answer = find_last_boxed_answer(pred_text)
    extracted_answer = answer.content

    if not answer.found:
        return MatchResult(False, REASON_CAN_NOT_EXTRACT, extracted_answer)

    try:
        if _literal_match(gt_text, extracted_answer):
            return MatchResult(True, REASON_LITERAL_MATCH, extracted_answer)

        if _numeric_match(gt_text, extracted_answer):
            return MatchResult(True, REASON_NUMERIC_MATCH, extracted_answer)

        if _symbolic_match(gt_text, extracted_answer):
            return MatchResult(True, REASON_SYMBOLIC_MATCH, extracted_answer)
    except Exception:
        # Keep output contract stable even when one row is malformed.
        return MatchResult(False, REASON_NO_MATCH, extracted_answer)

    return MatchResult(False, REASON_NO_MATCH, extracted_answer)
