#!/usr/bin/env python
"""
I/O
- Input rows use `label` (ground truth) and `model_output` (raw generation).
- Writes two files next to the input generations file:
   prediction.jsonl, suspect_false_negative.jsonl.
- Prediction rows contain:
   run_id, question_idx, label, extracted_answer, is_correct, reason,
   output_token_length, finish_reason, last_box_source, think_type.
- Suspect false-negative rows contain no-match cases where label or extracted
  answer is not a simple integer/single-letter answer:
   question_idx, label, extracted_answer.

Extraction policy:
- If the prediction contains a valid last `\\boxed{...}`, use its content as
  extracted_answer.
- If no valid `\\boxed{...}` exists, extracted_answer is the full solution text and
  reason must be `can_not_extract`.

last_box_source:
- solution: last valid box is after `</think>`, or there is no think tag.
- thought: last valid box is before `</think>`.
- thought_no_close: last valid box is after `<think>` with no closing `</think>`.
- none: no valid box can be extracted.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable

import matcher as matcher


LAST_BOX_SOURCE_SOLUTION = "solution"
LAST_BOX_SOURCE_THOUGHT = "thought"
LAST_BOX_SOURCE_THOUGHT_NO_CLOSE = "thought_no_close"
LAST_BOX_SOURCE_NONE = "none"

THINK_TYPE_NO_THOUGHT = "no_thought"
THINK_TYPE_THOUGHT_ANSWER_ONLY = "thought_answer_only"
THINK_TYPE_NORMAL_THOUGHT = "normal_thought"
THINK_TYPE_UNCLOSED_THOUGHT = "unclosed_thought"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

REASON_NO_MATCH = "no_match"
PREDICTION_FILENAME = "prediction.jsonl"
SUSPECT_FALSE_NEGATIVE_FILENAME = "suspect_false_negative.jsonl"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc


def _thought_text_before_close(text: str) -> str:
    close_idx = text.find(THINK_CLOSE)
    before_close = text[:close_idx]
    open_idx = before_close.rfind(THINK_OPEN)
    if open_idx != -1:
        return before_close[open_idx + len(THINK_OPEN) :]
    return before_close


def classify_think_type(text: Any) -> str:
    s = matcher.to_text(text)
    has_open = THINK_OPEN in s
    has_close = THINK_CLOSE in s

    if has_open and not has_close:
        return THINK_TYPE_UNCLOSED_THOUGHT

    if not has_close:
        return THINK_TYPE_NO_THOUGHT

    thought_text = _thought_text_before_close(s)
    if not thought_text.strip():
        return THINK_TYPE_NO_THOUGHT

    without_boxes = matcher.remove_valid_boxed_expressions(thought_text)
    if re.sub(r"[ \t\r\n]+", "", without_boxes) == "":
        return THINK_TYPE_THOUGHT_ANSWER_ONLY

    return THINK_TYPE_NORMAL_THOUGHT


def classify_last_box_source(text: Any) -> str:
    s = matcher.to_text(text)
    answer = matcher.find_last_boxed_answer(s)

    if not answer.found or answer.start is None:
        return LAST_BOX_SOURCE_NONE

    close_idx = s.find(THINK_CLOSE)

    # Case 1: Có </think>
    if close_idx != -1:
        if answer.start >= close_idx + len(THINK_CLOSE):
            return LAST_BOX_SOURCE_SOLUTION
        return LAST_BOX_SOURCE_THOUGHT

    # Case 2: Không có </think>, nhưng có <think>
    # Nếu box nằm sau <think>, coi là thought chưa đóng.
    open_idx = s.rfind(THINK_OPEN)
    if open_idx != -1 and answer.start >= open_idx + len(THINK_OPEN):
        return LAST_BOX_SOURCE_THOUGHT

    # Case 3: Không có thought marker hoặc box nằm ngoài thought.
    # Theo quy ước của bạn: coi là solution.
    return LAST_BOX_SOURCE_SOLUTION


def is_integer_answer(value: Any) -> bool:
    if value is None:
        return False

    return bool(re.fullmatch(r"[+-]?\d+", str(value).strip()))


def is_single_letter_answer(value: Any) -> bool:
    if value is None:
        return False

    return bool(re.fullmatch(r"[A-Za-z]", str(value).strip()))


def is_simple_answer(value: Any) -> bool:
    return is_integer_answer(value) or is_single_letter_answer(value)


def should_keep_suspect_false_negative(label: Any, extracted_answer: Any) -> bool:
    return (not is_simple_answer(label)) or (not is_simple_answer(extracted_answer))


def _build_prediction_row(
    *,
    row_run_id: str,
    question_idx: Any,
    label: Any,
    result: Any,
    output_token_length: Any,
    finish_reason: Any,
    last_box_source: str,
    think_type: str,
) -> Dict[str, Any]:
    return {
        "run_id": row_run_id,
        "question_idx": question_idx,
        "label": matcher.to_text(label),
        "extracted_answer": result.extracted_answer,
        "is_correct": 1 if result.matched else 0,
        "reason": result.reason,
        "output_token_length": output_token_length,
        "finish_reason": finish_reason,
        "last_box_source": last_box_source,
        "think_type": think_type,
    }


def evaluate_file(
    input_path: Path,
    label_field: str = "label",
    pred_field: str = "model_output",
    question_field: str = "question",
    run_id_field: str = "run_id",
    qid_field: str = "question_idx",
) -> None:
    output_dir = input_path.parent
    prediction_path = output_dir / PREDICTION_FILENAME
    suspect_false_negative_path = output_dir / SUSPECT_FALSE_NEGATIVE_FILENAME

    total = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    with (
        prediction_path.open("w", encoding="utf-8") as prediction_file,
        suspect_false_negative_path.open("w", encoding="utf-8") as suspect_false_negative_file,
    ):
        for row_index, row in enumerate(_iter_jsonl(input_path)):
            total += 1

            gt = row.get(label_field, "")
            pred_text = row.get(pred_field, "")
            question_text = row.get(question_field, "")
            row_run_id = matcher.to_text(row.get(run_id_field, ""))
            question_idx = row.get(qid_field, row_index)

            result = matcher.match_answer(gt, pred_text, question_text)

            think_type = classify_think_type(pred_text)
            last_box_source = classify_last_box_source(pred_text)
            output_token_length = row.get("output_token_length")
            finish_reason = row.get("finish_reason")

            prediction_row = _build_prediction_row(
                row_run_id=row_run_id,
                question_idx=question_idx,
                label=gt,
                result=result,
                output_token_length=output_token_length,
                finish_reason=finish_reason,
                last_box_source=last_box_source,
                think_type=think_type,
            )
            prediction_file.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")

            if result.reason == REASON_NO_MATCH and should_keep_suspect_false_negative(
                gt,
                result.extracted_answer,
            ):
                negative_row = {
                    "question_idx": question_idx,
                    "label": matcher.to_text(gt),
                    "extracted_answer": result.extracted_answer,
                }
                suspect_false_negative_file.write(
                    json.dumps(negative_row, ensure_ascii=False) + "\n"
                )

    print(f"Input: {input_path}")
    print(f"Prediction: {prediction_path}")
    print(f"Suspect false negative: {suspect_false_negative_path}")
    print(f"Total: {total}")
    print(f"SymPy available: {matcher.SYMPY_AVAILABLE}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Math checker v2.")
    parser.add_argument("--input", required=True, help="Path to input generations.jsonl")
    parser.add_argument("--label-field", default="label", help="Ground-truth field name")
    parser.add_argument("--pred-field", default="model_output", help="Prediction field name")
    parser.add_argument("--question-field", default="question", help="Question text field name")
    parser.add_argument("--run-id-field", default="run_id", help="Run-id field name")
    parser.add_argument("--qid-field", default="question_idx", help="Question-id field name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_file(
        input_path=Path(args.input),
        label_field=args.label_field,
        pred_field=args.pred_field,
        question_field=args.question_field,
        run_id_field=args.run_id_field,
        qid_field=args.qid_field,
    )


if __name__ == "__main__":
    main()
