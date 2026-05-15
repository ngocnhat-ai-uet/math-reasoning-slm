#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Set


LLM_JUDGE_REASON = "llm_judge"
PREDICTION_FILENAME = "prediction.jsonl"
SUSPECT_FALSE_NEGATIVE_FILENAME = "suspect_false_negative.jsonl"


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc

            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")

            yield row


def _load_false_negative_ids(path: Path) -> Set[Any]:
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Expected a .txt false-negative index file, got: {path}")

    ids = set()
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            index = line.strip()
            if not index:
                continue
            if " " in index or "\t" in index:
                raise ValueError(f"Expected one index per line at {path}:{line_number}")
            ids.add(index)
    return ids


def _replace_file(tmp_path: Path, target_path: Path) -> None:
    try:
        os.replace(tmp_path, target_path)
    except PermissionError:
        shutil.copyfile(tmp_path, target_path)
        try:
            tmp_path.unlink()
        except PermissionError:
            print(f"Warning: could not remove temp file: {tmp_path}")


def _remove_ids_from_jsonl(path: Path, question_ids: Set[Any]) -> tuple[int, int]:
    total = 0
    removed = 0
    tmp_path = path.with_name(f"{path.name}.tmp")

    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in _iter_jsonl(path):
                total += 1
                if row.get("index") in question_ids:
                    removed += 1
                    continue
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        _replace_file(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except PermissionError:
                pass
        raise

    return total, removed


def apply_llm_judgement(
    prediction_path: Path,
    false_negative_path: Path,
    suspect_false_negative_path: Path | None = None,
) -> None:
    false_negative_ids = _load_false_negative_ids(false_negative_path)
    seen_ids = set()
    total = 0
    updated = 0
    suspect_total = 0
    suspect_removed = 0

    if suspect_false_negative_path is None:
        suspect_false_negative_path = prediction_path.with_name(
            SUSPECT_FALSE_NEGATIVE_FILENAME
        )

    tmp_path = prediction_path.with_name(f"{prediction_path.name}.tmp")

    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for row in _iter_jsonl(prediction_path):
                total += 1
                index = row.get("index")

                if index in false_negative_ids:
                    seen_ids.add(index)
                    row["is_correct"] = 1
                    row["reason"] = LLM_JUDGE_REASON
                    updated += 1

                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

        _replace_file(tmp_path, prediction_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except PermissionError:
                pass
        raise

    missing_ids = false_negative_ids - seen_ids
    if seen_ids and suspect_false_negative_path.is_file():
        suspect_total, suspect_removed = _remove_ids_from_jsonl(
            suspect_false_negative_path, seen_ids
        )
    elif seen_ids:
        print(
            "Warning: suspect false-negative file not found, "
            f"skip removal: {suspect_false_negative_path}"
        )

    print(f"Prediction: {prediction_path}")
    print(f"False negative: {false_negative_path}")
    print(f"Suspect false negative: {suspect_false_negative_path}")
    print(f"Prediction rows: {total}")
    print(f"False-negative ids: {len(false_negative_ids)}")
    print(f"Updated rows: {updated}")
    print(f"Suspect false-negative rows: {suspect_total}")
    print(f"Removed suspect false-negative rows: {suspect_removed}")
    if missing_ids:
        print(f"Warning: {len(missing_ids)} false-negative ids not found in prediction")
        for index in sorted(missing_ids, key=str):
            print(f"  - {index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply LLM false-negative judgements to prediction.jsonl."
    )
    parser.add_argument(
        "--prediction",
        help=(
            "Path to prediction.jsonl. Defaults to prediction.jsonl next to "
            "--false-negative."
        ),
    )
    parser.add_argument(
        "--false-negative",
        required=True,
        help="Path to one-index-per-line _false_negative_index.txt file",
    )
    parser.add_argument(
        "--suspect-false-negative",
        help=(
            "Path to suspect_false_negative.jsonl. Defaults to the file with this "
            "name next to prediction.jsonl."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    false_negative_path = Path(args.false_negative)
    prediction_path = (
        Path(args.prediction)
        if args.prediction
        else false_negative_path.with_name(PREDICTION_FILENAME)
    )
    apply_llm_judgement(
        prediction_path=prediction_path,
        false_negative_path=false_negative_path,
        suspect_false_negative_path=(
            Path(args.suspect_false_negative) if args.suspect_false_negative else None
        ),
    )


if __name__ == "__main__":
    main()
