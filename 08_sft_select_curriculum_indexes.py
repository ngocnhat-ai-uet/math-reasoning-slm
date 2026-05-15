#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_OUTPUT_ROOT = Path("local_data/sft_index")
PHASE1_FILENAME = "phase1_easy_indexes.txt"
PHASE2_FILENAME = "phase2_hard_indexes.txt"
STATS_FILENAME = "selection_stats.json"
RATIO_GROUPS = (
    (1, "ratio < 0.5", None, 0.5),
    (2, "0.5 <= ratio < 0.8", 0.5, 0.8),
    (3, "0.8 <= ratio < 1.0", 0.8, 1.0),
    (4, "1.0 <= ratio < 1.2", 1.0, 1.2),
    (5, "1.2 <= ratio < 1.5", 1.2, 1.5),
    (6, "1.5 <= ratio < 2.0", 1.5, 2.0),
    (7, "ratio >= 2.0", 2.0, None),
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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


def is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 1
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true"}
    return False


def numeric_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def ratio_group_id(ratio: float) -> int:
    for group_id, _, lower, upper in RATIO_GROUPS:
        if lower is not None and ratio < lower:
            continue
        if upper is not None and ratio >= upper:
            continue
        return group_id
    raise ValueError(f"Unexpected ratio: {ratio}")


def group_definitions() -> dict[str, str]:
    return {f"group_{group_id}": label for group_id, label, _, _ in RATIO_GROUPS}


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(path))
    seen: set[str] = set()
    duplicates: list[str] = []
    for row_number, row in enumerate(rows, start=1):
        if "index" not in row:
            raise ValueError(f"Missing index at {path}:{row_number}")
        index = str(row["index"])
        if index in seen:
            duplicates.append(index)
        seen.add(index)

    if duplicates:
        raise ValueError(f"{path} contains duplicate index values. Examples: {duplicates[:5]}")
    return rows


def load_train_tokens(dataset_path: Path) -> dict[str, float]:
    dataset = pd.read_parquet(dataset_path, columns=["index", "train_token"])
    missing = {"index", "train_token"} - set(dataset.columns)
    if missing:
        raise ValueError(f"{dataset_path} is missing required columns: {sorted(missing)}")

    index_as_text = dataset["index"].astype(str)
    duplicated = index_as_text.duplicated()
    if duplicated.any():
        examples = index_as_text.loc[duplicated].head(5).tolist()
        raise ValueError(f"{dataset_path} contains duplicate index values: {examples}")

    return {
        str(row["index"]): float(row["train_token"])
        for row in dataset[["index", "train_token"]].to_dict("records")
    }


def infer_split_name(prediction_path: Path, dataset_path: Path) -> str:
    candidates = [dataset_path.stem, prediction_path.parent.name]
    for candidate in candidates:
        match = re.search(r"(?:^|_)((?:le|gt)\d+)(?:_|$)", candidate)
        if match:
            return match.group(1)
    return prediction_path.parent.name


def default_output_dir(prediction_path: Path, dataset_path: Path) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"{infer_split_name(prediction_path, dataset_path)}_curriculum"


def sample_rows(rows: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if count <= 0:
        return []
    if count > len(rows):
        raise ValueError(f"Cannot sample {count} rows from bucket with {len(rows)} rows")
    return rng.sample(rows, count)


def prepare_rows(
    prediction_rows: list[dict[str, Any]],
    train_tokens_by_index: dict[str, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[int], int]:
    incorrect_rows: list[dict[str, Any]] = []
    correct_rows: list[dict[str, Any]] = []
    group_counts: Counter[int] = Counter()
    missing_correct_ratio_count = 0

    for row in prediction_rows:
        index = str(row["index"])
        annotated = {**row, "_index": index}
        if not is_correct(row.get("is_correct")):
            incorrect_rows.append(annotated)
            continue

        output_token = numeric_value(row.get("output_token_length"))
        target_token = train_tokens_by_index.get(index)
        if output_token is None or target_token is None or target_token <= 0:
            annotated["_ratio"] = None
            annotated["_ratio_group"] = None
            missing_correct_ratio_count += 1
            correct_rows.append(annotated)
            continue

        ratio = output_token / target_token
        group_id = ratio_group_id(ratio)
        annotated["_ratio"] = ratio
        annotated["_ratio_group"] = group_id
        group_counts[group_id] += 1
        correct_rows.append(annotated)

    return incorrect_rows, correct_rows, group_counts, missing_correct_ratio_count


def select_hard_rows(
    incorrect_rows: list[dict[str, Any]],
    correct_rows: list[dict[str, Any]],
    target_hard_size: int,
    seed: int,
) -> tuple[set[str], dict[str, int]]:
    rng = random.Random(seed)
    by_group: dict[int, list[dict[str, Any]]] = {group_id: [] for group_id, *_ in RATIO_GROUPS}
    for row in correct_rows:
        group_id = row.get("_ratio_group")
        if group_id in by_group:
            by_group[int(group_id)].append(row)

    selected_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}

    selected_rows.extend(incorrect_rows)
    selected_counts["incorrect"] = len(incorrect_rows)

    for group_id in (7, 6, 5):
        rows = by_group[group_id]
        selected_rows.extend(rows)
        selected_counts[f"group_{group_id}_all"] = len(rows)

    group4_half = sample_rows(by_group[4], len(by_group[4]) // 2, rng)
    group3_half = sample_rows(by_group[3], len(by_group[3]) // 2, rng)
    selected_rows.extend(group4_half)
    selected_rows.extend(group3_half)
    selected_counts["group_4_half_floor"] = len(group4_half)
    selected_counts["group_3_half_floor"] = len(group3_half)

    selected_indexes = {row["_index"] for row in selected_rows}
    selected_counts["base_before_fill"] = len(selected_indexes)
    if len(selected_indexes) > target_hard_size:
        raise ValueError(
            f"Base hard selection has {len(selected_indexes)} rows, "
            f"which exceeds target {target_hard_size}."
        )

    remaining = target_hard_size - len(selected_indexes)
    fill_pool = [
        row
        for row in by_group[1] + by_group[2]
        if row["_index"] not in selected_indexes
    ]
    fill_rows = sample_rows(fill_pool, remaining, rng)
    selected_indexes.update(row["_index"] for row in fill_rows)
    selected_counts["group_1_2_fill"] = len(fill_rows)
    selected_counts["final_hard"] = len(selected_indexes)

    return selected_indexes, selected_counts


def write_index_file(path: Path, indexes: list[str]) -> None:
    path.write_text("".join(f"{index}\n" for index in indexes), encoding="utf-8")


def build_stats(
    *,
    prediction_path: Path,
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    target_hard_size: int,
    prediction_rows: list[dict[str, Any]],
    incorrect_rows: list[dict[str, Any]],
    correct_rows: list[dict[str, Any]],
    group_counts: Counter[int],
    missing_correct_ratio_count: int,
    selected_counts: dict[str, int],
    phase1_indexes: list[str],
    phase2_indexes: list[str],
) -> dict[str, Any]:
    phase1_set = set(phase1_indexes)
    phase2_set = set(phase2_indexes)
    all_set = {str(row["index"]) for row in prediction_rows}
    overlap = sorted(phase1_set & phase2_set)
    union = phase1_set | phase2_set

    return {
        "inputs": {
            "prediction": str(prediction_path),
            "dataset": str(dataset_path),
        },
        "outputs": {
            "output_dir": str(output_dir),
            "phase1_easy_indexes": str(output_dir / PHASE1_FILENAME),
            "phase2_hard_indexes": str(output_dir / PHASE2_FILENAME),
            "selection_stats": str(output_dir / STATS_FILENAME),
        },
        "selection": {
            "seed": seed,
            "target_hard_size": target_hard_size,
            "sampling_policy": "random deterministic",
            "half_rounding": "floor",
            "fill_pool": "group_1 + group_2",
            "ratio": "output_token_length / train_token",
        },
        "counts": {
            "total_rows": len(prediction_rows),
            "correct_rows": len(correct_rows),
            "incorrect_rows": len(incorrect_rows),
            "missing_correct_ratio_rows": missing_correct_ratio_count,
            "phase1_easy_rows": len(phase1_indexes),
            "phase2_hard_rows": len(phase2_indexes),
        },
        "ratio_group_definitions": group_definitions(),
        "ratio_group_counts": {
            f"group_{group_id}": group_counts.get(group_id, 0)
            for group_id, *_ in RATIO_GROUPS
        },
        "selected_counts": selected_counts,
        "checks": {
            "overlap_count": len(overlap),
            "overlap_examples": overlap[:10],
            "union_count": len(union),
            "union_matches_total_rows": union == all_set,
            "phase2_matches_target": len(phase2_indexes) == target_hard_size,
        },
    }


def select_curriculum_indexes(
    prediction_path: Path,
    dataset_path: Path,
    output_dir: Path | None,
    target_hard_size: int,
    seed: int,
) -> dict[str, Any]:
    if target_hard_size <= 0:
        raise ValueError("--target-hard-size must be positive")

    prediction_rows = load_prediction_rows(prediction_path)
    if target_hard_size > len(prediction_rows):
        raise ValueError(
            f"--target-hard-size {target_hard_size} exceeds total rows {len(prediction_rows)}"
        )

    train_tokens_by_index = load_train_tokens(dataset_path)
    incorrect_rows, correct_rows, group_counts, missing_correct_ratio_count = prepare_rows(
        prediction_rows,
        train_tokens_by_index,
    )
    hard_indexes, selected_counts = select_hard_rows(
        incorrect_rows,
        correct_rows,
        target_hard_size,
        seed,
    )

    all_indexes_in_prediction_order = [str(row["index"]) for row in prediction_rows]
    phase2_indexes = [index for index in all_indexes_in_prediction_order if index in hard_indexes]
    phase1_indexes = [index for index in all_indexes_in_prediction_order if index not in hard_indexes]

    resolved_output_dir = output_dir or default_output_dir(prediction_path, dataset_path)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    write_index_file(resolved_output_dir / PHASE1_FILENAME, phase1_indexes)
    write_index_file(resolved_output_dir / PHASE2_FILENAME, phase2_indexes)

    stats = build_stats(
        prediction_path=prediction_path,
        dataset_path=dataset_path,
        output_dir=resolved_output_dir,
        seed=seed,
        target_hard_size=target_hard_size,
        prediction_rows=prediction_rows,
        incorrect_rows=incorrect_rows,
        correct_rows=correct_rows,
        group_counts=group_counts,
        missing_correct_ratio_count=missing_correct_ratio_count,
        selected_counts=selected_counts,
        phase1_indexes=phase1_indexes,
        phase2_indexes=phase2_indexes,
    )
    (resolved_output_dir / STATS_FILENAME).write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select easy/hard index files for two-phase SFT curriculum training."
    )
    parser.add_argument("--prediction", required=True, help="Path to prediction.jsonl")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to dataset parquet with index and train_token columns",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to local_data/sft_index/<split>_curriculum",
    )
    parser.add_argument("--target-hard-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = select_curriculum_indexes(
        prediction_path=Path(args.prediction),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        target_hard_size=args.target_hard_size,
        seed=args.seed,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
