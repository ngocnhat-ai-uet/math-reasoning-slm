import argparse
import json
import re
from pathlib import Path

import pandas as pd


TRAIN_DIR = Path("data/train")
HINT_CHECK_DIR = Path("data/hints_check_valid_v2")

SPLITS = {
    "gt2000": {
        "dataset": TRAIN_DIR / "omni_dataset_gt2000_valid_no_multibox_v1.parquet",
        "hints": TRAIN_DIR / "omni_hints_gt2000_valid_no_multibox_v1.parquet",
        "invalid_index": HINT_CHECK_DIR / "gt2000" / "gt2000_invalid_index.txt",
        "dataset_output": TRAIN_DIR / "omni_dataset_gt2000_valid_no_multibox_v2.parquet",
        "hints_output": TRAIN_DIR / "omni_hints_gt2000_valid_no_multibox_v2.parquet",
    },
    "le2000": {
        "dataset": TRAIN_DIR / "omni_dataset_le2000_valid_no_multibox_v1.parquet",
        "hints": TRAIN_DIR / "omni_hints_le2000_valid_no_multibox_v1.parquet",
        "invalid_index": HINT_CHECK_DIR / "le2000" / "le2000_invalid_index.txt",
        "dataset_output": TRAIN_DIR / "omni_dataset_le2000_valid_no_multibox_v2.parquet",
        "hints_output": TRAIN_DIR / "omni_hints_le2000_valid_no_multibox_v2.parquet",
    },
}

BOXED_RE = re.compile(r"\\boxed\s*\{")


def read_invalid_indexes(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Invalid index file not found: {path}")

    indexes = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        index = line.strip()
        if index:
            indexes.add(index)
    return indexes


def count_boxed_answers(solution: object) -> int:
    if not isinstance(solution, str):
        return 0
    return len(BOXED_RE.findall(solution))


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def require_unique_index(df: pd.DataFrame, path: Path) -> None:
    index_as_text = df["index"].astype(str)
    duplicated = index_as_text.duplicated()
    if duplicated.any():
        examples = index_as_text.loc[duplicated].head(5).tolist()
        raise ValueError(f"{path} contains duplicated index values: {examples}")


def align_hints_to_indexes(hints_path: Path, indexes: list[str]) -> pd.DataFrame:
    hints = pd.read_parquet(hints_path)
    require_columns(hints, hints_path, {"index"})
    require_unique_index(hints, hints_path)

    hints_by_index = hints.assign(_index_key=hints["index"].astype(str)).set_index(
        "_index_key",
        drop=False,
    )
    missing_indexes = [idx for idx in indexes if idx not in hints_by_index.index]
    if missing_indexes:
        raise ValueError(
            f"{hints_path} is missing {len(missing_indexes)} filtered indexes. "
            f"Examples: {missing_indexes[:5]}"
        )

    return hints_by_index.loc[indexes].drop(columns=["_index_key"]).reset_index(drop=True)


def build_filtered_dataset(
    split: str,
    dataset_path: Path,
    hints_path: Path,
    invalid_index_path: Path,
    dataset_output_path: Path,
    hints_output_path: Path,
) -> dict:
    dataset = pd.read_parquet(dataset_path)
    require_columns(dataset, dataset_path, {"index", "solution"})
    require_unique_index(dataset, dataset_path)

    invalid_indexes = read_invalid_indexes(invalid_index_path)
    index_as_text = dataset["index"].astype(str)

    invalid_mask = index_as_text.isin(invalid_indexes)
    after_invalid_filter = dataset.loc[~invalid_mask].copy()

    boxed_counts = after_invalid_filter["solution"].map(count_boxed_answers)
    multibox_mask = boxed_counts > 1
    multibox_rows = after_invalid_filter.loc[multibox_mask].copy()
    filtered = after_invalid_filter.loc[~multibox_mask].reset_index(drop=True)
    filtered_indexes = filtered["index"].astype(str).tolist()
    filtered_hints = align_hints_to_indexes(hints_path, filtered_indexes)

    dataset_output_path.parent.mkdir(parents=True, exist_ok=True)
    hints_output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(dataset_output_path, index=False)
    filtered_hints.to_parquet(hints_output_path, index=False)

    stats = {
        "split": split,
        "dataset_path": str(dataset_path),
        "hints_path": str(hints_path),
        "invalid_index_path": str(invalid_index_path),
        "dataset_output_path": str(dataset_output_path),
        "hints_output_path": str(hints_output_path),
        "original_rows": int(len(dataset)),
        "invalid_indexes_in_file": int(len(invalid_indexes)),
        "invalid_rows_matched": int(invalid_mask.sum()),
        "rows_after_invalid_filter": int(len(after_invalid_filter)),
        "multibox_rows_after_invalid_filter": int(len(multibox_rows)),
        "rows_after_multibox_filter": int(len(filtered)),
        "hints_rows_after_filter": int(len(filtered_hints)),
    }
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Filter invalid indexes, remove rows whose solution contains more than "
            "one \\boxed{...}, then write filtered datasets and aligned hints."
        )
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=TRAIN_DIR / "omni_dataset_valid_no_multibox_stats_v2.json",
        help="Path to write JSON statistics.",
    )
    args = parser.parse_args()

    all_stats = []
    for split, paths in SPLITS.items():
        stats = build_filtered_dataset(
            split=split,
            dataset_path=paths["dataset"],
            hints_path=paths["hints"],
            invalid_index_path=paths["invalid_index"],
            dataset_output_path=paths["dataset_output"],
            hints_output_path=paths["hints_output"],
        )
        all_stats.append(stats)

        print(f"{split}:")
        print(f"  original rows: {stats['original_rows']}")
        print(f"  invalid indexes in file: {stats['invalid_indexes_in_file']}")
        print(f"  invalid rows matched: {stats['invalid_rows_matched']}")
        print(f"  rows after invalid filter: {stats['rows_after_invalid_filter']}")
        print(
            "  multibox rows after invalid filter: "
            f"{stats['multibox_rows_after_invalid_filter']}"
        )
        print(f"  rows after multibox filter: {stats['rows_after_multibox_filter']}")
        print(f"  hint rows after filter: {stats['hints_rows_after_filter']}")
        print(f"  dataset output: {stats['dataset_output_path']}")
        print(f"  hints output: {stats['hints_output_path']}")
        print()

    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(
        json.dumps(all_stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"stats output: {args.stats_output}")


if __name__ == "__main__":
    main()
