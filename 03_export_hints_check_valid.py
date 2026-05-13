import argparse
import json
from pathlib import Path

import pandas as pd


TRAIN_DIR = Path("data/train")
OUTPUT_ROOT = Path("data/hints_check_valid_v2")
CHUNK_SIZE = 1_000

SPLITS = {
    "gt2000": {
        "dataset": TRAIN_DIR / "omni_dataset_gt2000_valid_no_multibox.parquet",
        "hints": TRAIN_DIR / "omni_hints_gt2000_valid_no_multibox.parquet",
    },
    "le2000": {
        "dataset": TRAIN_DIR / "omni_dataset_le2000_valid_no_multibox.parquet",
        "hints": TRAIN_DIR / "omni_hints_le2000_valid_no_multibox.parquet",
    },
}


def require_columns(df: pd.DataFrame, path: Path, columns: set[str]) -> None:
    missing = columns - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")


def require_unique_index(df: pd.DataFrame, path: Path) -> None:
    duplicated = df["index"].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "index"].head(5).tolist()
        raise ValueError(f"{path} contains duplicated index values: {examples}")


def load_joined_records(dataset_path: Path, hints_path: Path) -> list[dict]:
    dataset = pd.read_parquet(dataset_path)
    hints = pd.read_parquet(hints_path)

    require_columns(dataset, dataset_path, {"index", "question", "final_answer"})
    require_columns(hints, hints_path, {"index", "detailed_method_hint"})
    require_unique_index(dataset, dataset_path)
    require_unique_index(hints, hints_path)

    hints_by_index = hints.set_index("index", drop=False)
    missing_indexes = [idx for idx in dataset["index"].tolist() if idx not in hints_by_index.index]
    if missing_indexes:
        raise ValueError(
            f"{hints_path} is missing {len(missing_indexes)} dataset indexes. "
            f"Examples: {missing_indexes[:5]}"
        )

    aligned_hints = hints_by_index.loc[dataset["index"].tolist()].reset_index(drop=True)
    records = []
    for dataset_row, hint_row in zip(dataset.to_dict("records"), aligned_hints.to_dict("records")):
        records.append(
            {
                "index": dataset_row["index"],
                "question": dataset_row["question"],
                "final_answer": dataset_row["final_answer"],
                "hint": hint_row["detailed_method_hint"],
            }
        )
    return records


def write_chunks(split: str, records: list[dict], output_root: Path, chunk_size: int) -> None:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    split_dir = output_root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    total_parts = (len(records) + chunk_size - 1) // chunk_size
    row_width = len(str(len(records)))

    for part_idx in range(total_parts):
        start = part_idx * chunk_size
        end = min(start + chunk_size, len(records))
        output_path = split_dir / (
            f"{split}_detailed_method_hint_part{part_idx + 1:02d}_of{total_parts:02d}"
            f"_rows{start + 1:0{row_width}d}-{end:0{row_width}d}.json"
        )
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(records[start:end], file, ensure_ascii=False, indent=2)
        print(f"{split}: wrote {end - start} records -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export detailed-method hints into JSON chunks for manual validation."
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        help="Optional split to export. If omitted, all splits are exported.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_splits = {args.split: SPLITS[args.split]} if args.split else SPLITS
    for split, paths in selected_splits.items():
        records = load_joined_records(paths["dataset"], paths["hints"])
        write_chunks(split, records, args.output_root, args.chunk_size)


if __name__ == "__main__":
    main()
