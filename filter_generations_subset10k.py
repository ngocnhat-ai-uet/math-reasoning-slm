import json
from pathlib import Path

import pandas as pd


TRAIN_DIR = Path("data/train")
RUNS_DIR = Path("experiments/TN01_base_inference/runs")
OUTPUT_NAME = "generations_10k.jsonl"

INDEX_FILES = {
    "gt2000": TRAIN_DIR / "omni_dataset_gt2000.parquet",
    "le2000": TRAIN_DIR / "omni_dataset_le2000.parquet",
}


def load_indexes(path: Path) -> set[str]:
    df = pd.read_parquet(path, columns=["index"])
    indexes = set(df["index"].astype(str))
    if len(indexes) != len(df):
        raise ValueError(f"{path} contains duplicated index values")
    return indexes


def split_name_from_run_id(run_id: str) -> str:
    if "gt2000" in run_id:
        return "gt2000"
    if "le2000" in run_id:
        return "le2000"
    raise ValueError(f"Cannot infer gt2000/le2000 from run_id: {run_id}")


def filter_generation_file(input_path: Path, indexes_by_split: dict[str, set[str]]) -> None:
    output_path = input_path.with_name(OUTPUT_NAME)
    kept = 0
    total = 0
    split_name = None
    seen_indexes: set[str] = set()

    with input_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)

            row_run_id = str(row.get("run_id", input_path.parent.name))
            row_split_name = split_name_from_run_id(row_run_id)
            if split_name is None:
                split_name = row_split_name
            elif split_name != row_split_name:
                raise ValueError(
                    f"{input_path}:{line_number} mixes run splits: "
                    f"{split_name} and {row_split_name}"
                )

            index = str(row["index"])
            if index in indexes_by_split[row_split_name]:
                dst.write(line if line.endswith("\n") else f"{line}\n")
                kept += 1
                seen_indexes.add(index)

    if split_name is None:
        raise ValueError(f"{input_path} has no JSONL records")

    expected = len(indexes_by_split[split_name])
    missing = expected - len(seen_indexes)
    if kept != expected or missing:
        raise ValueError(
            f"{input_path}: kept {kept} rows with {len(seen_indexes)} unique indexes, "
            f"expected {expected}; missing {missing}"
        )

    print(f"{input_path} ({split_name}): {kept}/{total} -> {output_path}")


def main() -> None:
    indexes_by_split = {
        split_name: load_indexes(index_path)
        for split_name, index_path in INDEX_FILES.items()
    }

    generation_paths = sorted(RUNS_DIR.glob("*/generations.jsonl"))
    if not generation_paths:
        raise FileNotFoundError(f"No generations.jsonl files found under {RUNS_DIR}")

    for generation_path in generation_paths:
        filter_generation_file(generation_path, indexes_by_split)


if __name__ == "__main__":
    main()
