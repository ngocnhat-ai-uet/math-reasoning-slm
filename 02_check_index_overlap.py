import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_SUFFIXES = {".parquet", ".jsonl", ".json"}


def read_index_column(path: Path) -> pd.Series:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"{path} has unsupported extension '{path.suffix}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_SUFFIXES))}"
        )

    if suffix == ".parquet":
        df = pd.read_parquet(path, columns=["index"])
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = read_json_file(path)

    if "index" not in df.columns:
        raise ValueError(f"{path} does not contain required field/column 'index'")

    return df["index"]


def read_json_file(path: Path) -> pd.DataFrame:
    try:
        return pd.read_json(path)
    except ValueError:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict):
            for key in ("data", "rows", "items", "records"):
                if isinstance(data.get(key), list):
                    return pd.DataFrame(data[key])
            return pd.DataFrame([data])

        raise ValueError(f"{path} must contain a JSON object or list of objects")


def index_stats(indexes: pd.Series) -> dict[str, Any]:
    non_null = indexes.dropna()
    duplicated = non_null[non_null.duplicated()].drop_duplicates().tolist()
    return {
        "rows": len(indexes),
        "null_count": int(indexes.isna().sum()),
        "unique_count": int(non_null.nunique()),
        "duplicate_count": int(len(non_null) - non_null.nunique()),
        "duplicate_examples": duplicated[:10],
        "set": set(non_null.tolist()),
    }


def format_examples(values: set[Any], limit: int) -> str:
    examples = sorted(values, key=lambda value: str(value))[:limit]
    return json.dumps(examples, ensure_ascii=False)


def compare_indexes(left_path: Path, right_path: Path, sample_size: int) -> None:
    left_stats = index_stats(read_index_column(left_path))
    right_stats = index_stats(read_index_column(right_path))

    left_indexes = left_stats["set"]
    right_indexes = right_stats["set"]

    common = left_indexes & right_indexes
    left_only = left_indexes - right_indexes
    right_only = right_indexes - left_indexes

    print(f"Left file : {left_path}")
    print(f"Right file: {right_path}")
    print()
    print("Index summary")
    print(
        f"- Left : rows={left_stats['rows']}, unique_index={left_stats['unique_count']}, "
        f"null_index={left_stats['null_count']}, duplicate_index={left_stats['duplicate_count']}"
    )
    print(
        f"- Right: rows={right_stats['rows']}, unique_index={right_stats['unique_count']}, "
        f"null_index={right_stats['null_count']}, duplicate_index={right_stats['duplicate_count']}"
    )
    print()
    print("Set comparison")
    print(f"- Common indexes              : {len(common)}")
    print(f"- Indexes only in left file   : {len(left_only)}")
    print(f"- Indexes only in right file  : {len(right_only)}")
    print(f"- Same unique index set       : {left_indexes == right_indexes}")

    if left_only:
        print(f"- Left-only examples          : {format_examples(left_only, sample_size)}")
    if right_only:
        print(f"- Right-only examples         : {format_examples(right_only, sample_size)}")
    if left_stats["duplicate_examples"]:
        print(
            f"- Left duplicate examples     : "
            f"{json.dumps(left_stats['duplicate_examples'], ensure_ascii=False)}"
        )
    if right_stats["duplicate_examples"]:
        print(
            f"- Right duplicate examples    : "
            f"{json.dumps(right_stats['duplicate_examples'], ensure_ascii=False)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the unique 'index' values of two parquet/jsonl/json files."
        )
    )
    parser.add_argument("left", type=Path, help="First input file")
    parser.add_argument("right", type=Path, help="Second input file")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of example missing indexes to print for each side",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_indexes(args.left, args.right, args.sample_size)


if __name__ == "__main__":
    main()
