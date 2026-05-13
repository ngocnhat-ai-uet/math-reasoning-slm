import argparse
import re
from pathlib import Path

import pandas as pd


TRAIN_DIR = Path("data/train")
DATASET_PATHS = {
    "gt2000": TRAIN_DIR / "omni_dataset_gt2000_sub12k.parquet",
    "le2000": TRAIN_DIR / "omni_dataset_le2000_sub12k.parquet",
}

BOXED_RE = re.compile(r"\\boxed\s*\{")


def boxed_count(solution: object) -> int:
    if not isinstance(solution, str):
        return 0
    return len(BOXED_RE.findall(solution))


def summarize_dataset(name: str, path: Path, example_limit: int) -> tuple[int, int]:
    df = pd.read_parquet(path, columns=["index", "solution"])
    counts = df["solution"].map(boxed_count)
    multi_box = df.loc[counts > 1, ["index"]].copy()
    multi_box["boxed_count"] = counts.loc[counts > 1].to_numpy()

    print(f"{name}:")
    print(f"  path: {path}")
    print(f"  total rows: {len(df)}")
    print(f"  rows with >1 boxed answer in solution: {len(multi_box)}")

    if example_limit > 0 and not multi_box.empty:
        print(f"  examples (index, boxed_count):")
        for row in multi_box.head(example_limit).itertuples(index=False):
            print(f"    {row.index}\t{row.boxed_count}")

    return len(df), len(multi_box)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count rows whose solution contains more than one \\boxed{...} answer."
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=10,
        help="Number of example indexes to print per dataset. Use 0 to hide examples.",
    )
    args = parser.parse_args()

    total_rows = 0
    total_multi_box = 0
    for name, path in DATASET_PATHS.items():
        rows, multi_box = summarize_dataset(name, path, args.examples)
        total_rows += rows
        total_multi_box += multi_box
        print()

    print("combined:")
    print(f"  total rows: {total_rows}")
    print(f"  rows with >1 boxed answer in solution: {total_multi_box}")


if __name__ == "__main__":
    main()
