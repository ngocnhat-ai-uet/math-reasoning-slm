from pathlib import Path
import argparse

import pandas as pd


TRAIN_DIR = Path("data/train")
SAMPLE_SIZE = 10_000
SEED = 9

SPLITS = {
    "gt2000": {
        "dataset": TRAIN_DIR / "omni_dataset_gt2000_valid_no_multibox_v2.parquet",
        "hints": TRAIN_DIR / "omni_hints_gt2000_valid_no_multibox_v2.parquet",
    },
    "le2000": {
        "dataset": TRAIN_DIR / "omni_dataset_le2000_valid_no_multibox_v2.parquet",
        "hints": TRAIN_DIR / "omni_hints_le2000_valid_no_multibox_v2.parquet",
    },
}


def sample_size_label(sample_size: int) -> str:
    if sample_size % 1000 == 0:
        return f"{sample_size // 1000}k"
    return str(sample_size)


def subset_path(path: Path, sample_size: int, seed: int) -> Path:
    return path.with_name(
        f"{path.stem}_sub{sample_size_label(sample_size)}_seed{seed}{path.suffix}"
    )


def require_index_column(df: pd.DataFrame, path: Path) -> None:
    if "index" not in df.columns:
        raise ValueError(f"{path} does not contain required column 'index'")


def require_unique_index(df: pd.DataFrame, path: Path) -> None:
    index_as_text = df["index"].astype(str)
    duplicated = index_as_text.duplicated()
    if duplicated.any():
        examples = index_as_text.loc[duplicated].head(5).tolist()
        raise ValueError(f"{path} contains duplicated index values: {examples}")


def create_subset(
    name: str,
    dataset_path: Path,
    hints_path: Path,
    sample_size: int,
    seed: int,
) -> None:
    dataset = pd.read_parquet(dataset_path)
    hints = pd.read_parquet(hints_path)

    require_index_column(dataset, dataset_path)
    require_index_column(hints, hints_path)
    require_unique_index(dataset, dataset_path)
    require_unique_index(hints, hints_path)

    if len(dataset) < sample_size:
        raise ValueError(f"{dataset_path} has only {len(dataset)} rows, need {sample_size}")

    dataset_subset = dataset.sample(n=sample_size, random_state=seed)
    sampled_indexes = dataset_subset["index"].astype(str).tolist()

    hints_by_index = hints.assign(_index_key=hints["index"].astype(str)).set_index(
        "_index_key",
        drop=False,
    )
    missing_indexes = [idx for idx in sampled_indexes if idx not in hints_by_index.index]
    if missing_indexes:
        raise ValueError(
            f"{hints_path} is missing {len(missing_indexes)} sampled indexes. "
            f"Examples: {missing_indexes[:5]}"
        )

    hints_subset = (
        hints_by_index.loc[sampled_indexes]
        .drop(columns=["_index_key"])
        .reset_index(drop=True)
    )

    dataset_output = subset_path(dataset_path, sample_size, seed)
    hints_output = subset_path(hints_path, sample_size, seed)
    dataset_subset.to_parquet(dataset_output, index=False)
    hints_subset.to_parquet(hints_output, index=False)

    print(f"{name}: wrote {len(dataset_subset)} dataset rows -> {dataset_output}")
    print(f"{name}: wrote {len(hints_subset)} hint rows -> {hints_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matched 10k random dataset/hint subsets from v2 parquet files."
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        help="Optional split to process. If omitted, all splits are processed.",
    )
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_splits = {args.split: SPLITS[args.split]} if args.split else SPLITS
    for name, paths in selected_splits.items():
        create_subset(
            name=name,
            dataset_path=paths["dataset"],
            hints_path=paths["hints"],
            sample_size=args.sample_size,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
