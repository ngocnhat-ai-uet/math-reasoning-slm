from pathlib import Path
import argparse

import pandas as pd


TRAIN_DIR = Path("data/train")
SAMPLE_SIZE = 12_000
SEED = 9

SPLITS = {
    "gt2000": {
        "dataset": TRAIN_DIR / "omni_math_dataset_gt2000.parquet",
        "hints": TRAIN_DIR / "omni_hints_checkpoint_gt2000.parquet",
    },
    "le2000": {
        "dataset": TRAIN_DIR / "omni_math_dataset_le2000.parquet",
        "hints": TRAIN_DIR / "omni_hints_checkpoint_le2000.parquet",
    },
}


def sample_size_label(sample_size: int) -> str:
    if sample_size % 1000 == 0:
        return f"{sample_size // 1000}k"
    return str(sample_size)


def format_threshold_value(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def subset_path(
    path: Path,
    sample_size: int,
    seed: int,
    threshold_column: str | None = None,
    max_threshold: float | None = None,
) -> Path:
    threshold_part = ""
    if threshold_column and max_threshold is not None:
        threshold_part = f"_{threshold_column}_le{format_threshold_value(max_threshold)}"
    return path.with_name(
        f"{path.stem}{threshold_part}_sub{sample_size_label(sample_size)}_seed{seed}{path.suffix}"
    )


def require_index_column(df: pd.DataFrame, path: Path) -> None:
    if "index" not in df.columns:
        raise ValueError(f"{path} does not contain required column 'index'")


def require_unique_index(df: pd.DataFrame, path: Path) -> None:
    duplicated = df["index"].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "index"].head(5).tolist()
        raise ValueError(f"{path} contains duplicated index values: {examples}")


def filter_by_threshold(
    dataset: pd.DataFrame,
    dataset_path: Path,
    threshold_column: str | None,
    max_threshold: float | None,
) -> pd.DataFrame:
    if threshold_column is None and max_threshold is None:
        return dataset
    if threshold_column is None or max_threshold is None:
        raise ValueError("--threshold-column and --max-threshold must be provided together")
    if threshold_column not in dataset.columns:
        raise ValueError(f"{dataset_path} does not contain threshold column '{threshold_column}'")

    return dataset.loc[dataset[threshold_column] <= max_threshold].copy()


def create_subset(
    name: str,
    dataset_path: Path,
    hints_path: Path,
    sample_size: int,
    seed: int,
    threshold_column: str | None = None,
    max_threshold: float | None = None,
) -> None:
    dataset = pd.read_parquet(dataset_path)
    hints = pd.read_parquet(hints_path)

    require_index_column(dataset, dataset_path)
    require_index_column(hints, hints_path)
    require_unique_index(dataset, dataset_path)
    require_unique_index(hints, hints_path)

    original_count = len(dataset)
    dataset = filter_by_threshold(dataset, dataset_path, threshold_column, max_threshold)
    if len(dataset) < sample_size:
        filter_desc = ""
        if threshold_column and max_threshold is not None:
            filter_desc = (
                f" after filtering {threshold_column} <= {format_threshold_value(max_threshold)}"
            )
        raise ValueError(
            f"{dataset_path} has only {len(dataset)} rows{filter_desc}, need {sample_size}"
        )

    dataset_subset = dataset.sample(n=sample_size, random_state=seed)
    sampled_indexes = dataset_subset["index"].tolist()

    hints_by_index = hints.set_index("index", drop=False)
    missing_indexes = [idx for idx in sampled_indexes if idx not in hints_by_index.index]
    if missing_indexes:
        raise ValueError(
            f"{hints_path} is missing {len(missing_indexes)} sampled indexes. "
            f"Examples: {missing_indexes[:5]}"
        )

    hints_subset = hints_by_index.loc[sampled_indexes].reset_index(drop=True)

    dataset_output = subset_path(dataset_path, sample_size, seed, threshold_column, max_threshold)
    hints_output = subset_path(hints_path, sample_size, seed, threshold_column, max_threshold)
    dataset_subset.to_parquet(dataset_output, index=False)
    hints_subset.to_parquet(hints_output, index=False)

    if threshold_column and max_threshold is not None:
        print(
            f"{name}: kept {len(dataset)} of {original_count} dataset rows "
            f"where {threshold_column} <= {format_threshold_value(max_threshold)}"
        )
    print(f"{name}: wrote {len(dataset_subset)} dataset rows -> {dataset_output}")
    print(f"{name}: wrote {len(hints_subset)} hint rows -> {hints_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create matched random dataset/hint subsets by index."
    )
    parser.add_argument(
        "--split",
        choices=sorted(SPLITS),
        help="Optional split to process. If omitted, all splits are processed.",
    )
    parser.add_argument("--sample-size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threshold-column", help="Dataset column used for upper-bound filtering.")
    parser.add_argument(
        "--max-threshold",
        "--thredsold",
        dest="max_threshold",
        type=float,
        help="Keep dataset rows where threshold-column <= this value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_splits = {args.split: SPLITS[args.split]} if args.split else SPLITS
    for name, paths in selected_splits.items():
        create_subset(
            name,
            paths["dataset"],
            paths["hints"],
            args.sample_size,
            args.seed,
            args.threshold_column,
            args.max_threshold,
        )


if __name__ == "__main__":
    main()
