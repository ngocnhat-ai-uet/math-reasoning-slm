from pathlib import Path

import pandas as pd


TRAIN_DIR = Path("data/train")
SAMPLE_SIZE = 10_000
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


def subset_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}_subset10k_seed{SEED}{path.suffix}")


def require_index_column(df: pd.DataFrame, path: Path) -> None:
    if "index" not in df.columns:
        raise ValueError(f"{path} does not contain required column 'index'")


def require_unique_index(df: pd.DataFrame, path: Path) -> None:
    duplicated = df["index"].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "index"].head(5).tolist()
        raise ValueError(f"{path} contains duplicated index values: {examples}")


def create_subset(name: str, dataset_path: Path, hints_path: Path) -> None:
    dataset = pd.read_parquet(dataset_path)
    hints = pd.read_parquet(hints_path)

    require_index_column(dataset, dataset_path)
    require_index_column(hints, hints_path)
    require_unique_index(dataset, dataset_path)
    require_unique_index(hints, hints_path)

    if len(dataset) < SAMPLE_SIZE:
        raise ValueError(
            f"{dataset_path} has only {len(dataset)} rows, need {SAMPLE_SIZE}"
        )

    dataset_subset = dataset.sample(n=SAMPLE_SIZE, random_state=SEED)
    sampled_indexes = dataset_subset["index"].tolist()

    hints_by_index = hints.set_index("index", drop=False)
    missing_indexes = [idx for idx in sampled_indexes if idx not in hints_by_index.index]
    if missing_indexes:
        raise ValueError(
            f"{hints_path} is missing {len(missing_indexes)} sampled indexes. "
            f"Examples: {missing_indexes[:5]}"
        )

    hints_subset = hints_by_index.loc[sampled_indexes].reset_index(drop=True)

    dataset_output = subset_path(dataset_path)
    hints_output = subset_path(hints_path)
    dataset_subset.to_parquet(dataset_output, index=False)
    hints_subset.to_parquet(hints_output, index=False)

    print(f"{name}: wrote {len(dataset_subset)} dataset rows -> {dataset_output}")
    print(f"{name}: wrote {len(hints_subset)} hint rows -> {hints_output}")


def main() -> None:
    for name, paths in SPLITS.items():
        create_subset(name, paths["dataset"], paths["hints"])


if __name__ == "__main__":
    main()
