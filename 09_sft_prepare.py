import argparse
import json
import logging
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "local_data/omni/omni_dataset_gt2000.parquet"
DEFAULT_OUTPUT = "data/sft/gt2000_full.json"
FINAL_ANSWER_INSTRUCTION = r"Put the final answer inside \boxed{}."


def clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def build_output(thought, solution):
    thought = clean_text(thought)
    solution = clean_text(solution)
    if thought:
        return f"<think>\n{thought}\n</think>\n{solution}"
    return solution


def load_index_file(index_path):
    indexes = []
    seen = set()
    duplicates = []
    with index_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            index = line.strip()
            if not index:
                continue
            if index in seen:
                duplicates.append(index)
            seen.add(index)
            indexes.append(index)

    if duplicates:
        raise ValueError(
            f"{index_path} contains duplicate indexes. Examples: {duplicates[:5]}"
        )
    if not indexes:
        raise ValueError(f"{index_path} contains no indexes")
    return indexes


def filter_by_index(df, index_path):
    if "index" not in df.columns:
        raise ValueError("--index-file requires input parquet to contain an index column")

    indexes = load_index_file(index_path)
    indexed_df = df.assign(_index_text=df["index"].astype(str))
    duplicated = indexed_df["_index_text"].duplicated()
    if duplicated.any():
        examples = indexed_df.loc[duplicated, "_index_text"].head(5).tolist()
        raise ValueError(f"Input dataset contains duplicate indexes: {examples}")

    row_by_index = indexed_df.set_index("_index_text", drop=False)
    missing_indexes = [index for index in indexes if index not in row_by_index.index]
    if missing_indexes:
        raise ValueError(
            f"{index_path} has {len(missing_indexes)} indexes missing from dataset. "
            f"Examples: {missing_indexes[:5]}"
        )

    return row_by_index.loc[indexes].drop(columns=["_index_text"]).reset_index(drop=True)


def convert_dataset(input_path, output_path, index_file=None, limit=None):
    df = pd.read_parquet(input_path)

    required_columns = {"question", "solution"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if index_file is not None:
        df = filter_by_index(df, index_file)

    if limit is not None:
        df = df.head(limit)

    records = []
    skipped = 0
    for _, row in df.iterrows():
        question = clean_text(row.get("question"))
        solution = clean_text(row.get("solution"))
        if not question or not solution:
            skipped += 1
            continue

        question = f"{FINAL_ANSWER_INSTRUCTION} {question}"

        record = {
            "instruction": question,
            "output": build_output(row.get("thought"), solution),
        }

        index = clean_text(row.get("index"))
        if index:
            record["index"] = index

        records.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    logging.info("Wrote %d records to %s", len(records), output_path)
    if skipped:
        logging.warning("Skipped %d rows with empty question or solution", skipped)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, help="input parquet dataset path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="output SFT json path")
    parser.add_argument(
        "--index-file",
        default=None,
        help="optional text file with one index per line; output follows this index order",
    )
    parser.add_argument("--limit", type=int, default=None, help="optional number of rows to convert")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert_dataset(
        Path(args.input),
        Path(args.output),
        Path(args.index_file) if args.index_file else None,
        args.limit,
    )


if __name__ == "__main__":
    main()
