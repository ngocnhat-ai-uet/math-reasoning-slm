import argparse
import json
import logging
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = "data/train/omni_dataset_gt2000.parquet"
DEFAULT_OUTPUT = "data/train/omni_dataset_gt2000_sft.json"
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


def convert_dataset(input_path, output_path, limit=None):
    df = pd.read_parquet(input_path)

    required_columns = {"question", "solution"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

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
    parser.add_argument("--limit", type=int, default=None, help="optional number of rows to convert")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert_dataset(Path(args.input), Path(args.output), args.limit)


if __name__ == "__main__":
    main()
