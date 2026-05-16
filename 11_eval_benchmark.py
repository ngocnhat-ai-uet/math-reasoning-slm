import argparse
import copy
import json
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT = r"You are a careful mathematical reasoning assistant. Before responding, reason in <think>...</think>. Use it to understand the task, identify relevant constraints, reason logically, verify key steps, and correct mistakes when found. Then respond clearly and follow the user's instructions."
FINAL_ANSWER_INSTRUCTION = r"Put the final answer inside \boxed{}."


# Input: dataset_path (jsonl/parquet)
# Output: list of dict
def load_records(dataset_config):
    dataset_path = dataset_config["data_path"]
    suffix = Path(dataset_path).suffix.lower()

    if suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8-sig") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {dataset_path}")
        return data

    if suffix == ".jsonl":
        records = []
        with open(dataset_path, "r", encoding="utf-8-sig") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=dataset_path)["train"]
        return [dict(item) for item in dataset]

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def get_question(record):
    for field in ("question", "instruction", "prompt"):
        if field in record and record[field] is not None:
            return str(record[field])
    raise ValueError(f"Record has no question/instruction/prompt field: {record}")


def get_index(record, fallback_index):
    for field in ("index", "id", "question_idx"):
        if field in record and record[field] is not None:
            return record[field]
    return fallback_index


def get_dataset_name(record):
    dataset = record.get("dataset")
    if dataset is None:
        return "unknown"
    return str(dataset)


def build_prompt_text(question):
    question = f"{FINAL_ANSWER_INSTRUCTION} {question}"
    return question


def load_tokenizer_and_vllm(config):
    model_path = config["models"].get("student") or config["models"].get("model")
    model_revision = config["models"].get("revision")
    if not model_path:
        raise ValueError("Config must define models.student or models.model")

    logging.info(f"Loading ckpt and tokenizer: {model_path}")
    if model_revision:
        logging.info(f"Using model/tokenizer revision: {model_revision}")
    tokenizer_kwargs = {"trust_remote_code": True}
    if model_revision:
        tokenizer_kwargs["revision"] = model_revision
    tokenizer = AutoTokenizer.from_pretrained(model_path, **tokenizer_kwargs)
    tokenizer.padding_side = "left"

    if tokenizer.eos_token is None:
        raise ValueError("No available eos_token.")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logging.info(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    logging.info(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")

    num_gpus = torch.cuda.device_count()
    attention_backend = config["inference"].get("attention_backend")
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
        logging.info(f"Using vLLM attention backend: {attention_backend}")

    llm_kwargs = dict(
        model=model_path,
        tensor_parallel_size=num_gpus,
        enable_chunked_prefill=config["inference"]["enable_chunked_prefill"],
        gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
        trust_remote_code=config["inference"]["trust_remote_code"],
        dtype=torch.bfloat16,
        enforce_eager=config["inference"]["enforce_eager"],
        max_model_len=config["inference"]["max_model_len"],
    )
    if model_revision:
        llm_kwargs["revision"] = model_revision
    llm = LLM(**llm_kwargs)
    logging.info("vLLM model loaded successfully")
    return tokenizer, llm


def render_inputs(records, config, tokenizer):
    rendered = []
    for index, record in enumerate(records):
        question = get_question(record)
        record_index = get_index(record, index)
        prompt_text = build_prompt_text(question)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=config["inference"].get("enable_thinking", True),
        )
        rendered.append(
            {
                "dataset": get_dataset_name(record),
                "index": record_index,
                "question": question,
                "label": record.get("answer", record.get("final_answer", record.get("label"))),
                "input_text": full_text,
            }
        )
    return rendered


def write_resolved_config(config, run_dir):
    resolved_path = run_dir / "config.resolved.yaml"
    try:
        import yaml

        with open(resolved_path, "w", encoding="utf-8") as file:
            yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)
    except Exception:
        with open(resolved_path, "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)


def prepare_run_dirs(config):
    run_id = config["run_id"]
    root_dir = Path(config.get("output_root", "experiments"))
    run_dir = root_dir / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir)
    return run_dir


def build_sampling_params(config, max_new_tokens):
    return SamplingParams(
        n=1,
        top_k=config["inference"].get("top_k", 1),
        top_p=config["inference"].get("top_p", 1.0),
        min_p=config["inference"].get("min_p", 0.0),
        temperature=config["inference"]["temperature"],
        presence_penalty=config["inference"].get("presence_penalty", 0.0),
        seed=config["inference"]["seed"],
        skip_special_tokens=False,
        ignore_eos=False,
        max_tokens=int(max_new_tokens),
        stop=["<turn|>"],
    )


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_records(records, config, tokenizer, llm, generations_path, max_new_tokens, write_mode):
    rendered = render_inputs(records, config, tokenizer)
    batch_size = config["inference"].get("batch_size", 32)
    sampling_params = build_sampling_params(config, max_new_tokens)

    with open(generations_path, write_mode, encoding="utf-8") as output_file:
        for start in tqdm(range(0, len(rendered), batch_size), desc="Generating responses"):
            batch = rendered[start:start + batch_size]
            outputs = llm.generate([item["input_text"] for item in batch], sampling_params)

            for item, output in zip(batch, outputs):
                first_output = output.outputs[0]
                row = {
                    "run_id": config["run_id"],
                    "dataset": item["dataset"],
                    "index": item["index"],
                    "question": item["question"],
                    "label": item["label"],
                    "model_output": first_output.text,
                    "output_token_length": len(getattr(first_output, "token_ids", []) or []),
                    "input_text": item["input_text"],
                    "finish_reason": getattr(first_output, "finish_reason", None),
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_length_generations(generations_path):
    kept_rows = []
    length_rows = []

    with open(generations_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("finish_reason") == "length":
                length_rows.append(row)
            else:
                kept_rows.append(row)

    write_jsonl(generations_path, kept_rows)
    return kept_rows, length_rows


def generation_key(row):
    return (str(row.get("dataset", "unknown")), str(row.get("index")))


def dataset_key(record, fallback_index):
    return (get_dataset_name(record), str(get_index(record, fallback_index)))


def to_retry_record(record, fallback_index):
    return {
        "dataset": get_dataset_name(record),
        "index": get_index(record, fallback_index),
        "label": record.get("answer", record.get("final_answer", record.get("label"))),
        "question": get_question(record),
    }


def build_length_retry_dataset(records, length_rows):
    length_keys = []
    seen_length_keys = set()
    for row in length_rows:
        key = generation_key(row)
        if key in seen_length_keys:
            raise ValueError(f"Duplicate length generation key: {key}")
        seen_length_keys.add(key)
        length_keys.append(key)

    source_records_by_key = {}
    source_order = []
    for index, record in enumerate(records):
        key = dataset_key(record, index)
        if key in source_records_by_key:
            raise ValueError(f"Duplicate source dataset key: {key}")
        source_records_by_key[key] = to_retry_record(record, index)
        source_order.append(key)

    missing_keys = set(length_keys) - set(source_records_by_key)
    if missing_keys:
        examples = sorted(missing_keys)[:10]
        raise ValueError(
            "Length rows contain keys not found in the source dataset; "
            f"will not copy retry records from generations. Examples: {examples}"
        )

    length_key_set = set(length_keys)
    retry_records = []
    for key in source_order:
        if key not in length_key_set:
            continue
        retry_records.append(source_records_by_key[key])

    return retry_records


def get_limited_records(config):
    records = load_records(config["dataset"])
    limit = config["dataset"].get("limit")
    if limit is not None:
        records = records[: int(limit)]
    return records


def get_retry_dataset_path(config, run_dir):
    retry_config = config.get("length_retry", {})
    dataset_name = retry_config.get("dataset_name", "length_retry_benchmark")
    filename = dataset_name if dataset_name.endswith(".jsonl") else f"{dataset_name}.jsonl"
    return run_dir / filename


def apply_cli_overrides(config, args):
    if args.run_id:
        config["run_id"] = args.run_id
    if args.student:
        config.setdefault("models", {})["student"] = args.student
    if args.revision:
        config.setdefault("models", {})["revision"] = args.revision
    return config


def generate(config):
    run_dir = prepare_run_dirs(config)
    generations_path = run_dir / "generations.jsonl"
    retry_dataset_path = get_retry_dataset_path(config, run_dir)

    phase1_max_new_tokens = config["inference"].get("max_new_tokens", 4096)
    phase2_max_new_tokens = config.get("length_retry", {}).get("max_new_tokens", 38912)

    records = get_limited_records(config)
    tokenizer, llm = load_tokenizer_and_vllm(config)

    logging.info("Starting phase 1 with max_new_tokens=%s", phase1_max_new_tokens)
    generate_records(records, config, tokenizer, llm, generations_path, phase1_max_new_tokens, "w")
    logging.info("Phase 1 generations written to %s", generations_path)

    kept_rows, length_rows = split_length_generations(generations_path)
    logging.info(
        "Filtered phase 1 generations: kept=%d, length_retry=%d",
        len(kept_rows),
        len(length_rows),
    )

    retry_records = build_length_retry_dataset(records, length_rows)
    write_jsonl(retry_dataset_path, retry_records)
    logging.info("Length retry dataset written to %s", retry_dataset_path)

    if not retry_records:
        logging.info("No length rows found; skipping phase 2")
        return

    retry_config = copy.deepcopy(config)
    retry_config["dataset"]["name"] = retry_config.get("length_retry", {}).get(
        "dataset_name", "length_retry_benchmark"
    )
    retry_config["dataset"]["data_path"] = str(retry_dataset_path)
    retry_config["inference"]["max_new_tokens"] = phase2_max_new_tokens

    logging.info("Starting phase 2 with max_new_tokens=%s", phase2_max_new_tokens)
    generate_records(retry_records, retry_config, tokenizer, llm, generations_path, phase2_max_new_tokens, "a")
    logging.info("Final generations written to %s", generations_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to the json config file")
    parser.add_argument("--run-id", type=str, help="benchmark run id")
    parser.add_argument("--student", type=str, help="student model path")
    parser.add_argument("--revision", type=str, help="model/tokenizer revision")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    config = apply_cli_overrides(config, args)
    generate(config)


if __name__ == "__main__":
    main()
