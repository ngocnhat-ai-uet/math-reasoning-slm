import argparse
import json
import logging
import os
from pathlib import Path

import torch
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SYSTEM_PROMPT = r"You are a careful math solver. Think through the solution step by step before writing the final answer. Put the final answer inside \boxed{}."

# Input: dataset_path (jsonl/parquet)
# Output: list of dict
def load_records(dataset_config):
    dataset_path = dataset_config["data_path"]
    suffix = Path(dataset_path).suffix.lower()

    if suffix == ".json":
        with open(dataset_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list in {dataset_path}")
        return data

    if suffix == ".jsonl":
        records = []
        with open(dataset_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    if suffix == ".parquet":
        dataset = load_dataset("parquet", data_files=dataset_path)["train"]
        return [dict(item) for item in dataset]

    raise ValueError(f"Unsupported dataset format: {dataset_path}")


# Input: hint_path (jsonl/parquet)
# Output: dict
def load_hints(hint_path):
    if not hint_path:
        return {}

    suffix = Path(hint_path).suffix.lower()
    if suffix == ".parquet":
        rows = load_dataset("parquet", data_files=hint_path)["train"]
        records = [dict(item) for item in rows]
    elif suffix == ".json":
        with open(hint_path, "r", encoding="utf-8") as file:
            records = json.load(file)
    elif suffix == ".jsonl":
        records = []
        with open(hint_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported hints format: {hint_path}")

    missing_index = [i for i, item in enumerate(records) if "index" not in item]
    if missing_index:
        first_missing = missing_index[0]
        raise ValueError(f"Hint record at position {first_missing} has no required 'index' field")

    return {str(item["index"]): item for item in records}


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


def resolve_hint(record, hints_by_id, condition):
    if condition == "nohint":
        return None

    hint_fields = {
        "concise": "concise_hint",
        "detailmethod": "detailed_method_hint",
        "detailscaffold": "detailed_scaffold_hint",
    }
    hint_field = hint_fields.get(condition)
    if hint_field is None:
        raise ValueError(f"Unsupported prompt condition: {condition}")

    record_index = str(get_index(record, ""))
    hint_record = hints_by_id.get(record_index, {})
    hint = hint_record.get(hint_field)
    if hint is None:
        return None
    return str(hint)


def build_prompt_text(question, hint, prompt_config):
    if not hint:
        return question
    hint_format = prompt_config.get("hint_format", "{question}\n\nHint: {hint}")
    return hint_format.format(question=question, hint=hint)


def load_tokenizer_and_vllm(config, eos_token=None):
    model_path = config["models"].get("student") or config["models"].get("model")
    if not model_path:
        raise ValueError("Config must define models.student or models.model")

    logging.info(f"Loading ckpt and tokenizer: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if eos_token:
        eos_token_id = tokenizer.convert_tokens_to_ids(eos_token)
        logging.info(f"eos_token {eos_token} from user input")
    elif hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id:
        logging.info(f"Initial eos_token_id {tokenizer.eos_token_id} from tokenizer")
        eos_token_id = tokenizer.eos_token_id
        eos_token = tokenizer.convert_ids_to_tokens(eos_token_id)
    else:
        raise ValueError("No available eos_token or eos_token_id.")

    try:
        tokenizer.eos_token = eos_token
        tokenizer.eos_token_id = eos_token_id
        tokenizer.pad_token = eos_token
        tokenizer.pad_token_id = eos_token_id
    except Exception:
        logging.info("[WARNING] Cannot set tokenizer.eos_token")

    logging.info(f"tokenizer's eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    logging.info(f"tokenizer's eos_token_id: {tokenizer.eos_token_id}, pad_token_id: {tokenizer.pad_token_id}")

    num_gpus = torch.cuda.device_count()
    attention_backend = config["inference"].get("attention_backend")
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
        logging.info(f"Using vLLM attention backend: {attention_backend}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        enable_chunked_prefill=config["inference"]["enable_chunked_prefill"],
        gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
        trust_remote_code=config["inference"]["trust_remote_code"],
        dtype=torch.bfloat16,
        enforce_eager=config["inference"]["enforce_eager"],
        max_model_len=config["inference"]["max_model_len"],
    )
    logging.info("vLLM model loaded successfully")
    return tokenizer, llm


def render_inputs(records, config):
    template_path = config["dataset"]["template"]
    template_dir = os.path.dirname(template_path)
    template_file = os.path.basename(template_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)

    prompt_config = config.get("prompt", {})
    condition = prompt_config.get("condition", "nohint") # return nohint/concise/detail...
    hints_by_id = load_hints(config["dataset"].get("hint_path")) # dict: index : hint record

    rendered = []
    for index, record in enumerate(records):
        question = get_question(record)
        record_index = get_index(record, index)
        hint = resolve_hint(record, hints_by_id, condition)
        prompt_text = build_prompt_text(question, hint, prompt_config)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ]
        full_text = template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=config["inference"].get("enable_thinking", True),
        )
        rendered.append(
            {
                "index": record_index,
                "question": question,
                "hint": hint,
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
    experiment_id = config.get("experiment_id", "TN01_base_inference")
    run_id = config["run_id"]
    root_dir = Path(config.get("output_root", "experiments"))
    run_dir = root_dir / experiment_id / "runs" / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, run_dir)
    return run_dir


def generate(config):
    records = load_records(config["dataset"])
    limit = config["dataset"].get("limit")
    if limit is not None:
        records = records[: int(limit)]

    rendered = render_inputs(records, config)
    _, llm = load_tokenizer_and_vllm(config)
    run_dir = prepare_run_dirs(config)
    generations_path = run_dir / "generations.jsonl"

    batch_size = config["inference"].get("batch_size", 32)
    sampling_params = SamplingParams(
        n=1,
        top_k=config["inference"].get("top_k", 1),
        top_p=config["inference"].get("top_p", 1.0),
        min_p=config["inference"].get("min_p", 0.0),
        temperature=config["inference"]["temperature"],
        presence_penalty=config["inference"].get("presence_penalty", 0.0),
        seed=config["inference"]["seed"],
        skip_special_tokens=False,
        ignore_eos=False,
        max_tokens=config["inference"]["max_new_tokens"],
        stop=["<turn|>"],
    )

    with open(generations_path, "w", encoding="utf-8") as output_file:
        for start in tqdm(range(0, len(rendered), batch_size), desc="Generating responses"):
            batch = rendered[start:start + batch_size]
            outputs = llm.generate([item["input_text"] for item in batch], sampling_params)

            for item, output in zip(batch, outputs):
                first_output = output.outputs[0]
                row = {
                    "run_id": config["run_id"],
                    "index": item["index"],
                    "question": item["question"],
                    "hint": item["hint"],
                    "label": item["label"],
                    "model_output": first_output.text,
                    "output_token_length": len(getattr(first_output, "token_ids", []) or []),
                    "finish_reason": getattr(first_output, "finish_reason", None),
                }
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    logging.info(f"Generations written to {generations_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to the json config file")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    generate(config)


if __name__ == "__main__":
    main()
