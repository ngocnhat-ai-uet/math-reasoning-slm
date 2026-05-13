import json
import argparse
import logging
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


def build_messages(example, default_system_prompt=None, include_assistant=True):
    system_prompt = example.get("system") or default_system_prompt
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": example["instruction"]})
    if include_assistant:
        messages.append({"role": "assistant", "content": example["output"]})

    return messages


def make_tokenize_func(tokenizer, max_length, default_system_prompt=None):
    def tokenize_func(example):
        try:
            prompt_text = tokenizer.apply_chat_template(
                build_messages(example, default_system_prompt, include_assistant=False),
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                build_messages(example, default_system_prompt, include_assistant=True),
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
            tokenized = tokenizer(
                full_text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_length,
            )
            labels = tokenized["input_ids"].copy()
            prompt_length = min(len(prompt_ids), len(labels))
            labels[:prompt_length] = [-100] * prompt_length
            tokenized["labels"] = labels
            return tokenized
        except Exception as e:
            logging.warning(f"Error processing sample: {str(e)}")
            return {"input_ids": [], "attention_mask": [], "labels": []}

    return tokenize_func


def train(config):
    dataset = load_dataset("json", data_files=config["dataset"]["labeled_path"])
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    if student_tokenizer.chat_template is None:
        raise ValueError("Student tokenizer has no chat_template; cannot use apply_chat_template.")

    system_prompt = config["dataset"].get("system_prompt")
    dataset_kwargs = config["training"].setdefault("dataset_kwargs", {})
    dataset_kwargs.setdefault("skip_prepare_dataset", True)
    training_arguments = SFTConfig(**config["training"])

    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    limit = config["dataset"].get("limit")
    if limit is not None:
        dataset["train"] = dataset["train"].select(range(min(limit, len(dataset["train"]))))

    train_dataset = dataset["train"].map(
        make_tokenize_func(student_tokenizer, training_arguments.max_length, system_prompt),
        remove_columns=dataset["train"].column_names,
    )
    trainer = SFTTrainer(
        model=student_model,
        processing_class=student_tokenizer,
        args=training_arguments,
        train_dataset=train_dataset,
    )
        
    trainer.train()
    trainer.save_model(config["training"]["output_dir"])
    student_tokenizer.save_pretrained(config["training"]["output_dir"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to the json config file')
    args = parser.parse_args()
    config = json.load(open(args.config))
    train(config)  


if __name__ == "__main__":
    main()
