import json
import argparse
import logging
import os
from jinja2 import Environment, BaseLoader, FileSystemLoader
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer,SFTConfig


def formatting_func(examples):
    env = Environment(loader=BaseLoader())
    try:
        message = {"content": examples["instruction"],"output":examples["output"]}
        messages = [
            {"role": "user", "content": examples["instruction"]},
            {"role": "assistant", "content": examples["output"]},
        ]
        full_text = template.render(
            messages=messages,
            message=message,
            add_generation_prompt=False,
            add_output=True,
            tools=[],
        )
        return full_text
    except Exception as e:
        logging.warning(f"Error processing sample: {str(e)}")
        return ""


def train(config):
    dataset = load_dataset("json", data_files=config["dataset"]["labeled_path"])
    
    student_tokenizer = AutoTokenizer.from_pretrained(
        config["models"]["student"], 
        trust_remote_code=True
    )
    student_model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        trust_remote_code=True
    )

    global template
    full_path = config["dataset"]["template"]
    template_dir = os.path.dirname(full_path)
    template_file = os.path.basename(full_path)
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(template_file)
    training_arguments = SFTConfig(**config["training"])

    dataset = dataset.shuffle(seed=config["dataset"]["seed"])
    trainer = SFTTrainer(
        model=student_model,
        processing_class=student_tokenizer,
        args=training_arguments,
        train_dataset=dataset["train"],
        formatting_func=formatting_func
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
