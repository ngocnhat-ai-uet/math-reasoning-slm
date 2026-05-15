import argparse
import json
import logging
import os
import subprocess
import sys


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run_cmd(cmd):
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            universal_newlines=True,
        )

        error_detected = False
        error_keywords = [
            "ERROR",
            "Error",
            "error",
            "Unrecognized model",
            "failed",
            "exception",
            "Traceback",
        ]

        while True:
            line = process.stdout.readline()
            if not line:
                break
            logging.info(line.rstrip())
            if any(keyword.lower() in line.lower() for keyword in error_keywords):
                error_detected = True
                logging.error(f"Detected error in output: {line.strip()}")

        returncode = process.wait()
        if error_detected or returncode != 0:
            logging.error(f"Command failed (returncode={returncode}, errors detected)")
            return False
        return True

    except Exception as exc:
        logging.error(f"Unexpected error running command: {exc}")
        return False


def process(job_type, config_path):
    if not os.path.isabs(config_path):
        config_path = os.path.abspath(config_path)

    entrypoints = {
        "infer": "07_infer.py",
        "sft": "10_sft.py",
        "dpo": "dpo.py",
    }
    if job_type not in entrypoints:
        logging.error(f"Unknown job type: {job_type}. Expected one of: {', '.join(entrypoints)}")
        sys.exit(1)

    script_path = os.path.join(SCRIPT_DIR, entrypoints[job_type])
    cmd = f'"{sys.executable}" "{script_path}" --config "{config_path}"'
    logging.info(f"Running command: {cmd}")
    success = run_cmd(cmd)
    if not success:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to the json config file")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as file:
        config = json.load(file)
    process(config["job_type"], args.config)


if __name__ == "__main__":
    main()
