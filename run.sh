#!/usr/bin/env bash
set -e

python cli.py --config configs/infer_0.6b_omni_le2000_nohint.json
python cli.py --config configs/infer_0.6b_omni_le2000_concise.json
python cli.py --config configs/infer_0.6b_omni_le2000_detailmethod.json
python cli.py --config configs/infer_0.6b_omni_le2000_detailscaffold.json

python cli.py --config configs/infer_1.7b_omni_gt2000_nohint.json
python cli.py --config configs/infer_1.7b_omni_gt2000_concise.json
python cli.py --config configs/infer_1.7b_omni_gt2000_detailmethod.json
python cli.py --config configs/infer_1.7b_omni_gt2000_detailscaffold.json


python cli.py --config configs/sft_0.6b_easy.json
python cli.py --config configs/sft_0.6b_hard.json
python cli.py --config configs/sft_0.6b_full.json

python cli.py --config configs/sft_1.7b_easy.json
python cli.py --config configs/sft_1.7b_hard.json
python cli.py --config configs/sft_1.7b_full.json

python 11_eval_benchmark.py --config configs/eval_benchmark_0.6b_sft_full.json
python 11_eval_benchmark.py --config configs/eval_benchmark_0.6b_sft_easy.json
python 11_eval_benchmark.py --config configs/eval_benchmark_0.6b_sft_hard.json