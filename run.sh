#!/usr/bin/env bash
set -e

# python cli.py --config configs/infer_0.6b_omni_le2000_nohint.json
# python cli.py --config configs/infer_0.6b_omni_le2000_concise.json
# python cli.py --config configs/infer_0.6b_omni_le2000_detailmethod.json
# python cli.py --config configs/infer_0.6b_omni_le2000_detailscaffold.json

# python cli.py --config configs/infer_1.7b_omni_gt2000_nohint.json
# python cli.py --config configs/infer_1.7b_omni_gt2000_concise.json
# python cli.py --config configs/infer_1.7b_omni_gt2000_detailmethod.json
# python cli.py --config configs/infer_1.7b_omni_gt2000_detailscaffold.json


# python cli.py --config configs/sft_0.6b_easy.json
# python cli.py --config configs/sft_0.6b_hard.json
# python cli.py --config configs/sft_0.6b_full.json

# python cli.py --config configs/sft_1.7b_easy.json
# python cli.py --config configs/sft_1.7b_hard.json
# python cli.py --config configs/sft_1.7b_full.json

# python 11_eval_benchmark.py --config configs/eval_benchmark.json --run-id benchmark_0.6b_sft_full --student ngocnhat/qwen3-0.6b-math-sft --revision full_lr1e-5_warmup0.1_decay0.05
# python 11_eval_benchmark.py --config configs/eval_benchmark.json --run-id benchmark_0.6b_sft_easy --student ngocnhat/qwen3-0.6b-math-sft --revision easy_lr1e-5_warmup0.1_decay0.05
# python 11_eval_benchmark.py --config configs/eval_benchmark.json --run-id benchmark_0.6b_sft_hard --student ngocnhat/qwen3-0.6b-math-sft --revision hard_lr1e-5_warmup0.05_decay0.05

# python 11_eval_benchmark.py --config configs/eval_benchmark.json --run-id benchmark_1.7b_sft_full --student ngocnhat/qwen3-1.7b-math-sft --revision full_lr1e-5_warmup0.1_decay0.05
# python 11_eval_benchmark.py --config configs/eval_benchmark.json --run-id benchmark_0.6b_sft_easy --student ngocnhat/qwen3-0.6b-math-sft --revision easy_lr1e-5_warmup0.1_decay0.05
# python 11_eval_benchmark.py --config configs/eval_benchmark.json --run-id benchmark_1.7b_sft_hard --student ngocnhat/qwen3-1.7b-math-sft --revision hard_lr1e-5_warmup0.05_decay0.05



python 12_eval_benchmark_greedy.py --run-id benchmark_0.6b_sft_easy --student ngocnhat/qwen3-0.6b-math-sft --revision easy_lr1e-5_warmup0.1_decay0.05
python 12_eval_benchmark_greedy.py --run-id benchmark_0.6b_sft_full --student ngocnhat/qwen3-0.6b-math-sft --revision full_lr1e-5_warmup0.1_decay0.05
python 12_eval_benchmark_greedy.py --run-id benchmark_0.6b_sft_hard --student ngocnhat/qwen3-0.6b-math-sft --revision hard_lr1e-5_warmup0.05_decay0.05

# python 12_eval_benchmark_greedy.py --run-id benchmark_1.7b_sft_easy --student ngocnhat/qwen3-1.7b-math-sft --revision easy_lr1e-5_warmup0.1_decay0.05
# python 12_eval_benchmark_greedy.py --run-id benchmark_1.7b_sft_full --student ngocnhat/qwen3-1.7b-math-sft --revision full_lr1e-5_warmup0.1_decay0.05
# python 12_eval_benchmark_greedy.py --run-id benchmark_1.7b_sft_hard --student ngocnhat/qwen3-1.7b-math-sft --revision hard_lr1e-5_warmup0.05_decay0.05

