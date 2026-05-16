$ErrorActionPreference = "Stop"

python cli.py --config configs/infer_0.6b_omni_le2000_nohint.json
python cli.py --config configs/infer_0.6b_omni_le2000_concise.json
python cli.py --config configs/infer_0.6b_omni_le2000_detailmethod.json
python cli.py --config configs/infer_0.6b_omni_le2000_detailscaffold.json

python cli.py --config configs/infer_1.7b_omni_gt2000_nohint.json
python cli.py --config configs/infer_1.7b_omni_gt2000_concise.json
python cli.py --config configs/infer_1.7b_omni_gt2000_detailmethod.json
python cli.py --config configs/infer_1.7b_omni_gt2000_detailscaffold.json

python cli.py --config configs/sft_0.6b_easy.json
python cli.py --config configs/sft_0.6b_full.json
python cli.py --config configs/sft_0.6b_hard.json

python cli.py --config configs/sft_1.7b_easy.json
python cli.py --config configs/sft_1.7b_full.json
python cli.py --config configs/sft_1.7b_hard.json
