# math-reasoning-slm

## 1 - Infer training set

- In window:

```powershell
.\run_omni_infer.ps1
```

- In server:

```bash
bash run_omni_infer.sh
```

## 2 - Rule-based evaluate

- Single run:

```bash
python eval/rule_eval.py --input "experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailmethod_pv1/generations.jsonl"

python eval/rule_eval.py --input "experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_nohint_pv1/generations.jsonl"
```

- Multiple run:

```powershell
Get-ChildItem experiments/TN01_base_inference/runs -Directory | Where-Object Name -like "*1.7b*" | ForEach-Object { python eval/rule_eval.py --input "$($_.FullName)/generations.jsonl" }

Get-ChildItem -Path experiments/TN01_base_inference/runs -Recurse -Filter generations.jsonl | ForEach-Object { python eval/rule_eval.py --input $_.FullName }
```


## 3 - Analysis

### 3.1 - Negative analysis

"""
python eval/negative_analysis.py --prediction experiments/TN01_base_inference/runs/tn01_qwen3_0.6b_base_omni_le2000_detailmethod_pv1/prediction.jsonl

python eval/negative_analysis.py --prediction experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailmethod_pv1/prediction.jsonl
"""

### 3.2 - Positive analysis

"""
python eval/positive_analysis.py --prediction experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailmethod_pv1/prediction.jsonl

python eval/positive_analysis.py --prediction experiments/TN01_base_inference/runs/tn01_qwen3_0.6b_base_omni_le2000_detailmethod_pv1/prediction.jsonl
"""

## 4 - LLM evaluate

### 4.1 - Export suspect false negative (for LLM judge)
"""
python eval/export_suspect_false_negative.py --prediction experiments/TN01_base_inference/runs/tn01_qwen3_0.6b_base_omni_le2000_detailmethod_pv1/prediction.jsonl

python eval/export_suspect_false_negative.py --prediction experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailmethod_pv1/prediction.jsonl

"""


### 4.2 - LLM judge

- Single run:

```bash
python eval/llm_eval.py --false-negative experiments/TN01_base_inference/runs/tn01_qwen3_0.6b_base_omni_le2000_detailmethod_pv1/_false_negative_index.txt

python eval/llm_eval.py --false-negative experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailmethod_pv1/_false_negative_index.txt
```


## Random split to create subset dataset with 12k sample

If threshold-column is None and max-threshold is None: no filter

```bash
python 01_filter_dataset_hint_subset_12k.py --split le2000 --sample-size 12000 --threshold-column total_token --max-threshold 1950
```

## Verify unique and overlap index

```bash
python 02_check_index_overlap.py data/train/omni_hints_gt2000_sub12k.parquet data/train/omni_dataset_gt2000_sub12k.parquet
python 02_check_index_overlap.py data/train/omni_hints_le2000_sub12k.parquet data/train/omni_dataset_le2000_sub12k.parquet
```

## Create subset hint for check valid

```bash
python 03_export_hints_check_valid.py
```

Nếu chỉ tạo cho le2000

```bash
python 03_export_hints_check_valid.py --split le2000
```

Nếu muốn đổi folder output hoặc chunk size:

```bash
python 03_export_hints_check_valid.py --output-root data/hints_check_valid --chunk-size 1000
```

## Count multibox and filter final dataset (valid hint, no multi-box)
python 04_count_multi_box_solutions.py
python 05_filter_invalid_multibox_sample.py 


## Check index overlap after filter
python 02_check_index_overlap.py data/train/omni_dataset_gt2000_valid_no_multibox.parquet data/train/omni_hints_gt2000_valid_no_multibox.parquet
python 02_check_index_overlap.py data/train/omni_dataset_le2000_valid_no_multibox.parquet data/train/omni_hints_le2000_valid_no_multibox.parquet


## Filter subset10k after hint valid check
python 06_filter_dataset_hint_subset_10k.py


## Check index overlap after filter
python 02_check_index_overlap.py data/train/omni_hints_gt2000.parquet data/train/omni_dataset_gt2000.parquet
python 02_check_index_overlap.py data/train/omni_hints_le2000.parquet data/train/omni_dataset_le2000.parquet


## Create dataset for finetune

```bash
python sft_prepare.py
```



