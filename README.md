# math-reasoning-slm

## Infer training set 
- In window: 
.\run_omni_infer.ps1
- In server
bash run_omni_infer.sh


## rule-based eval

- Single run:

"""
python eval/rule_eval.py --input "experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_concise_pv1/generations.jsonl"

python eval/rule_eval.py --input "experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailmethod_pv1/generations.jsonl"

python eval/rule_eval.py --input "experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailscaffold_pv1/generations.jsonl"

python eval/rule_eval.py --input "experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_nohint_pv1/generations.jsonl"
"""

- Multiple run:

"""
Get-ChildItem experiments/TN01_base_inference/runs -Directory | Where-Object Name -like "*1.7b*" | ForEach-Object { python eval/rule_eval.py --input "$($_.FullName)/generations.jsonl" }

Get-ChildItem -Path experiments/TN01_base_inference/runs -Recurse -Filter generations.jsonl | ForEach-Object { python eval/rule_eval.py --input $_.FullName }
"""


## llm eval

- Single run:

"""
python eval/llm_eval.py --prediction experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailscaffold_pv1/prediction.jsonl --false-negative experiments/TN01_base_inference/runs/tn01_qwen3_1.7b_base_omni_gt2000_detailscaffold_pv1/llm_false_negative_cases.json
"""

- Multiple runs:

"""
Get-ChildItem experiments/TN01_base_inference/runs -Directory | ForEach-Object {
  $prediction = Join-Path $_.FullName "prediction.jsonl"
  $falseNegative = Join-Path $_.FullName "llm_false_negative_cases.json"

  if ((Test-Path $prediction) -and (Test-Path $falseNegative)) {
    python eval/llm_eval.py --prediction $prediction --false-negative $falseNegative
  }
}
"""

## Random split to create subset dataset with 10k sample

"""

python filter_dataset_hint_subset10k.py

python filter_generations_subset10k.py

"""
