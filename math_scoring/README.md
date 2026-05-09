python math_scoring\compare_reason_diffs.py `
  --v1 math_scoring\qwen3_0.6b_base_bespoke_concise_pv1_predictions_matcher_v1.jsonl `
  --v2 math_scoring\qwen3_0.6b_base_bespoke_concise_pv1_predictions_matcher_v2.jsonl `
  --output math_scoring\reason_diffs_v1_vs_v2.jsonl

python export_no_match_answers.py  --input_path math_scoring\qwen3_0.6b_base_bespoke_concise_pv1_predictions_matcher_v1.jsonl   --output_path experiments\TN01_base_inference\runs\tn01_qwen3_0.6b_base_bespoke_concise_pv1\no_match_samples.jsonl   

python -m unittest tests.test_math_answer_matcher                               

python output_analysis.py `        
>>   --input experiments\TN01_base_inference\runs\tn01_qwen3_0.6b_base_bespoke_concise_pv1\generations.jsonl `                                                        
>>   --output experiments\TN01_base_inference\runs\tn01_qwen3_0.6b_base_bespoke_concise_pv1\predictions.jsonl