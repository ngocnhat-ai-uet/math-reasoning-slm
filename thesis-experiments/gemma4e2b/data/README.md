data/
    processed/
        problems.parquet
        hints.parquet
    labels/
        tn01_bespoke_hint_classification_v1.parquet
    splits/
        tn02_m0_seed9.parquet
        tn02_m4_seed9.parquet
        tn03_c3_seed9.parquet
    preferences/
        dpo_self_wrong_v1.parquet
        dpo_external_wrong_pv1.parquet


#  data/processed

bespoke.parquet
- question_idx
- question
- thought
- solution
- final_answer
- token_length => total_token
- train_token (coming soon)

hints.parquent
- question_idx
- concise_hint
- detailed_method_hint
- detailed_scaffold_hint
- model
- hint_valid

# data/labels

Gồm static label (TN01) và dynamic label (TN04)

Schema:
- question_idx
- nohint_correct
- concise_correct
- detail_method_correct
- detail_scaffold_correct
- pattern
- label (A/B1/B2/C1/C2/ANOMALY)

# data/splits/*.parquet

Chỉ lưu question_idx để thể hiện: config training này dùng những câu hỏi nào. Dữ liệu sẽ được lấy ở data/processed/bespoke.dataset theo question_idx.

Schema
- question_idx
- difficulty_label (A/B1/B2/C1/C2)
- group (easy/medium/hard)
- stage (early/middle/late)
- seed

# data/preferences

Lưu trữ các cặp chosen/rejected cho DPO

Schema
- question_idx
- chosen
- rejected
- rejected_source (self_generated/external_llm)
- rejected_run_id
- difficulty_label