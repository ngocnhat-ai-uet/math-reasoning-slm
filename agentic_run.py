# import os
# from datetime import datetime

# timestamp = datetime.now().strftime("%m%d_%H%M")
# # log_dir = f"logs/aime2026/agentSV_gemma4e2b_{timestamp}"
# log_dir = f"logs/aime2026/direct_gemma4e4b_{timestamp}"

# # cmd = f"""
# # python aime_ollama.py \
# #   --mode direct \
# #   --log_dir "{log_dir}" \
# #   --dataset_name "MathArena/aime_2026" \
# #   --idx "[30]" \
# #   --max_runs 1 \
# #   --max_pass 5 \
# #   --max_fail 10 \
# #   --check_complete on
# # """


# # cmd = f"""
# # python aime_ollama.py \
# #   --mode direct \
# #   --log_dir "{log_dir}" \
# #   --dataset_name "MathArena/aime_2026" \
# #   --idx "[7,9,10,11,12,13,14,15,17,18,23,25,26,27,28,29,30]" \
# #   --max_runs 1 \
# #   --max_pass 2 \
# #   --max_fail 5 \
# #   --check_complete on
# # """

# # cmd = (
# #     "python aime.py "
# #     "--mode agent "
# #     f"--log_dir {log_dir} "
# #     "--dataset_name MathArena/aime_2026 "
# #     "--limit 1 "
# #     "--max_runs 10 "
# #     "--max_pass 2 "
# #     "--max_fail 4"
# # )

# os.system(cmd)