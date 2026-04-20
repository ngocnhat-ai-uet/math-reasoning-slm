import os
from datetime import datetime

timestamp = datetime.now().strftime("%m%d_%H%M")
log_dir = f"logs/aime2026/agentSV_gemma4e2b_{timestamp}"
# log_dir = f"logs/aime2026/direct_gemma4e2b_{timestamp}"
# log_dir = f"logs/aime2026/agentSV_phi3_{timestamp}"
# log_dir = f"logs/aime2026/direct_phi3_{timestamp}"
# log_dir = f"logs/aime2026/agent_SV_gpt5nano_{timestamp}"
# log_dir = f"logs/aime2026/direct_gpt5nano_{timestamp}"

cmd = (
    "python aime_ollama.py "
    "--mode agent "
    f"--log_dir {log_dir} "
    "--dataset_name MathArena/aime_2026 "
    "--idx \"[3, 6, 7, 9, 10, 11, 12, 13, 14, 15]\" "
    "--max_runs 1 "
    "--max_pass 2 "
    "--max_fail 5"
)

# cmd = (
#     "python aime_ollama.py "
#     "--mode direct "
#     f"--log_dir {log_dir} "
#     "--dataset_name MathArena/aime_2026 "
#     "--idx \"[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]\" "
#     "--max_runs 2 "
#     "--max_pass 2 "
#     "--max_fail 5"
# )

# cmd = (
#     "python aime.py "
#     "--mode agent "
#     f"--log_dir {log_dir} "
#     "--dataset_name MathArena/aime_2026 "
#     "--idx \"[15, 17, 29, 30]\" "
#     "--max_runs 1 "
#     "--max_pass 2 "
#     "--max_fail 5"
# )

# cmd = (
#     "python aime.py "
#     "--mode agent "
#     f"--log_dir {log_dir} "
#     "--dataset_name MathArena/aime_2026 "
#     "--limit 1 "
#     "--max_runs 10 "
#     "--max_pass 2 "
#     "--max_fail 4"
# )

os.system(cmd)
