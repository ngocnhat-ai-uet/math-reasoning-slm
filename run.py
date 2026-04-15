import os
from datetime import datetime

timestamp = datetime.now().strftime("%m%d_%H%M")
log_dir = f"logs/aime2026/agent_SV_gpt5nano_{timestamp}"
# log_dir = f"logs/aime2026/direct_gpt5nano_{timestamp}"

cmd = (
    "python aime.py "
    "--mode agent "
    f"--log_dir {log_dir} "
    "--dataset_name MathArena/aime_2026 "
    "--idx \"[15, 17, 29, 30]\" "
    "--max_runs 1 "
    "--max_pass 2 "
    "--max_fail 5"
)

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
