# -*- coding: utf-8 -*-
import subprocess, sys

TOTAL = 100
BASE_SEED = 1000

for i in range(TOTAL):
    seed = BASE_SEED + i
    cmd = [
        sys.executable, "trainer.py",
        "--seed", str(seed),
        "--total_steps", "30000",
        "--save_every", "3000",
        "--run_tag", f"seed{seed}",
        "--n_agents", "8"
        # 可按需加 "--shield"
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
