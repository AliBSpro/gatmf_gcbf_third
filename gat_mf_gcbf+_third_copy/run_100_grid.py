# -*- coding: utf-8 -*-
import itertools, subprocess, sys

alphas  = [0.5, 1.0, 1.5, 2.0, 2.5]          # 5
gammas  = [0.005, 0.01, 0.02, 0.03, 0.05]    # 5
etas    = [0.3, 1.0, 3.0, 5.0]               # 4
rho     = 200
T       = 16
base_seed = 2000

i = 0
for a, g, e in itertools.product(alphas, gammas, etas):
    tag = f"a{a}_g{g}_e{e}"
    seed = base_seed + i
    cmd = [
        sys.executable, "trainer.py",
        "--alpha", str(a), "--gamma_margin", str(g), "--eta_ctrl", str(e),
        "--rho", str(rho), "--horizon_T", str(T),
        "--seed", str(seed), "--run_tag", tag,
        "--total_steps", "30000", "--save_every", "3000",
        "--n_agents", "8"
    ]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True)
    i += 1
