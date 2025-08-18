# bench.py
import time, torch
from trainer import Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"
T = Trainer(device=device, n_agents=8)

warmup, measure = 100, 400   # 预热100步，测400步
print("Warming up...")
T.train(total_steps=warmup, shield=False)

print("Measuring...")
t1 = time.time()
T.train(total_steps=measure, shield=False)
t2 = time.time()

sps = measure / (t2 - t1)
total_steps = 3000  # 你计划正式训练的步数
threshold = total_steps / 3600.0

print(f"Measured steps/s: {sps:.3f}")
print(f"Threshold steps/s for {total_steps} steps in <=1h: {threshold:.3f}")
print("Under 1 hour? ", sps >= threshold)
