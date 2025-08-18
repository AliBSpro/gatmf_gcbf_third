# -*- coding: utf-8 -*-
import csv
from pathlib import Path

model_dir = Path("model")
out_csv = model_dir / "summary_all_models.csv"

rows = []
for f in sorted(model_dir.glob("run_*_metrics.csv")):
    run_id = f.stem.replace("_metrics","")
    last = None
    with open(f, "r") as r:
        reader = csv.reader(r)
        header = next(reader)
        for row in reader:
            last = row
    if last is None:
        continue
    rows.append([run_id] + last)

with open(out_csv, "w", newline="") as w:
    writer = csv.writer(w)
    writer.writerow([
        "run_id","step","Lcbf","Lctrl","lam",
        "eval_return","eval_unsafe","eval_success_rate",
        "alpha","gamma_margin","eta_ctrl","rho","horizon_T","seed","tag"
    ])
    writer.writerows(rows)

print(f"汇总完成 -> {out_csv}")
