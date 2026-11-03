"""Training progress check: val_loss series + latest train rows, csv.DictReader only."""
import csv

P = ("/data/resource/wanglab/aivc/aivc2026/data/interim/dl_v06/run_v06/"
     "state_v06_gwps_rpe1/version_0/metrics.csv")

rows = list(csv.DictReader(open(P)))
val = [(int(r["epoch"]), int(r["step"]), float(r["val_loss"]))
       for r in rows if r.get("val_loss")]
tr = [(int(r["epoch"]), int(r["step"]), float(r["train_loss"]))
      for r in rows if r.get("train_loss")]
print("val series (last 8):")
for e, s, v in val[-8:]:
    print(f"  epoch {e} step {s} val {v:.4f}")
print("train (last 3):")
for e, s, v in tr[-3:]:
    print(f"  epoch {e} step {s} loss {v:.4f}")
best = min(val, key=lambda t: t[2])
print(f"best val: {best[2]:.4f} @ step {best[1]}")
