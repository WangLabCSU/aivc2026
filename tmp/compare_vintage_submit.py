#!/usr/bin/env python
"""Decisive comparison: 顶层 v03 agg_delta_submit (Oct 4 vintage) vs gen2/ (Oct 6)."""
import numpy as np, pandas as pd
from pathlib import Path
V03 = Path("data/interim/bizard/v03")
for cx in ["A", "B", "C"]:
    try:
        old = pd.read_parquet(V03 / f"agg_delta_submit_{cx}.parquet")
        new = pd.read_parquet(V03 / "gen2" / f"agg_delta_submit_{cx}.parquet")
    except Exception as e:
        print(f"submit_{cx}: LOAD FAIL {type(e).__name__}: {e}")
        continue
    same_grid = (len(old) == len(new)
                 and (old["target_gene"].to_numpy() == new["target_gene"].to_numpy()).all()
                 and (old["gene"].to_numpy() == new["gene"].to_numpy()).all())
    d_old, d_new = old["delta"].to_numpy(float), new["delta"].to_numpy(float)
    diff = d_new - d_old
    sf = (np.sign(d_old) * np.sign(d_new) < 0) & (np.abs(d_old) > 1e-6) & (np.abs(d_new) > 1e-6)
    print(f"submit_{cx}: rows={len(old)} grid_same={same_grid} "
          f"corr={np.corrcoef(d_old, d_new)[0,1]:.4f} "
          f"mean|diff|={np.abs(diff).mean():.4f} max|diff|={np.abs(diff).max():.4f} "
          f"signflip={sf.mean():.4f} old(mean={d_old.mean():.4f} abs={np.abs(d_old).mean():.4f}) "
          f"new(mean={d_new.mean():.4f} abs={np.abs(d_new).mean():.4f})")
