# iter024: (b) top-k 集中化 local bench A/B — 构造三个 delta 变体, Q_HepG2 单 context bench
# 变体: none(对照=v03/gen2 λ2.5 原样) / shrink15(delta<分位阈收缩到15%) / hardzero(delta<阈置0)
# 阈值按 per-target 内 |Δ| 分位: rank 越靠后方向越不可信 → 收缩其幅度
import subprocess, sys, os, json
import pandas as pd
import numpy as np

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SRC = f"{ROOT}/data/interim/bizard/v03/gen2/agg_delta_Q_HepG2.parquet"
OUT = f"{ROOT}/tmp/v08_topk"
os.makedirs(OUT, exist_ok=True)

d = pd.read_parquet(SRC)
print("src:", d.shape)

# per-target 内 |delta| 分位阈 (q25): 集中最强的 75%, 收缩最弱 25%
def make_variant(d, mode, q=0.25):
    out = d.copy()
    thr = out.groupby("target_gene")["delta"].transform(lambda s: s.abs().quantile(q))
    weak = out["delta"].abs() < thr
    if mode == "shrink15":
        out.loc[weak, "delta"] = out.loc[weak, "delta"] * 0.15
    elif mode == "hardzero":
        out.loc[weak, "delta"] = 0.0
    return out

variants = {
    "none": d,
    "shrink15": make_variant(d, "shrink15"),
    "hardzero": make_variant(d, "hardzero"),
}
for name, v in variants.items():
    v.to_parquet(f"{OUT}/delta_{name}.parquet")
    nz = (v.delta != 0).mean()
    print(name, "nonzero frac:", round(nz, 4))

cmds = []
for name in variants:
    tag = f"bzv08_{name}"
    cmds.append([sys.executable, "src/models/bizard/bench_bird.py", "--tag", tag,
                 "--lam", "2.5", "--contexts", "Q_HepG2",
                 "--delta-dir", OUT, "--delta-pattern", f"delta_{name}.parquet"])
print(json.dumps([c[2:] for c in cmds], indent=1))
for c in cmds:
    print("RUN", " ".join(c), flush=True)
    subprocess.run(c, check=True, cwd=ROOT)
print("ALL DONE")
