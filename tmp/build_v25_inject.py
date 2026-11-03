#!/usr/bin/env python
"""iter043-15 轴21 臂 — v0.9 签名 (HY12) + vcc2025 16官方靶 Δ 注入 → 新签名 parquet.

注入规则: OFF16 官方靶整靶替换为 vcc2025 伪批量 logFC (真值同分布第三票),
其余靶 = v0.9 原值逐位。两源 (gwps_calHY12 / hct116_all_calHY12) 同步注入,
下游 make_predictions 等权聚合后 16 靶处 Δ = vcc2025 值 (非两源均值)。
"""
import numpy as np
import pandas as pd

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"

OFF16 = ['AKT2', 'HSBP1', 'MED13', 'MTA1', 'RNF2', 'SHPRH', 'SIN3B', 'SMARCA5',
         'STAT6', 'TARBP2', 'TRAPPC6A', 'TWF2', 'ZNF714', 'ADNP', 'ZFP62', 'BRPF1']

v25 = pd.read_parquet(f"{SIG}/vcc2025_train.parquet")
v25 = v25[v25.target_gene.isin(OFF16)].copy()
v25["gene"] = v25["gene"].astype(str)
v25 = v25.drop_duplicates(["target_gene", "gene"])
print("injection rows:", len(v25), "targets:", v25.target_gene.nunique(), flush=True)

for src in ["gwps_calHY12", "hct116_all_calHY12"]:
    df = pd.read_parquet(f"{SIG}/{src}.parquet")
    n0 = len(df)
    df = df[~df.target_gene.isin(OFF16)]
    inj = v25[["target_gene", "gene", "logfc"]].rename(columns={"logfc": "logfc"})
    df = pd.concat([df, inj], ignore_index=True)
    assert not df.duplicated(["target_gene", "gene"]).any(), f"dup keys in {src}!"
    out = f"{SIG}/{src}v25.parquet"
    df.to_parquet(out, index=False)
    print(f"{src}: {n0} -> {len(df)} rows, wrote {out}", flush=True)
print("DONE")
