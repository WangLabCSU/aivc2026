#!/usr/bin/env python
"""iter043-15 轴21 臂: v0.9 签名 (gwps_calHY12 + hct116_all_calHY12) + vcc2025 16官方靶 Δ 注入.

注入规则: 官方16靶处整靶 Δ ← vcc2025 伪批量 logFC (面板过滤后, 不衰减 — λ_trans=1.0 起步,
真值同分布源不需要跨源衰减; 衰减量交给 λ_real 全局系数管)。其余靶不动 (= v0.9 逐位)。
输出: {gwps,hct116_all}_calHY12v25.parquet (两源同值注入, 保持两源结构不变).
"""
import numpy as np
import pandas as pd

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"

OFF16 = ['AKT2','HSBP1','MED13','MTA1','RNF2','SHPRH','SIN3B','SMARCA5','STAT6',
         'TARBP2','TRAPPC6A','TWF2','ZNF714','ADNP','ZFP62','BRPF1']

v25 = pd.read_parquet(f"{SIG}/vcc2025_train.parquet")
v25 = v25[v25.target_gene.isin(OFF16)]
print("injection rows:", len(v25), "targets:", v25.target_gene.nunique())

for src in ["gwps_calHY12", "hct116_all_calHY12"]:
    df = pd.read_parquet(f"{SIG}/{src}.parquet")
    n0 = len(df)
    # 删16靶旧行, 换 vcc2025 Δ
    df = df[~df.target_gene.isin(OFF16)]
    inj = v25.rename(columns={})[["target_gene", "gene", "logfc"]].copy()
    inj["gene"] = inj["gene"].astype(str)
    df = pd.concat([df, inj], ignore_index=True)
    # 键去重 (v25 每 (t,g) 唯一, 聚合源可能也有这些键 — 已删16靶所以无碰撞)
    assert not df.duplicated(["target_gene", "gene"]).any(), "dup keys!"
    out = f"{SIG}/{src}_calHY12v25.parquet"
    df.to_parquet(f"{SIG}/{src}_calHY12v25.parquet", index=False)
    print(f"{src}: {n0} -> {len(df)} rows, wrote {out}")
print("DONE")
