#!/usr/bin/env python
"""决定性检验: load_keys 的 np.unique 排序是否撕裂了 keys↔lf 对齐 (iter030-修正3).

假设: 无重复时 dedup 分支被跳过, 但 keys_u 已被 np.unique 排序, lf 仍为文件序.
若文件自然顺序 ≠ key 升序, 则 x/y 值相对键被置换 → corr 归零而边际分布(故 MAE)不变.
判据: 修复对齐后 key 管线 corr 应回到 ~0.214 (string-merge 金标准).
"""
import time, numpy as np, pandas as pd

SIG = "/data/resource/wanglab/aivc/aivc2026/data/interim/signatures"
cols = ["target_gene", "gene", "logfc"]
t0 = time.time()

g = pd.read_parquet(f"{SIG}/gwps.parquet", columns=cols)
h = pd.read_parquet(f"{SIG}/hepg2_all.parquet", columns=cols)
print(f"loaded t={time.time()-t0:.0f}s", flush=True)

vocab_p = pd.Index(sorted(set(g["target_gene"].astype(str)) | set(h["target_gene"].astype(str))))
vocab_g = pd.Index(sorted(set(g["gene"].astype(str)) | set(h["gene"].astype(str))))
NG = len(vocab_g)
print(f"vocab g={NG} p={len(vocab_p)} t={time.time()-t0:.0f}s", flush=True)

def load(name, df):
    cp = vocab_p.get_indexer(df["target_gene"].astype(str).values).astype(np.int64)
    cg = vocab_g.get_indexer(df["gene"].astype(str).values).astype(np.int64)
    keys = cp * NG + cg
    lf = df["logfc"].values.astype(np.float32)
    mono = bool(np.all(keys[1:] >= keys[:-1]))
    keys_u, inv = np.unique(keys, return_inverse=True)
    n_dup = len(keys) - len(keys_u)
    # 修复对齐: 值跟随键的排序位置
    lf_al = np.empty_like(lf)
    lf_al[inv] = lf
    print(f"{name}: n={len(keys):,} file-order-monotonic={mono} dup-rows={n_dup:,} t={time.time()-t0:.0f}s", flush=True)
    return keys_u, lf, lf_al

kg, xg_file, xg_al = load("gwps", g)
kh, xh_file, xh_al = load("hepg2_all", h)
del g, h

shared, ia, ib = np.intersect1d(kg, kh, assume_unique=True, return_indices=True)
assert np.array_equal(np.searchsorted(kg, shared), ia), "searchsorted != ia"
assert np.array_equal(np.searchsorted(kh, shared), ib), "searchsorted != ib"
print(f"shared={len(shared):,}", flush=True)

for tag, xa, xb in [("BUGGY(文件序值×排序键)", xg_file[ia], xh_file[ib]),
                    ("FIXED(值随键排序)",    xg_al[ia],  xh_al[ib])]:
    corr = float(np.corrcoef(xa, xb)[0, 1])
    mae = float(np.mean(np.abs(xa - xb)))
    mae_z = float(np.mean(np.abs(xb)))
    print(f"{tag}: corr={corr:.4f} mae={mae:.4f} (zero {mae_z:.4f})", flush=True)

print("gold (string-merge): corr=0.2142 mae=0.0658 zero=0.0577", flush=True)
print(f"done t={time.time()-t0:.0f}s", flush=True)
