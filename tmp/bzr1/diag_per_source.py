#!/usr/bin/env python
"""per-source direct-transfer 基线诊断 (bench-100 hold vs 全池参考), bzr1 v1 FAIL 追因."""
import time, numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
CTX = ["gwps", "hct116_all", "hepg2_all", "jurkat_all", "rpe1_all", "k562_ess_all"]
t0 = time.time()

genes, perts = set(), set()
for n in CTX:
    df = pd.read_parquet(f"{SIG}/{n}.parquet", columns=["target_gene", "gene"])
    genes |= set(pd.unique(df["gene"].astype(str)))
    perts |= set(pd.unique(df["target_gene"].astype(str)))
    del df
vocab_g = pd.Index(sorted(genes)); vocab_p = pd.Index(sorted(perts))
NG = len(vocab_g)
print(f"vocab: g={NG} p={len(vocab_p)} t={time.time()-t0:.0f}s", flush=True)

bench = pd.read_csv(f"{ROOT}/outputs/metrics/local_bench/dualdeep_lam70mf14_Q/Q_HepG2_eval/results.csv")
hold_perts = set(bench["perturbation"].astype(str))
hold_codes = np.where(vocab_p.isin(hold_perts))[0]

def load_keys(name):
    df = pd.read_parquet(f"{SIG}/{name}.parquet", columns=["target_gene", "gene", "logfc"])
    cp = vocab_p.get_indexer(df["target_gene"].astype(str).values)
    cg = vocab_g.get_indexer(df["gene"].astype(str).values)
    keys = (cp.astype(np.int64) * NG + cg.astype(np.int64))
    lf = df["logfc"].values.astype(np.float32)
    del df
    keys_u, inv = np.unique(keys, return_inverse=True)
    # 对齐修复 (iter030-修正3): np.unique 排序键后, 值必须经 inv 重排跟随, 无条件执行
    cnt = np.bincount(inv, minlength=len(keys_u)).astype(np.float64)
    s = np.bincount(inv, weights=lf.astype(np.float64), minlength=len(keys_u))
    lf = (s / cnt).astype(np.float32)
    del keys, inv
    return keys_u, lf

data = {}
for n in CTX:
    data[n] = load_keys(n)
    print(f"loaded {n} t={time.time()-t0:.0f}s", flush=True)

def psp(a, b):
    if a.std() == 0 or b.std() == 0: return np.nan
    return spearmanr(a, b).statistic

print("\n=== per-source baselines vs hepg2_all ===", flush=True)
for a in [c for c in CTX if c != "hepg2_all"]:
    ka, xa = data[a]; kb, xb = data["hepg2_all"]
    shared, ia, ib = np.intersect1d(ka, kb, assume_unique=True, return_indices=True)
    x = xa[ia]; y = xb[ib]
    cpsh = (shared // NG)
    hold = np.isin(cpsh, hold_codes); tr = ~hold
    out = [f"{a}->hepg2:"]
    for tag, m in [("HOLD", hold), ("POOL", tr)]:
        if m.sum() < 100: continue
        xs, ys = x[m], y[m]
        corr = float(np.corrcoef(xs, ys)[0, 1])
        sp = float(np.median([psp(xs[cpsh[m] == c], ys[cpsh[m] == c])
                              for c in np.unique(cpsh[m])
                              if (cpsh[m] == c).sum() >= 30]))
        mae = float(np.mean(np.abs(xs - ys))); mae_z = float(np.mean(np.abs(ys)))
        # 最小二乘最优标量收缩
        al = float(np.cov(xs, ys)[0, 1] / np.var(xs))
        mae_al = float(np.mean(np.abs(al * xs - ys)))
        out.append(f"{tag} n={int(m.sum()):,} corr={corr:.4f} spMed={sp:.4f} mae={mae:.4f} (zero {mae_z:.4f}, a*={al:.3f} mae={mae_al:.4f})")
    print(" | ".join(out), flush=True)
print(f"done t={time.time()-t0:.0f}s", flush=True)
