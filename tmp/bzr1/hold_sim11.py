#!/usr/bin/env python
"""sim11 — amp头v1 集成 probe (iter037-续4 预注册路径).

amp_hold_v1.parquet (5源 × hold-100 perts × genes, abs_sq/abs_hu) → 按 (p,g)
平均 → 6146 列映射 → 替换 HY8 的 |YH| top-8 括号幅度源 (sign 门 X*YH>0 不变).

聚合两档 (triple-proxy 律下另加一轴):
  agg2: src∈{gwps,hct116_all} 均值 — 与 sim5_cache X/YH 聚合完全一致
  agg5: 全 5 源均值 — 覆盖更大, 但与 X/YH 聚合不一致

候选 (KDE=8, 与 sim8c 完全同框架):
  refs : raw / nc / hy8 (须复现 sim8_table hy8 行 = 完整性自检)
  test : amp_sq_brYH     括号=|YH|, 幅度=model sq
         amp_hu_brYH     括号=|YH|, 幅度=model hu
         amp_sq_brM      括号=model, 幅度=model sq
         amp_hu_brM      括号=model, 幅度=model hu
         amp_brM_magYH   括号=model sq, 幅度=|YH| (只换选位)
         amp_max         括号=|YH|, 幅度=max(|YH|, model sq)

双框架读数 (triple-proxy 律: 任一单读数不迁移, 两框架同向才进 bench 链):
  sim5 框架: pds(rank-cosine) + fidpx + nmaepx   — 门: d_vs_A ≥ +0.045, dF ≥ -0.001
  sim8c 框架: n_pred/agree/pds/nmae/fid          — 对照 hy8 行 (pds 0.1642)
集成判定: 候选 sim5-pds > hy8 sim5-pds 且 sim8c-pds ≥ hy8 (不倒退) → bench 链.
"""
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = "/data/resource/wanglab/aivc/aivc2026"
LAM = 7.0
CAP = np.log2(14.0)
KDE = 8

t0 = time.time()
z = np.load(f"{ROOT}/tmp/bzr1/sim5_cache.npz", allow_pickle=True)
X, Y, YH = z["X"], z["Y"], z["YH"]  # (100, 6146)
n, G = X.shape
g5 = pd.Index(np.load(f"{ROOT}/tmp/bzr1/sim5_genes.npy", allow_pickle=True).astype(str))

# ---- NTC 统计 (sim8c 同款) ----
ntc_mean = np.load(f"{ROOT}/tmp/bzr1/ntc_mean_g6146.npy").astype(np.float64)
ntc_sd = np.load(f"{ROOT}/tmp/bzr1/ntc_sd_g6146.npy").astype(np.float64)
cv = ntc_sd / np.maximum(ntc_mean, 0.5)
theta = 2.5 * cv + 0.05

def R(v):
    return np.minimum(LAM * v, CAP)

def n_pred_sim(v):
    pl = R(v)
    p = 1.0 / (1.0 + np.exp(-(np.abs(pl) - theta) / (0.25 * theta + 1e-6)))
    return p.sum(axis=1)

# ---- amp_hold_v1 → (p,g) 聚合 → 6146 列 ----
amp = pd.read_parquet(f"{ROOT}/tmp/bzr1/amp_hold_v1.parquet")
print(f"amp rows={len(amp):,} srcs={sorted(amp['src'].unique())} t={time.time()-t0:.0f}s", flush=True)

hold_perts = sorted(amp["target_gene"].unique())
assert len(hold_perts) == 100, f"expect 100 perts, got {len(hold_perts)}"

def amp_matrix(df):
    agg = df.groupby(["target_gene", "gene"], sort=False)[["abs_sq", "abs_hu"]].mean().reset_index()
    A_sq = agg.pivot(index="target_gene", columns="gene", values="abs_sq").reindex(
        index=hold_perts, columns=g5).to_numpy(dtype=np.float32)  # NaN = 无模型覆盖
    A_hu = agg.pivot(index="target_gene", columns="gene", values="abs_hu").reindex(
        index=hold_perts, columns=g5).to_numpy(dtype=np.float32)
    return A_sq, A_hu

SRC5 = sorted(amp["src"].unique())
A2 = amp_matrix(amp[amp["src"].isin(["gwps", "hct116_all"])])
A5 = amp_matrix(amp)
print(f"coverage agg2={np.isfinite(A2[0]).mean():.3f} agg5={np.isfinite(A5[0]).mean():.3f} "
      f"t={time.time()-t0:.0f}s", flush=True)

# ---- 基础构造 (sim8c 逐行复刻) ----
mu = X.mean(0, keepdims=True)
H = (X - mu).astype(np.float32)
NC = np.where(X * H > 0, H, X).astype(np.float32)

aby = np.abs(YH)
th8 = np.partition(aby, -KDE, axis=1)[:, -KDE][:, None]
de = aby >= th8
sel = de & (X * YH > 0)
HY = NC.copy(); HY[sel] = np.sign(X[sel]) * aby[sel]

def build(mag, bracket):
    """bracket: 'yh' (|YH| top8) 或 'model' (|mag| top8); 幅度=mag (NaN→aby)."""
    M = np.where(np.isfinite(mag), mag, aby).astype(np.float32)
    if bracket == "yh":
        s = sel
    else:
        thM = np.partition(M, -KDE, axis=1)[:, -KDE][:, None]
        s = (M >= thM) & (X * YH > 0)
    out = NC.copy()
    out[s] = np.sign(X[s]) * M[s]
    return out

def variant(A_sq, A_hu):
    return [
        ("amp_sq_brYH",      A_sq, "yh"),
        ("amp_hu_brYH",      A_hu, "yh"),
        ("amp_sq_brM",       A_sq, "model"),
        ("amp_hu_brM",       A_hu, "model"),
        ("amp_brM_magYH",    None, "model"),   # 括号=model sq, 幅度=|YH|
        ("amp_max",          np.maximum(aby, np.where(np.isfinite(A_sq), A_sq, 0.0)), "yh"),
    ]

cands = {"raw": X, "nc": NC, "hy8": HY}
for aggname, (A_sq, A_hu) in (("2src", A2), ("5src", A5)):
    for name, mag, bracket in variant(A_sq, A_hu):
        if name == "amp_brM_magYH":
            M = np.where(np.isfinite(A_sq), A_sq, aby).astype(np.float32)
            thM = np.partition(M, -KDE, axis=1)[:, -KDE][:, None]
            s = (M >= thM) & (X * YH > 0)
            out = NC.copy(); out[s] = np.sign(X[s]) * aby[s]
        else:
            out = build(mag, bracket)
        cands[f"{name}_{aggname}"] = out

# ---- 框架1: sim5 rank-pds + fidpx + nmaepx ----
def truth_de_mask(Yv, k=50):
    k = min(k, Yv.shape[1])
    ath = np.partition(np.abs(Yv), -k, axis=1)[:, -k][:, None]
    return np.abs(Yv) >= ath

DE = truth_de_mask(Y)
Rn = Y / np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
den = float(np.abs(Y)[DE].mean())

def sim5_metrics(P):
    Pn = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    D = 1.0 - Pn @ Rn.T
    ranks = rankdata(D, axis=1)
    pds = float(np.mean(1.0 - ranks[np.arange(n), np.arange(n)] / n))
    agree = (np.sign(P) == np.sign(Y)) & (np.sign(Y) != 0)
    fp = float(np.mean((agree & DE).sum(1) / np.maximum(DE.sum(1), 1)))
    nmae_p = float(np.mean(np.abs(P - Y)[DE & np.isfinite(P)]) / den)
    return pds, fp, nmae_p

print("\n== sim5 框架 ==", flush=True)
r5 = {}
for name, P in cands.items():
    r5[name] = sim5_metrics(P)
    print(f"{name:20s} pds={r5[name][0]:.4f} fidpx={r5[name][1]:.4f} nmaepx={r5[name][2]:.4f}", flush=True)

# ---- 框架2: sim8c 逐行复刻 ----
def cosrow(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    return (a * b).sum(1) / np.maximum(na * nb, 1e-9)

print("\n== sim8c 框架 ==", flush=True)
r8 = {}
for name, v in cands.items():
    pl = R(v)
    npr = n_pred_sim(v)
    agr = (np.sign(pl) == np.sign(Y)).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        nmae = np.minimum(np.abs(pl - Y) / np.maximum(np.abs(Y), 1e-3), 10.0)
    pds = cosrow(np.sign(pl) * np.abs(np.minimum(np.abs(pl), 6.0)),
                 np.sign(Y) * np.abs(np.minimum(np.abs(Y), 6.0))).mean()
    n_conf = (np.abs(Y) > theta).sum(1)
    fid = float(np.mean(50.0 / np.maximum(npr, n_conf)))
    r8[name] = (float(np.median(npr)), float(agr), float(pds),
                float(np.minimum(nmae, 10.0).mean()), fid)
    print(f"{name:20s} n_pred={r8[name][0]:.1f} agree={agr:.4f} pds={pds:.4f} "
          f"nmae={r8[name][3]:.4f} fid={fid:.4f}", flush=True)

# ---- 门判定 ----
print("\n== 门判定 ==", flush=True)
baseA, baseF = r5["raw"][0], r5["raw"][1]
hy8_p5, hy8_p8 = r5["hy8"][0], r8["hy8"][2]
rows = []
for name in cands:
    rows.append(dict(cand=name, pds5=round(r5[name][0], 4),
                     d_vs_A=round(r5[name][0] - baseA, 4), dF=round(r5[name][1] - baseF, 4),
                     d_vs_HY8=round(r5[name][0] - hy8_p5, 4),
                     fidpx=round(r5[name][1], 4),
                     pds8=round(r8[name][2], 4), d8_vs_hy8=round(r8[name][2] - hy8_p8, 4),
                     n_pred=round(r8[name][0], 1), nmae8=r8[name][3], fid8=round(r8[name][4], 4)))
df = pd.DataFrame(rows)
print(df.to_string(index=False), flush=True)

fpH = sim5_metrics(H)[1]
print(f"\nS1 hy8_pds<raw_pds  : {'PASS' if hy8_p5 < baseA else 'FAIL'} ({hy8_p5:.4f} vs {baseA:.4f})")
print(f"S3 |raw_pds-0.5121|<0.10 : {'PASS' if abs(baseA-0.5121) < 0.10 else 'FAIL'} (raw={baseA:.4f})")
print(f"F1 fidpx(H)<fidpx(raw): {'PASS' if fpH < baseF else 'FAIL'} ({fpH:.4f} vs {baseF:.4f})")

adv = df[(df["d_vs_A"] >= 0.045) & (df["dF"] >= -0.001) &
         (df["d_vs_HY8"] > 0) & (df["d8_vs_hy8"] >= 0)]
print("\nADVANCE (绝对门 + 双框架超 hy8):")
print(adv[["cand", "d_vs_A", "dF", "d_vs_HY8", "d8_vs_hy8"]].to_string(index=False) if len(adv) else "  (无)")
df.to_csv(f"{ROOT}/tmp/bzr1/sim11_table.csv", index=False)
print(f"[done] sim11_table.csv t={time.time()-t0:.0f}s", flush=True)
