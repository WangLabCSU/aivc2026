#!/usr/bin/env python
"""sim10 — wrong-drop 细分: HY8 只动 de∩disagree (X*YH<0), 保留 sel (agree) 收缩.
候选: nc / hy8 / hy8wd (zero@disagree) / hy8wd5 (half@disagree) / wonly (只zero@disagree, sel保留NC).
同一代理框架 (sim8c), 输出 n_pred/agree/pds/nmae/fid.
"""
import numpy as np

ROOT = "/data/resource/wanglab/aivc/aivc2026"
LAM = 7.0
CAP = np.log2(14.0)
KDE = 8

z = np.load(f"{ROOT}/tmp/bzr1/sim5_cache.npz", allow_pickle=True)
X, Y, YH = z["X"], z["Y"], z["YH"]
n, G = X.shape

ntc_mean = np.load(f"{ROOT}/tmp/bzr1/ntc_mean_g6146.npy").astype(np.float64)
ntc_sd = np.load(f"{ROOT}/tmp/bzr1/ntc_sd_g6146.npy").astype(np.float64)
cv = ntc_sd / np.maximum(ntc_mean, 0.5)

def R(v):
    return np.minimum(LAM * v, CAP)

C0 = 2.5
theta = C0 * cv + 0.05

def n_pred_sim(v):
    pl = R(v)
    p = 1.0 / (1.0 + np.exp(-(np.abs(pl) - theta) / (0.25 * theta + 1e-6)))
    return p.sum(axis=1)

mu = X.mean(0, keepdims=True)
H = (X - mu).astype(np.float32)
NC = np.where(X * H > 0, H, X).astype(np.float32)

aby = np.abs(YH)
th = np.partition(aby, -KDE, axis=1)[:, -KDE][:, None]
de = aby >= th
sel = de & (X * YH > 0)
dis = de & (X * YH < 0)
print(f"[regime] de rows/row={de.sum(1).mean():.2f} sel={sel.sum(1).mean():.2f} dis={dis.sum(1).mean():.2f}")

HY = NC.copy();  HY[sel] = np.sign(X[sel]) * aby[sel]
WD  = HY.copy(); WD[dis] = 0.0
WD5 = HY.copy(); WD5[dis] = HY[dis] * 0.5
WONLY = NC.copy(); WONLY[dis] = 0.0

cands = {"nc": NC, "hy8": HY, "hy8wd": WD, "hy8wd5": WD5, "wonly": WONLY}

def truth_de_mask(Y, k=50):
    ath = np.partition(np.abs(Y), -k, axis=1)[:, -k][:, None]
    return np.abs(Y) >= ath

tde = truth_de_mask(Y)

rows = []
for name, v in cands.items():
    pl = R(v)
    npr = n_pred_sim(v)
    agr = (np.sign(pl) == np.sign(Y))[tde].mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        nmae = np.minimum(np.abs(pl - Y) / np.maximum(np.abs(Y), 1e-3), 10.0)
    def cosrow(a, b):
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        return (a * b).sum(1) / np.maximum(na * nb, 1e-9)
    pds = cosrow(np.sign(pl) * np.abs(np.minimum(np.abs(pl), 6.0)),
                 np.sign(Y) * np.abs(np.minimum(np.abs(Y), 6.0))).mean()
    n_conf = (np.abs(Y) > theta).sum(1)
    fid = np.mean(50.0 / np.maximum(npr, n_conf))
    rows.append(dict(cand=name, n_pred=float(np.median(npr)),
                     agree=round(float(agr), 4), pds=round(float(pds), 4),
                     nmae=round(float(nmae.mean()), 4), fid=round(float(fid), 4)))

import pandas as pd
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv(f"{ROOT}/tmp/bzr1/sim10_table.csv", index=False)
print("[done] sim10_table.csv")
