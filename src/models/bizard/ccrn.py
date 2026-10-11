"""Bizard CCRN: Context-Conditioned Response Network (v0.1 深度骨干).

设计 (docs/bizard/v01_architecture.md §2.1):
  输入 = fusion v3.5 的 22 维逐行特征 (跨源签名统计 + basal + 模块标志)
       + 上下文指纹 9 维 (6 模块 basal 分数 + basal 均值/SD/非零占比, 全部可从
         NTC basal 推出 → 对未见上下文 A/B/C 与 LOCO-HepG2 可泛化)
       + 交互项 6 维 (上下文模块分数 × 响应基因模块标志)
       + 1 维   (basal_j − 上下文平均 basal, 表达中心化)
  骨干 = MLP 38 → 2048 → 1024 → 1 (SiLU + Dropout), Huber + AdamW + cosine LR
  训练 = feat_train.npz 30.78M 行 (6 CRISPRi 上下文 + tahoe aux w=0.3), 全程 GPU 常驻

子命令: build / train --arm {loco,full} / predict --src ... --arm ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
CFG_PATH = ROOT / "configs/bizard/ccrn.yaml"
# feat_train ctx id → 行上下文名; bench/submit 追加 8-10
N_BASE_FEAT = 22
MOD_COL0 = 13          # X 内 mod_mt..mod_myc 列起点
BASAL_J_COL = 3        # basal_j 列


def load_cfg() -> dict:
    import os
    p = os.environ.get("CCRN_CFG") or "configs/bizard/ccrn.yaml"
    with open(ROOT / p) as f:
        return yaml.safe_load(f)


def wd(cfg: dict) -> Path:
    d = ROOT / cfg["workdir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def module_flags(vocab: list[str], cfg: dict) -> np.ndarray:
    """6 模块 × vocab 标志阵 (与 l15_fusion.common.module_flags 同逻辑, 独立轻量版)."""
    ms = yaml.safe_load(open(ROOT / "configs/l15_fusion/v35.yaml"))["features"]["module_sets"]
    names = list(ms.keys())
    genes = np.asarray(vocab)
    flags = np.zeros((len(names), len(vocab)), np.float32)
    for i, m in enumerate(names):
        spec = ms[m]
        if isinstance(spec, list):
            spec = {"genes": spec}
        hit = np.zeros(len(vocab), bool)
        for pre in spec.get("prefix", []):
            hit |= np.char.startswith(genes, pre)
        for g in spec.get("genes", []):
            hit |= (genes == g)
        flags[i] = hit
    return flags


def ctx_fingerprint(basal: np.ndarray, mflags: np.ndarray) -> np.ndarray:
    """上下文指纹: [6 模块 basal 分数, basal 均值, SD, 非零占比] (9,).

    模块在上下文面板中无表达 (全 0, 如 A/B/C 缺 RPS/RPL) → NaN,
    由 build_fingerprints 用训练上下文中位数插补 (OOD 安全, 避免 -7σ 外推)。
    """
    scores = []
    for i in range(mflags.shape[0]):
        sel = mflags[i] > 0
        s = float(basal[sel].mean()) if sel.any() else 0.0
        scores.append(s if s > 0.1 else np.nan)   # 模块在面板中缺失 (≈0) → NaN 待插补
    return np.array(scores + [float(basal.mean()), float(basal.std()),
                              float((basal > 0).mean())], np.float32)


def all_ctx_names(cfg: dict) -> list[str]:
    return list(cfg["train_contexts"]) + list(cfg["extra_contexts"])


def build_fingerprints(cfg: dict, mflags: np.ndarray) -> dict[str, np.ndarray]:
    bdir = ROOT / cfg["fusion_v35_dir"]
    out = {}
    for name in all_ctx_names(cfg):
        b = np.load(bdir / f"basal_{name}.npy").astype(np.float32)
        out[name] = ctx_fingerprint(b, mflags)
    # 缺失模块分数 → 训练上下文 (0-7) 中位数插补 (A/B/C 面板缺 RPS/RPL 等)
    F = np.stack([out[n] for n in all_ctx_names(cfg)])
    tr_med = np.nanmedian(F[:len(cfg["train_contexts"])], axis=0)
    F = np.where(np.isnan(F), tr_med[None, :], F)
    return {n: F[i] for i, n in enumerate(all_ctx_names(cfg))}


def augment(X: np.ndarray, ctx_ids: np.ndarray, fps: np.ndarray,
            mflags: np.ndarray) -> np.ndarray:
    """22 维 → 38 维 (指纹广播 9 + 指纹×模块交互 6 + basal 中心化 1)."""
    n = X.shape[0]
    X2 = np.zeros((n, N_BASE_FEAT + 16), np.float32)
    X2[:, :N_BASE_FEAT] = X
    F = fps[ctx_ids]                                  # (n, 9)
    X2[:, N_BASE_FEAT:N_BASE_FEAT + 9] = F
    for m in range(mflags.shape[0]):                  # 交互: ctx 模块分数 × j 模块标志
        X2[:, N_BASE_FEAT + 9 + m] = F[:, m] * X[:, MOD_COL0 + m]
    X2[:, -1] = X[:, BASAL_J_COL] - F[:, 6]           # basal_j − ctx 均值
    return X2


# ------------------------------------------------------------------- build
def cmd_build(cfg: dict) -> None:
    t0 = time.time()
    out = wd(cfg)
    vocab = json.load(open(ROOT / cfg["vocab_json"]))
    mflags = module_flags(vocab, cfg)
    fps_list = build_fingerprints(cfg, mflags)
    names = all_ctx_names(cfg)
    fps = np.stack([fps_list[n] for n in names])
    np.save(out / "fingerprints.npy", fps)
    print(f"[{time.time()-t0:.0f}s] fingerprints for {names}", flush=True)

    v35 = ROOT / cfg["fusion_v35_dir"]
    z = np.load(v35 / "feat_train.npz", allow_pickle=False)
    ctx = z["ctx"].astype(np.int64)
    X2 = augment(z["X"], ctx, fps, mflags)
    np.save(out / "X_train_ccrn.npy", X2)
    np.savez(out / "train_meta.npz", y=z["y"], ctx=ctx, w=z["w"])
    del X2, z
    print(f"[{time.time()-t0:.0f}s] train {len(ctx)} rows x 38 saved", flush=True)

    for src, cname in [("feat_bench_hepg2.npz", "hepg2"),
                       ("feat_submit_A.npz", "a"),
                       ("feat_submit_B.npz", "b"),
                       ("feat_submit_C.npz", "c")]:
        zs = np.load(v35 / src, allow_pickle=False)
        cid = names.index(cname)
        X2 = augment(zs["X"], np.full(len(zs["X"]), cid), fps, mflags)
        tag = src.split(".")[0].replace("feat_", "")
        np.save(out / f"X_{tag}_ccrn.npy", X2)
        extra = {k: zs[k] for k in zs.files if k != "X"}
        np.savez(out / f"meta_{tag}.npz", **extra)
        del X2, zs
        print(f"[{time.time()-t0:.0f}s] {src} → 38 维 saved", flush=True)
    print("build complete", flush=True)


# ------------------------------------------------------------------- train
def make_net(cfg: dict, n_in: int, device: str):
    import torch.nn as nn
    h = cfg["model"]["hidden"]
    return nn.Sequential(
        nn.Linear(n_in, h[0]), nn.SiLU(), nn.Dropout(cfg["model"]["dropout"]),
        nn.Linear(h[0], h[1]), nn.SiLU(), nn.Dropout(cfg["model"]["dropout"]),
        nn.Linear(h[1], 1)).to(device)


def cmd_train(cfg: dict, arm: str) -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    t0 = time.time()
    out = wd(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tc = cfg["train_contexts"]
    hold = tc.index(cfg["loco_holdout"])
    meta = np.load(out / "train_meta.npz")
    y, ctx, w = meta["y"], meta["ctx"], meta["w"]
    if arm == "loco":
        tr, va = ctx != hold, ctx == hold
    else:
        rng = np.random.default_rng(cfg["train"]["seed"])
        u = rng.random(len(y))
        tr, va = u >= 0.02, u < 0.02
    tr_i, va_i = np.where(tr)[0], np.where(va)[0]
    # 全量上 GPU (28M x 38 f32 ≈ 4.3GB, L20 46GB 足够); 标准化统计用 1/20 抽样估
    X = np.load(out / "X_train_ccrn.npy", mmap_mode="r")
    S = np.asarray(X[tr_i[::20]], np.float64)          # ~1.4M 行样本
    mu = np.nanmean(S, axis=0).astype(np.float32)      # NaN(缺签名)不计统计
    sd = np.nanstd(S, axis=0).astype(np.float32)
    del S
    sd[sd < 1e-6] = 1.0
    mut, sdt = torch.from_numpy(mu).to(device), torch.from_numpy(sd).to(device)

    Xg = torch.empty((len(tr_i), X.shape[1]), dtype=torch.float32, device=device)
    CH = 2_000_000
    for s in range(0, len(tr_i), CH):
        rows = tr_i[s:s + CH]
        Xb = torch.from_numpy(np.where(np.isfinite(X[rows]), X[rows], 0.0)).to(device)
        Xg[s:s + len(rows)] = (Xb - mut) / sdt
        del Xb
    yg = torch.from_numpy(np.clip(y[tr_i], -cfg["train"]["label_clip"],
                                  cfg["train"]["label_clip"])).to(device)
    wg = torch.from_numpy(w[tr_i]).to(device)
    Xv = torch.empty((len(va_i), X.shape[1]), dtype=torch.float32, device=device)
    for s in range(0, len(va_i), CH):
        rows = va_i[s:s + CH]
        Xb = torch.from_numpy(np.where(np.isfinite(X[rows]), X[rows], 0.0)).to(device)
        Xv[s:s + len(rows)] = (Xb - mut) / sdt
        del Xb
    yv = torch.from_numpy(np.clip(y[va_i], -cfg["train"]["label_clip"],
                                  cfg["train"]["label_clip"])).to(device)
    print(f"[{time.time()-t0:.0f}s] arm={arm} train={len(tr_i)} val={len(va_i)} dev={device}",
          flush=True)

    tc_, mc = cfg["train"], cfg["model"]
    torch.manual_seed(tc_["seed"])
    net = make_net(cfg, X.shape[1], device)
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=tc_["lr"], weight_decay=tc_["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc_["max_epochs"])
    huber = torch.nn.HuberLoss(delta=tc_["huber_beta"], reduction="none")
    bs = tc_["batch_size"]
    best, best_state, bad, hist = 1e9, None, 0, []
    n = len(tr_i)
    for ep in range(tc_["max_epochs"]):
        net.train()
        perm = torch.randperm(n, device=device)
        tot = wtot = 0.0
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            lv = huber(net(Xg[idx]).squeeze(-1), yg[idx])
            loss = (lv * wg[idx]).sum() / wg[idx].sum()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * float(wg[idx].sum())
            wtot += float(wg[idx].sum())
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = 0.0
            for s in range(0, len(Xv), bs):
                vl += float(huber(net(Xv[s:s + bs]).squeeze(-1), yv[s:s + bs]).sum())
            vl /= max(len(Xv), 1)
        hist.append({"epoch": ep, "train": tot / max(wtot, 1), "val": vl,
                     "sec": round(time.time() - t0, 1)})
        print(f"  ep{ep}: train={tot/max(wtot,1):.4f} val={vl:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= tc_["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "n_in": int(X.shape[1]),
                "n_params": int(n_par), "arm": arm}, out / f"ccrn_{arm}.pt")
    (out / f"ccrn_{arm}_hist.json").write_text(json.dumps(
        {"hist": hist, "best_val": best, "n_params": n_par,
         "train_rows": int(len(tr_i)), "val_rows": int(len(va_i))}))
    print(f"train[{arm}] complete: best_val={best:.4f} params={n_par} "
          f"({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------------- predict
def cmd_predict(cfg: dict, src: str, arm: str, clip: float) -> None:
    import torch
    torch.backends.cuda.matmul.allow_tf32 = True
    t0 = time.time()
    out = wd(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(out / f"ccrn_{arm}.pt", map_location=device, weights_only=False)
    net = make_net(cfg, ck["n_in"], device)
    net.load_state_dict(ck["state"])
    net.eval()
    mu = torch.from_numpy(ck["mu"]).to(device)
    sd = torch.from_numpy(ck["sd"]).to(device)
    X = np.load(out / f"X_{src}_ccrn.npy", mmap_mode="r")
    meta = np.load(out / f"meta_{src}.npz", allow_pickle=True)
    preds = np.empty(len(X), np.float32)
    with torch.no_grad():
        for s in range(0, len(X), 1 << 20):
            Xb = torch.from_numpy(np.where(np.isfinite(X[s:s + (1 << 20)]),
                                           X[s:s + (1 << 20)], 0.0)).to(device)
            preds[s:s + len(Xb)] = net((Xb - mu) / sd).squeeze(-1).cpu().numpy()
    preds = np.clip(np.nan_to_num(preds, nan=0.0), -clip, clip).astype(np.float32)
    tag = f"ccrn1_{arm}_{src}"
    np.save(out / f"pred_{tag}.npy", preds)
    df = pd.DataFrame({"target_gene": meta["target"].astype(str),
                       "gene": meta["gene"].astype(str), "delta": preds})
    df.to_parquet(out / f"pred_{tag}.parquet")
    print(f"predict[{tag}] done: n={len(preds)} mean={preds.mean():.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "train", "predict"])
    ap.add_argument("--arm", default="loco", choices=["loco", "full"])
    ap.add_argument("--src", default="bench_hepg2",
                    help="bench_hepg2 | submit_A | submit_B | submit_C")
    ap.add_argument("--clip", type=float, default=4.0)
    args = ap.parse_args()
    cfg = load_cfg()
    if args.cmd == "build":
        cmd_build(cfg)
    elif args.cmd == "train":
        cmd_train(cfg, args.arm)
    else:
        cmd_predict(cfg, args.src, args.arm, args.clip)


if __name__ == "__main__":
    main()
