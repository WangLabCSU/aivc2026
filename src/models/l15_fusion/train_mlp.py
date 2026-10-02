"""L1.5 step4b: 训练 MLP (分支 B, torch GPU)：手工特征 + SE/Tahoe 嵌入。

stage1: 训练={gwps,k562_ess,rpe1}, 验证=jurkat 行; stage2: 全 4 上下文(2% 随机验证)
--no-emb: 消融臂（仅手工特征）
用法: python src/models/l15_fusion/train_mlp.py --stage 1 [--no-emb]
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import load_cfg, work_dir


def load_embeddings(cfg, cache, n_vocab: int):
    """返回 (V, 256) 拼接嵌入（SE 128 + Tahoe 128），缺文件用零块占位。"""
    embs, used = [], []
    for tag, f in [("se", cache / "emb_se.npy"), ("tahoe", cache / "emb_tahoe.npy")]:
        if f.exists():
            e = np.load(f).astype(np.float32)
            embs.append(e)
            used.append(tag)
        else:
            embs.append(np.zeros((n_vocab, int(cfg["embeddings"]["pca_dim"])), np.float32))
    return np.concatenate(embs, axis=1), used


def train(X, y, ctx, gpos, E, mc: dict, tr: np.ndarray, va: np.ndarray, tag: str,
          cache, device: str, tpos_rows: np.ndarray, w: np.ndarray | None = None,
          auxc: dict | None = None) -> dict:
    import torch
    import torch.nn as nn
    t0 = time.time()
    clip, beta = mc["label_clip"], mc["huber_beta"]
    y = np.clip(y, -clip, clip)
    w = np.ones(len(y), np.float32) if w is None else w.astype(np.float32)
    auxc = auxc or {}
    # 基特征标准化（NaN→0）
    mu = np.nanmean(X[tr], axis=0)
    sd = np.nanstd(X[tr], axis=0)
    sd[sd < 1e-6] = 1.0
    Xn = np.where(np.isfinite(X), X, 0.0)
    Xn = (Xn - mu) / sd
    nb = X.shape[1]
    ne = 0 if E is None else E.shape[1]
    E = None if E is None else E.astype(np.float32)
    hidden = mc["hidden"]
    net = nn.Sequential(
        nn.Linear(nb + 2 * ne, hidden[0]), nn.SiLU(), nn.Dropout(mc["dropout"]),
        nn.Linear(hidden[0], hidden[1]), nn.SiLU(), nn.Dropout(mc["dropout"]),
        nn.Linear(hidden[1], 1)).to(device)
    # E 矩阵 = 基因嵌入；j 与 g 各查一次（g 的位置由调用方给 tpos_gpos 数组）
    opt = torch.optim.AdamW(net.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    total_epochs = mc["max_epochs"]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
    huber = nn.HuberLoss(delta=beta, reduction="none")
    bs = mc["batch_size"]
    best, best_state, bad = 1e9, None, 0
    hist = []
    tr_rows = np.where(tr)[0]
    va_rows = np.where(va)[0]

    # ---- v3.5 warm-up: 仅主行（ctx < n_main_ctx）先训 warmup_steps 步（warmup 批次可更小），
    #      5k 步 @batch262k≈43 epoch 超预算 → 以 warmup_batch_size（默认 batch/4）实现 5k 步 ≈ 5 epoch
    n_main = int(auxc.get("n_main_ctx", 6))
    warmup_steps = int(auxc.get("warmup_steps", 0))
    if warmup_steps > 0:
        wbs = int(auxc.get("warmup_batch_size", bs // 4))
        main_rows = tr_rows[ctx[tr_rows] < n_main]
        wm = main_rows[np.random.default_rng(0).permutation(len(main_rows))]
        wi = 0
        net.train()
        for step in range(warmup_steps):
            if wi + wbs > len(wm):  # 重采样续批
                wm = main_rows[np.random.default_rng(step).permutation(len(main_rows))]
                wi = 0
            rows = wm[wi:wi + wbs]
            wi += wbs
            Xb = torch.from_numpy(Xn[rows]).to(device)
            x = Xb
            if E is not None:
                x = torch.cat([Xb, torch.from_numpy(E[gpos[rows]]).to(device),
                               torch.from_numpy(E[tpos_rows[rows]]).to(device)], dim=1)
            yb = torch.from_numpy(y[rows]).to(device)
            opt.zero_grad()
            loss = huber(net(x).squeeze(-1), yb).mean()
            loss.backward()
            opt.step()
        print(f"  warmup {warmup_steps} steps (main-only, bs={wbs}) done "
              f"({time.time()-t0:.0f}s)", flush=True)

    for ep in range(total_epochs):
        net.train()
        perm = np.random.default_rng(ep).permutation(tr_rows)
        tot = 0.0
        wtot = 0.0
        for s in range(0, len(perm), bs):
            rows = perm[s:s + bs]
            Xb = torch.from_numpy(Xn[rows]).to(device)
            x = Xb
            if E is not None:
                ej = torch.from_numpy(E[gpos[rows]]).to(device)
                eg = torch.from_numpy(E[tpos_rows[rows]]).to(device)
                x = torch.cat([Xb, ej, eg], dim=1)
            yb = torch.from_numpy(y[rows]).to(device)
            wb = torch.from_numpy(w[rows]).to(device)
            opt.zero_grad()
            lossv = huber(net(x).squeeze(-1), yb)
            loss = (lossv * wb).sum() / wb.sum()
            loss.backward()
            opt.step()
            tot += float(loss) * float(wb.sum())
            wtot += float(wb.sum())
        sched.step()
        net.eval()
        with torch.no_grad():
            vx = torch.from_numpy(Xn[va_rows]).to(device)
            if E is not None:
                ve = torch.cat([torch.from_numpy(E[gpos[va_rows]]).to(device),
                                torch.from_numpy(E[tpos_rows[va_rows]]).to(device)], dim=1)
                vx = torch.cat([vx, ve], dim=1)
            vy = torch.from_numpy(y[va_rows]).to(device)
            vl = float(huber(net(vx).squeeze(-1), vy).mean())
        hist.append({"epoch": ep, "train": tot / max(wtot, 1), "val": vl})
        print(f"  ep{ep}: train={tot/max(wtot,1):.4f} val={vl:.4f} ({time.time()-t0:.0f}s)",
              flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= mc["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)

    # 验证集预测分块输出
    preds = np.empty(len(va_rows), np.float32)
    with torch.no_grad():
        for s in range(0, len(va_rows), bs):
            rows = va_rows[s:s + bs]
            x = torch.from_numpy(Xn[rows]).to(device)
            if E is not None:
                ve = torch.cat([torch.from_numpy(E[gpos[rows]]).to(device),
                                torch.from_numpy(E[tpos_rows[rows]]).to(device)], dim=1)
                x = torch.cat([x, ve], dim=1)
            preds[s:s + len(rows)] = net(x).squeeze(-1).cpu().numpy()
    np.save(cache / f"mlp_{tag}_valpred.npy", preds)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "names_in": nb, "ne": ne},
               cache / f"mlp_{tag}.pt")
    (cache / f"mlp_{tag}_hist.json").write_text(json.dumps(hist))
    return {"best_val": best, "epochs": len(hist)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--no-emb", action="store_true")
    ap.add_argument("--model-tag", default=None, help="默认 s{stage}[_noemb]")
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    t0 = time.time()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    z = np.load(cache / "feat_train.npz", allow_pickle=False)
    X, y, ctx, gpos = z["X"], z["y"], z["ctx"], z["gpos"]
    w = z["w"] if "w" in z.files else np.ones(len(y), np.float32)
    # target-as-gene 词表位置：assemble 现统一写 tpos_train.npy（含 aux 占位 0）
    tp_path = cache / "tpos_train.npy"
    if tp_path.exists():
        tpos_rows = np.load(tp_path)
    else:
        from src.models.l15_fusion.common import build_task_table, build_vocab
        vocab = build_vocab(cfg)
        parts = []
        for c in cfg["train_contexts"]:
            tb = build_task_table(c, cfg, vocab, cache)
            parts.append(vocab.get_indexer(pd_index(tb["target_gene"])))
        tpos_rows = np.concatenate(parts).astype(np.int32)
        np.save(tp_path, tpos_rows)
    jidx = cfg["train_contexts"].index(cfg["stage1_holdout"])
    if args.stage == 1:
        tr, va = ctx != jidx, ctx == jidx
        tag = (args.model_tag or f"s{args.stage}") + ("_noemb" if args.no_emb else "")
    else:
        rng = np.random.default_rng(0)
        u = rng.random(len(y))
        va = u < 0.02
        tr = ~va
        tag = (args.model_tag or f"s{args.stage}") + ("_noemb" if args.no_emb else "")
    E, used = (None, []) if args.no_emb else load_embeddings(cfg, cache, int(z["gpos"].max()) + 1)
    auxc = dict(cfg.get("aux") or {})
    auxc["n_main_ctx"] = len(cfg["train_contexts"])
    has_aux = bool(auxc.get("contexts")) and "w" in z.files
    print(f"[{time.time()-t0:.0f}s] train={tr.sum()} val={va.sum()} emb={used} dev={device} "
          f"aux={'on' if has_aux else 'off'} (w={auxc.get('weight')}, "
          f"warmup_steps={auxc.get('warmup_steps', 0) if has_aux else 0})", flush=True)
    # gpos.max 可能小于词表长（未覆盖尾部基因），E 按 vocab 长构造
    from src.models.l15_fusion.common import build_vocab
    nv = len(build_vocab(cfg))
    if E is not None and len(E) < nv:
        E = np.concatenate([E, np.zeros((nv - len(E), E.shape[1]), np.float32)])
        print("  padded E to vocab", flush=True)
    res = train(X, y, ctx, gpos, E, cfg["mlp"], tr, va, tag, cache, device, tpos_rows,
                w=w if has_aux else None, auxc=auxc if has_aux else None)
    res["emb_used"] = used
    (cache / f"mlp_{tag}_result.json").write_text(json.dumps(res))
    print(f"train_mlp {tag} complete: {res} ({time.time()-t0:.0f}s)", flush=True)


def pd_index(s):
    import pandas as pd
    return pd.Index(s)


if __name__ == "__main__":
    main()
