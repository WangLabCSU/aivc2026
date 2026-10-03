"""Bizard L2: 单鸟训练（fusion v3.5 骨干, config 驱动变异）。

每鸟 = feat_train.npz（v35 共享缓存, 只读）× 鸟专属 sample_weight（主场景行 ×main_weight,
aux 行 = aux_weight）× seed。分支（对齐 v3.5 stage2 提交臂）:
  XGB — warm-up 200 树（仅主行）→ 混训 aux 行（w），早停 = 随机 2% 训练行
  MLP — no-emb 256/128 SiLU Huber + 5k 步主行 warm-up
产物落 <workdir>/<bird>/xgb_{bird}.json, mlp_{bird}.pt（每鸟独立目录, 每鸟落盘防中断）。

用法: python src/models/bizard/train_bird.py --bird b1_p [--branch xgb|mlp|both]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.bizard.swarm import ROOT, birds, fusion_cfg, load_cfg, workdir  # noqa: E402


def bird_weights(fcfg: dict, ctx: np.ndarray, bird: dict) -> np.ndarray:
    """鸟专属行权: 主场景训练行 ×main_weight；aux 行 = aux_weight。"""
    n_main = len(fcfg["train_contexts"])
    w = np.ones(len(ctx), np.float32)
    if bird.get("scenario"):
        main_idx = fcfg["train_contexts"].index(
            load_cfg()["scenarios"][bird["scenario"]]["train_ctx"])
        w[ctx == main_idx] = float(bird["main_weight"])
    w[ctx >= n_main] = float(bird.get("aux_weight", 0.3))
    return w


def train_xgb_bird(fcfg: dict, outdir: Path, bird: dict, w: np.ndarray, X: np.ndarray,
                   y: np.ndarray, ctx: np.ndarray, t0: float) -> None:
    import xgboost as xgb
    xc, clip = fcfg["xgb"], float(fcfg["xgb"]["label_clip"])
    n_main = len(fcfg["train_contexts"])
    is_aux = ctx >= n_main
    rng = np.random.default_rng(0)
    va = rng.random(len(y)) < 0.02
    tr = ~va
    y = np.clip(y, -clip, clip)
    auxc = fcfg.get("aux") or {}
    warmup = bool(auxc.get("warmup", True)) and bool(is_aux.any()) \
        and float(bird["aux_weight"]) > 0
    n_warm = int(xc["n_estimators"] * 0.5) if warmup else 0
    tag = bird["name"]
    print(f"[xgb {tag}] train={tr.sum()} (aux {int((tr & is_aux).sum())}) val={va.sum()} "
          f"main_weight={bird['main_weight']} seed={bird['seed']}", flush=True)
    params = {"objective": xc["objective"], "max_depth": xc["max_depth"],
              "learning_rate": xc["learning_rate"], "subsample": xc["subsample"],
              "colsample_bytree": xc["colsample_bytree"], "tree_method": xc["tree_method"],
              "n_jobs": xc["n_jobs"], "eval_metric": "mphe", "seed": int(bird["seed"])}
    booster = None
    if warmup and n_warm > 0:
        m = xgb.XGBRegressor(n_estimators=n_warm, **params)
        m.fit(X[tr & ~is_aux], y[tr & ~is_aux], sample_weight=w[tr & ~is_aux], verbose=False)
        booster = m.get_booster()
        print(f"[xgb {tag}] [{time.time()-t0:.0f}s] warm-up {n_warm} trees", flush=True)
    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w[tr])
    dva = xgb.DMatrix(X[va], label=y[va], weight=w[va])
    booster = xgb.train(params, dtr,
                        num_boost_round=xc["n_estimators"] - (n_warm if warmup and booster else 0),
                        evals=[(dva, "val")], early_stopping_rounds=xc["early_stopping_rounds"],
                        xgb_model=booster, verbose_eval=100)
    booster.save_model(str(outdir / f"xgb_{tag}.json"))
    vp = booster.predict(xgb.DMatrix(X[va]))
    np.save(outdir / f"xgb_{tag}_valpred.npy", vp)
    np.save(outdir / f"xgb_{tag}_validx.npy", np.where(va)[0])
    print(f"[xgb {tag}] fit done best_iter={booster.best_iteration} "
          f"({time.time()-t0:.0f}s)", flush=True)


def train_mlp_bird(fcfg: dict, outdir: Path, bird: dict, w: np.ndarray, X: np.ndarray,
                   y: np.ndarray, ctx: np.ndarray, t0: float) -> None:
    import torch
    import torch.nn as nn
    mc = fcfg["mlp"]
    clip, beta = float(mc["label_clip"]), float(mc["huber_beta"])
    tag = bird["name"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    y = np.clip(y, -clip, clip)
    rng = np.random.default_rng(0)
    va = rng.random(len(y)) < 0.02
    va_rows = np.where(va)[0]
    tr_rows = np.where(~va)[0]
    mu = np.nanmean(X[tr_rows], axis=0)
    sd = np.nanstd(X[tr_rows], axis=0)
    sd[sd < 1e-6] = 1.0
    Xn = np.where(np.isfinite(X), X, 0.0)
    Xn = (Xn - mu) / sd
    nb = X.shape[1]
    ne = 0  # no-emb（对齐 v3.5 stage2 提交臂）
    net = nn.Sequential(nn.Linear(nb + 2 * ne, mc["hidden"][0]), nn.SiLU(),
                        nn.Dropout(mc["dropout"]), nn.Linear(mc["hidden"][0], mc["hidden"][1]),
                        nn.SiLU(), nn.Dropout(mc["dropout"]), nn.Linear(mc["hidden"][1], 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=mc["lr"], weight_decay=mc["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=mc["max_epochs"])
    huber = nn.HuberLoss(delta=beta, reduction="none")
    bs = mc["batch_size"]
    n_main = len(fcfg["train_contexts"])
    auxc = dict(fcfg.get("aux") or {})
    warmup_steps = int(auxc.get("warmup_steps", 0))
    if warmup_steps > 0:
        wbs = int(auxc.get("warmup_batch_size", bs // 4))
        main_rows = tr_rows[ctx[tr_rows] < n_main]
        wm = main_rows[np.random.default_rng(int(bird["seed"])).permutation(len(main_rows))]
        wi = 0
        net.train()
        for step in range(warmup_steps):
            if wi + wbs > len(wm):
                wm = main_rows[np.random.default_rng(
                    step + int(bird["seed"])).permutation(len(main_rows))]
                wi = 0
            rows = wm[wi:wi + wbs]
            wi += wbs
            x = torch.from_numpy(Xn[rows]).to(dev)
            yb = torch.from_numpy(y[rows]).to(dev)
            opt.zero_grad()
            loss = huber(net(x).squeeze(-1), yb).mean()
            loss.backward()
            opt.step()
        print(f"[mlp {tag}] warmup {warmup_steps} steps ({time.time()-t0:.0f}s)", flush=True)

    best, best_state, bad = 1e9, None, 0
    hist = []
    for ep in range(mc["max_epochs"]):
        net.train()
        perm = np.random.default_rng(ep + int(bird["seed"])).permutation(tr_rows)
        tot = wtot = 0.0
        for s in range(0, len(perm), bs):
            rows = perm[s:s + bs]
            x = torch.from_numpy(Xn[rows]).to(dev)
            yb = torch.from_numpy(y[rows]).to(dev)
            wb = torch.from_numpy(w[rows]).to(dev)
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
            vx = torch.from_numpy(Xn[va_rows]).to(dev)
            vy = torch.from_numpy(y[va_rows]).to(dev)
            vl = float(huber(net(vx).squeeze(-1), vy).mean())
        hist.append({"epoch": ep, "train": tot / max(wtot, 1), "val": vl})
        print(f"[mlp {tag}] ep{ep}: train={tot/max(wtot,1):.4f} val={vl:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= mc["patience"]:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd, "names_in": nb, "ne": ne},
               outdir / f"mlp_{tag}.pt")
    (outdir / f"mlp_{tag}_hist.json").write_text(json.dumps(hist))
    print(f"[mlp {tag}] complete best_val={best:.5f} ({time.time()-t0:.0f}s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird", required=True)
    ap.add_argument("--branch", default="both", choices=["both", "xgb", "mlp"])
    args = ap.parse_args()
    cfg = load_cfg()
    bird = next(b for b in birds(cfg) if b["name"] == args.bird)
    if bird.get("reuse"):
        print(f"[{bird['name']}] reuse v35 模型 {bird['reuse']} — 无需训练", flush=True)
        return
    fcfg = fusion_cfg(cfg)
    cache = ROOT / cfg["v35_workdir"]
    outdir = workdir(cfg) / bird["name"]
    outdir.mkdir(parents=True, exist_ok=True)
    done_flag = outdir / ".train_done"
    branch_flag = outdir / f".{args.branch}_done"
    if (done_flag if args.branch == "both" else branch_flag).exists() or done_flag.exists():
        print(f"[{bird['name']}] {args.branch} already done, skip", flush=True)
        return
    t0 = time.time()
    z = np.load(cache / "feat_train.npz", allow_pickle=False)
    X, y, ctx = z["X"], z["y"], z["ctx"]
    w = bird_weights(fcfg, ctx, bird)
    print(f"[{bird['name']}] feat={X.shape} w: main={bird['main_weight']} "
          f"aux={bird['aux_weight']} seed={bird['seed']}", flush=True)
    if args.branch in ("both", "xgb"):
        train_xgb_bird(fcfg, outdir, bird, w, X, y, ctx, t0)
    if args.branch in ("both", "mlp"):
        train_mlp_bird(fcfg, outdir, bird, w, X, y, ctx, t0)
    flag = done_flag if args.branch == "both" else branch_flag
    flag.write_text(json.dumps({"bird": bird, "branch": args.branch,
                                "seconds": time.time() - t0}))
    print(f"train_bird {bird['name']} {args.branch} complete ({time.time()-t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    main()
