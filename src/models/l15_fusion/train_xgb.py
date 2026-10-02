"""L1.5 step4a: 训练 XGBoost (分支 A)。

stage1: 训练={gwps,k562_ess,rpe1}+aux, 早停+评估=jurkat 行（LOCO-jurkat）
stage2: 训练=全部上下文+aux, 早停=随机 2% 训练行
v3.5 aux（cfg["aux"]）: tahoe 药扰行（ctx>=len(train_contexts)）sample_weight=aux.weight，
  先主任务 warm-up（aux.xgb_warmup_trees 轮, 仅主行）再 xgb_model 续训混训。
--model-tag: 模型保存名（默认 s{stage}；v3.5 用 s35/s2_35 区分）。

用法: python src/models/l15_fusion/train_xgb.py --stage 1
      python src/models/l15_fusion/train_xgb.py --stage 1 --model-tag s35
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import load_cfg, work_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--model-tag", default=None, help="默认 s{stage}")
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    t0 = time.time()
    z = np.load(cache / "feat_train.npz", allow_pickle=False)
    X, y, ctx = z["X"], z["y"], z["ctx"]
    w = z["w"] if "w" in z.files else np.ones(len(y), np.float32)
    names = [str(s) for s in z["names"]]
    jidx = cfg["train_contexts"].index(cfg["stage1_holdout"])
    xc = cfg["xgb"]
    clip = float(xc["label_clip"])
    n_main_ctx = len(cfg["train_contexts"])
    is_aux = ctx >= n_main_ctx
    if args.stage == 1:
        tr = ctx != jidx
        va = ctx == jidx
    else:
        rng = np.random.default_rng(0)
        u = rng.random(len(y))
        va = u < 0.02
        tr = ~va
    tag = args.model_tag or f"s{args.stage}"
    y = np.clip(y, -clip, clip)
    auxc = cfg.get("aux") or {}
    warmup = bool(auxc.get("warmup", True)) and is_aux.any() and auxc.get("weight", 0) > 0
    n_warm = int(xc["n_estimators"] * 0.5) if warmup else 0  # warm-up 轮数 = 一半
    print(f"[{time.time()-t0:.0f}s] train={tr.sum()} (aux {int((tr&is_aux).sum())}) "
          f"val={va.sum()} warmup_trees={n_warm}", flush=True)

    import xgboost as xgb
    params = {"objective": xc["objective"], "max_depth": xc["max_depth"],
              "learning_rate": xc["learning_rate"], "subsample": xc["subsample"],
              "colsample_bytree": xc["colsample_bytree"], "tree_method": xc["tree_method"],
              "n_jobs": xc["n_jobs"], "eval_metric": "mphe"}
    if warmup and n_warm > 0:
        # phase 1: 仅主行 warm-up（等价 LOCO 主任务预热）
        m_model = xgb.XGBRegressor(n_estimators=n_warm, **params)
        m_model.fit(X[tr & ~is_aux], y[tr & ~is_aux], sample_weight=w[tr & ~is_aux],
                    verbose=False)
        booster = m_model.get_booster()
        print(f"[{time.time()-t0:.0f}s] warm-up {n_warm} trees (main only) done", flush=True)
    else:
        booster = None
    # phase 2: 全量混训（aux 行 w=aux.weight），早停同 v3
    dtr = xgb.DMatrix(X[tr], label=y[tr], weight=w[tr])
    dva = xgb.DMatrix(X[va], label=y[va], weight=w[va])
    booster = xgb.train(params, dtr,
                        num_boost_round=xc["n_estimators"] - (n_warm if warmup and booster else 0),
                        evals=[(dva, "val")], early_stopping_rounds=xc["early_stopping_rounds"],
                        xgb_model=booster, verbose_eval=25)
    print(f"[{time.time()-t0:.0f}s] fit done, best_iter={booster.best_iteration}", flush=True)
    vp = booster.predict(xgb.DMatrix(X[va]))
    booster.save_model(str(cache / f"xgb_{tag}.json"))
    np.save(cache / f"xgb_{tag}_valpred.npy", vp)
    np.save(cache / f"xgb_{tag}_validx.npy", np.where(va)[0])
    bi = booster.get_score(importance_type="gain")
    imp = sorted(((names[int(k[1:])], float(v)) for k, v in bi.items()), key=lambda t: -t[1])
    tot = sum(v for _, v in imp) or 1.0
    imp_norm = [(n, round(v, 2), round(v / tot, 4)) for n, v in imp]
    (cache / f"xgb_{tag}_importance.json").write_text(json.dumps(imp_norm, indent=1))
    print("top10 gain:", imp_norm[:10], flush=True)
    print(f"train_xgb {tag} complete ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
