"""L1.5 step4a: 训练 XGBoost (分支 A)。

stage1: 训练={gwps,k562_ess,rpe1}, 早停+评估=jurkat 行（LOCO-jurkat）
stage2: 训练=全部 4 上下文, 早停=随机 2% 训练行

用法: python src/models/l15_fusion/train_xgb.py --stage 1
      python src/models/l15_fusion/train_xgb.py --stage 2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.l15_fusion.common import load_cfg, work_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    args = ap.parse_args()
    cfg = load_cfg()
    cache = work_dir(cfg)
    t0 = time.time()
    z = np.load(cache / "feat_train.npz", allow_pickle=False)
    X, y, ctx = z["X"], z["y"], z["ctx"]
    names = [str(s) for s in z["names"]]
    jidx = cfg["train_contexts"].index(cfg["stage1_holdout"])
    xc = cfg["xgb"]
    clip = float(xc["label_clip"])
    if args.stage == 1:
        tr = ctx != jidx
        va = ctx == jidx
        tag = "s1"
    else:
        rng = np.random.default_rng(0)
        u = rng.random(len(y))
        va = u < 0.02
        tr = ~va
        tag = "s2"
    y = np.clip(y, -clip, clip)
    print(f"[{time.time()-t0:.0f}s] train={tr.sum()} val={va.sum()}", flush=True)

    import xgboost as xgb
    model = xgb.XGBRegressor(
        objective=xc["objective"], max_depth=xc["max_depth"],
        n_estimators=xc["n_estimators"], learning_rate=xc["learning_rate"],
        subsample=xc["subsample"], colsample_bytree=xc["colsample_bytree"],
        tree_method=xc["tree_method"], n_jobs=xc["n_jobs"],
        early_stopping_rounds=xc["early_stopping_rounds"], eval_metric="mphe")
    model.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=25)
    print(f"[{time.time()-t0:.0f}s] fit done, best_iter={model.best_iteration}", flush=True)
    vp = model.predict(X[va])
    model.get_booster().save_model(str(cache / f"xgb_{tag}.json"))
    np.save(cache / f"xgb_{tag}_valpred.npy", vp)
    np.save(cache / f"xgb_{tag}_validx.npy", np.where(va)[0])
    # 特征重要性（gain）
    bi = model.get_booster().get_score(importance_type="gain")
    imp = sorted(((names[int(k[1:])], float(v)) for k, v in bi.items()),
                 key=lambda t: -t[1])
    tot = sum(v for _, v in imp) or 1.0
    imp_norm = [(n, round(v, 2), round(v / tot, 4)) for n, v in imp]
    (cache / f"xgb_{tag}_importance.json").write_text(json.dumps(imp_norm, indent=1))
    print("top10 gain:", imp_norm[:10], flush=True)
    print(f"train_xgb stage{args.stage} complete ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
