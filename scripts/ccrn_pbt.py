"""CCRN PBT 驱动 — 种群 4、代际 exploit/explore，奖励 = LOCO bench anchor。

每 worker 独立 workdir（特征缓存 symlink 自 base ccrn/，仅 ckpt/pred 本地），
每代: train(loco) → predict(bench_hepg2) → λ∈{2.5,4.0} 双点 bench → reward=max。
judge 纪律: bizard 家族官方迁移未定，PBT 产物仅在官方正读数后晋级。

用法: python scripts/ccrn_pbt.py --gens 6 --pop 4
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
PY = "/data/home/wsx/miniconda3/envs/aivc2026/bin/python"
BASE = ROOT / "data/interim/bizard/ccrn"
PBT = ROOT / "data/interim/bizard/ccrn_pbt"
STATE = PBT / "pbt_state.json"

GRID = {
    "hidden": [[2048, 1024], [1024, 512], [4096, 1024], [3072, 1536]],
    "lr": [1.0e-3, 2.0e-3, 4.0e-3],
    "dropout": [0.0, 0.1, 0.3],
    "seed": [0, 1, 2],
}


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"workers": {}, "gen": 0, "history": []}


def save_state(s: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(s, indent=1))


def worker_dir(w: int) -> Path:
    d = PBT / f"w{w}"
    d.mkdir(parents=True, exist_ok=True)
    for f in ["X_train_ccrn.npy", "X_bench_hepg2_ccrn.npy", "meta_bench_hepg2.npz",
              "train_meta.npz", "fingerprints.npy"]:
        src, dst = BASE / f, d / f
        if src.exists() and not dst.exists():
            dst.symlink_to(os.path.relpath(src, d))
    return d


def write_cfg(w: int, hp: dict) -> Path:
    base = yaml.safe_load(open(ROOT / "configs/bizard/ccrn.yaml"))
    base["workdir"] = str((PBT / f"w{w}").relative_to(ROOT))
    base["model"]["hidden"] = hp["hidden"]
    base["model"]["dropout"] = hp["dropout"]
    base["train"]["lr"] = hp["lr"]
    base["train"]["seed"] = hp["seed"]
    p = PBT / f"ccrn_pbt_w{w}.yaml"
    p.write_text(yaml.safe_dump(base, sort_keys=False))
    return p


def run_eval_worker(w: int, hp: dict) -> dict:
    d = worker_dir(w)
    cfgp = write_cfg(w, hp)
    env = dict(os.environ, CCRN_CFG=str(cfgp.relative_to(ROOT)),
               TMPDIR="/data/resource/wanglab/aivc/tmpdir")
    t0 = time.time()
    for cmd in (["train", "--arm", "loco"], ["predict", "--src", "bench_hepg2", "--arm", "loco"]):
        r = subprocess.run([PY, str(ROOT / "src/models/bizard/ccrn.py"), *cmd],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            return {"hp": hp, "error": r.stderr[-500:], "reward": float("-inf")}
    rewards = {}
    for lam in (2.5, 4.0):
        tag = f"ccrnpbt_w{w}_lam{str(lam).replace('.', '')}"
        r = subprocess.run([PY, str(ROOT / "src/models/bizard/bench_bird.py"),
                            "--tag", tag, "--lam", str(lam), "--contexts", "Q_HepG2",
                            "--delta-dir", str(d),
                            "--delta-pattern", f"pred_ccrn1_loco_bench_hepg2.parquet"],
                           cwd=ROOT, env=env, capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if "overall=" in l]
        rewards[lam] = float(line[0].split("overall=")[1].split()[0]) if line else float("-inf")
    out = {"hp": hp, "rewards": rewards, "reward": max(rewards.values()),
           "sec": round(time.time() - t0)}
    return out


def sample_hp(rng) -> dict:
    import numpy as np

    def conv(v):
        v = v.item() if hasattr(v, "item") else v
        return list(v) if isinstance(v, (list, tuple)) else \
            (int(v) if isinstance(v, (np.integer, int)) else float(v))
    hp = {}
    for k, vals in GRID.items():
        pick = vals[int(rng.integers(len(vals)))]
        hp[k] = list(pick) if isinstance(pick, list) else conv(pick)
    return hp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=6)
    ap.add_argument("--pop", type=int, default=4)
    args = ap.parse_args()

    import numpy as np
    st = load_state()
    rng = np.random.default_rng(42)
    # 初始化种群（若 state 为空）
    for w in range(args.pop):
        if str(w) not in st["workers"]:
            hp = sample_hp(rng)
            hp["hidden"] = list(hp["hidden"])
            st["workers"][str(w)] = {"hp": hp, "reward": None}
    save_state(st)

    for g in range(st["gen"], st["gen"] + args.gens):
        print(f"=== gen {g} ===", flush=True)
        for w in range(args.pop):
            rec = st["workers"][str(w)]
            if rec.get("reward") is not None and rec.get("gen") == g:
                continue
            res = run_eval_worker(w, rec["hp"])
            rec["reward"] = res["reward"]
            rec["rewards"] = res.get("rewards")
            rec["gen"] = g
            rec["sec"] = res.get("sec")
            if "error" in res:
                rec["error"] = res["error"]
                print(f"[w{w}] FAILED: {res['error'][:200]}", flush=True)
                continue
            print(f"[w{w}] hp={rec['hp']} reward={rec['reward']:.4f} "
                  f"({res.get('sec', 0)}s)", flush=True)
            save_state(st)
        # exploit/explore：top-half 保留，bottom-half 继承 top 并扰动一维
        ranked = sorted(st["workers"].items(), key=lambda kv: -(kv[1]["reward"] or -9))
        top_hp = ranked[0][1]["hp"]
        for w, rec in ranked[len(ranked) // 2:]:
            hp = json.loads(json.dumps(top_hp))
            dim = list(GRID.keys())[int(rng.integers(len(GRID)))]
            pick = GRID[dim][int(rng.integers(len(GRID[dim])))]
            hp[dim] = list(pick) if isinstance(pick, list) else \
                (int(pick) if dim == "seed" else float(pick))
            rec["hp"], rec["reward"], rec["parent"] = hp, None, ranked[0][0]
            print(f"[explore w{w}] dim={dim} -> {hp[dim]} (parent {ranked[0][0]})", flush=True)
        st["gen"] = g + 1
        st["history"].append({"gen": g, "best": ranked[0][1]["reward"],
                              "best_hp": ranked[0][1]["hp"]})
        save_state(st)
    print("PBT complete. best:", json.dumps(st["history"][-1], indent=1), flush=True)


if __name__ == "__main__":
    main()
