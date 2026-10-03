"""Bizard v0.1 核心：鸟定义、适应度、群飞聚合。

三层对应（docs/bizard/architecture.md）:
  L1 特征层  — 复用 v35 缓存（feat_train.npz 22 列 6 上下文池 + tahoe aux 行）
  L2 学习层  — 每鸟 = XGB + MLP(no-emb) stage2 式训练, 鸟间差异 = (主场景行权重 τ, seed, aux 权重)
  L3 群飞层  — 局部适应度 = 主场景 anchor 归一化 overall；
               全局适应度 = 五场景 affine overall 均值 − 0.15×场景min 惩罚；
               聚合 = 六鸟 Δ̂ 的 soft-rank average（v0.1 等权, 记录选择）
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
CFG_PATH = ROOT / "configs/bizard/v01.yaml"
SCENARIO_ORDER = ["P_H1", "Q_HepG2", "R_Jurkat", "S_RPE1", "S_HCT116"]


def load_cfg(path: Path | None = None) -> dict:
    with open(path or CFG_PATH) as f:
        return yaml.safe_load(f)


def workdir(cfg: dict) -> Path:
    d = ROOT / cfg["workdir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def fusion_cfg(cfg: dict) -> dict:
    """加载骨干 fusion 配置并固定 L15_CFG 环境变量（l15_fusion.common.load_cfg 全链一致）。"""
    import os
    p = ROOT / cfg["fusion_cfg"]
    os.environ["L15_CFG"] = str(p)
    import src.models.l15_fusion.common as fcommon
    fcommon.CFG_PATH = p
    with open(p) as f:
        return yaml.safe_load(f)


def birds(cfg: dict) -> list[dict]:
    """鸟册（config 驱动），补全派生字段。"""
    out = []
    for b in cfg["birds"]:
        b = dict(b)
        b["scenario"] = b.get("scenario")
        b["reuse"] = b.get("reuse")
        out.append(b)
    return out


def bird_tag(bird: dict) -> str:
    return bird["name"]


# ------------------------------------------------------------------- fitness
def affine(local_overall: float, fit: dict) -> float:
    """校准 v5: official_overall ≈ a·local − b（heuristic 家族标定, 同族内相对比较用）。"""
    v = fit["affine_a"] * float(local_overall) - fit["affine_b"]
    lo, hi = fit["clip"]
    return float(np.clip(v, lo, hi))


def local_fitness(per_ctx_overall: dict[str, float]) -> float:
    """局部适应度 = 主场景 anchor 归一化 overall（bench_score bundle 标度）。"""
    if not per_ctx_overall:
        return float("nan")
    return float(np.mean(list(per_ctx_overall.values())))


def global_fitness(per_ctx_overall: dict[str, float], fit: dict) -> dict:
    """全局适应度 = 五场景 affine overall 均值 − min_penalty × max(0, −min_ctx affine)。

    场景min 惩罚: 任一场景 affine 后仍在官方零线下（fid 型坍塌前兆）→ 线性扣分。
    per_ctx_overall 覆盖不足五场景时按可用场景计（partial=True 标注）。
    affine_mean_raw: 未 clip 的 affine 均值（诊断用——clip 上限 +0.10 会使
    local>0.30 的鸟在排名奖励上饱和, 原始值保留区分度）。
    """
    vals = {c: affine(v, fit) for c, v in per_ctx_overall.items()}
    if not vals:
        return {"global": float("nan"), "affine_mean": float("nan"),
                "affine_mean_raw": float("nan"),
                "min_ctx": None, "min_penalty": 0.0, "partial": True}
    m = float(np.mean(list(vals.values())))
    raw = float(np.mean([fit["affine_a"] * v - fit["affine_b"]
                         for v in per_ctx_overall.values()]))
    mn = float(min(vals.values()))
    pen = fit["min_penalty"] * max(0.0, -mn)
    return {"global": m - pen, "affine_mean": m, "affine_mean_raw": raw, "min_ctx": mn,
            "min_penalty": pen, "partial": len(vals) < len(SCENARIO_ORDER)}


# ---------------------------------------------------------------- aggregation
def soft_rank_average(preds: list[np.ndarray], weights: np.ndarray | None = None,
                      reference: np.ndarray | None = None) -> np.ndarray:
    """soft-rank average: 各鸟 Δ̂ → 自身经验分位 →（加权）均值分位 → 参照分布分位值映射回 Δ̂ 标度。

    参照分布 = 六鸟 Δ̂ 合并池（reference=None 时自动）; 等权 v0.1（weights=None）。
    每鸟 pred 必须行对齐（同 target×gene 网格）。
    """
    P = np.stack([p.astype(np.float64) for p in preds])  # (B, N)
    B, N = P.shape
    w = np.ones(B) if weights is None else np.asarray(weights, np.float64)
    w = w / w.sum()
    # 每鸟经验 CDF（平均秩, 处理并列）
    q = np.empty((B, N), np.float64)
    for b in range(B):
        order = np.argsort(P[b], kind="mergesort")
        ranks = np.empty(N, np.float64)
        sp_ = P[b][order]
        # 平均秩: 同值段共享 (i+j+1)/2 位次
        i = 0
        r = np.arange(1, N + 1, dtype=np.float64)
        while i < N:
            j = i
            while j + 1 < N and sp_[j + 1] == sp_[i]:
                j += 1
            r[i:j + 1] = (i + j + 2) / 2.0
            i = j + 1
        ranks[order] = r
        q[b] = ranks / (N + 1.0)
    qm = (w[:, None] * q).sum(axis=0)
    ref = P.reshape(-1) if reference is None else np.asarray(reference, np.float64)
    ref = np.sort(ref[np.isfinite(ref)])
    # 分位值映射: qm → ref 的 (qm·(len-1)) 分位
    pos = qm * (len(ref) - 1)
    lo = np.floor(pos).astype(np.int64)
    hi = np.minimum(lo + 1, len(ref) - 1)
    frac = pos - lo
    return (ref[lo] * (1 - frac) + ref[hi] * frac).astype(np.float32)


def simple_average(preds: list[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    P = np.stack([p.astype(np.float64) for p in preds])
    if weights is None:
        return P.mean(axis=0).astype(np.float32)
    w = np.asarray(weights, np.float64)
    w = w / w.sum()
    return (w[:, None] * P).sum(axis=0).astype(np.float32)


def aggregate(preds: list[np.ndarray], method: str = "soft_rank",
              weights: np.ndarray | None = None) -> np.ndarray:
    if method == "soft_rank":
        return soft_rank_average(preds, weights)
    if method == "simple":
        return simple_average(preds, weights)
    raise ValueError(method)


# ---------------------------------------------------------------------- misc
def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
