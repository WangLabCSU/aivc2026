#!/usr/bin/env python
"""calib_append_official.py —— 官方点回填仪器: log FINAL_STATUS → OFFICIAL_POINTS → refit。

分析侧仪器: scripts/official_alignment.py 与 docs/research/official_alignment_v1.json
均为 bench 对齐层 (src/ 零引用); 生产提交链 (configs/bizard/v02.yaml 冻结 affine、
src/models/bizard/*) 与本脚本零交集。不 git commit, 不读大数据文件。

用法:
  python scripts/calib_append_official.py --log logs/submit_lam47_1014.log \
      --stem 20261014_bizard_v02_lam47 --local bzbiz10skip_lam47

流程:
  1. 解析 --log 的最后一个 FINAL_STATUS 块 (status 必须 published)
  2. 幂等追加 (family, stem) 官方点进 OFFICIAL_POINTS (原地文本编辑, /tmp 备份,
     编辑后 compile 校验失败即回滚)
  3. fit_calib() refit (重写 docs/research/official_alignment_v1.json)
  4. 打印: 追加点回读 / 新旧 per-metric 系数对比 / 逐指标 leave-one-out 误差 /
          追加前旧映射对该点 local 的 OOS 预测 / 新拟合 5 发布点回代 (in-sample) /
          overall 级 (local avg_score → official Overall) 5 点拟合 + LOO
          + 对 local=0.3455 的官方期望预测 (含 λ39 文档冲突敏感性)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
TARGET = ROOT / "scripts/official_alignment.py"
CALIB = ROOT / "docs/research/official_alignment_v1.json"

# bizard 家族已发布点的 overall 级数据 (local avg_score → official Overall)。
# λ39/λ55 无 FINAL_STATUS 日志: λ39 取机器快照 20261008_bizard_v10_online.json
# (entry 1I1bUHfcecnbACv4X0Bc, rank 884, 0.05069); iter021/rev_R1 文档引 0.0886
# 与启发式家族 λ6.5 官方分同值, 判为编号混淆, 以机器快照为准 (报告已标注)。
# λ55 取 iter020 台账 0.0330 (rank 925)。λ47/λ37/λ35 以 logs FINAL_STATUS 为准。
# 注: local tag 名 lam16 为历史命名, 对应 artifact 实为 λ39。
OVERALL_POINTS = [
    ("bzbiz10skip_lam16", 0.050687944985472105, 39,
     "20261008_bizard_v10_online.json entry 1I1bUHfcecnbACv4X0Bc rank884 (文档冲突: iter021/rev_R1 引 0.0886)"),
    ("bzbiz10skip_lam55", 0.0330, 55, "rank925, iter020 台账 (无 FINAL_STATUS 日志)"),
    ("bzbiz10skip_lam47", 0.04565493138760657, 47, "entry BWy91Q4UXdusW6QcxjCU, submit_lam47_1014.log"),
    ("bzbiz10skip_lam37", 0.0502137177306288, 37, "entry SgXJeIQwv7taWP4B0GND, submit_lam37_1015.log"),
    ("bzbiz10skip_lam35", 0.049121598951417295, 35, "entry 58kzxoM9gGrF1QOdXS5k, submit_lam35_1015.log"),
]

NUM = r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
FIELDS = {
    "entry": re.compile(r"^\s+entry:\s+(\S+)\s*$", re.M),
    "model": re.compile(r"^\s+model:\s+(\S+)\s*$", re.M),
    "status": re.compile(r"^\s+status:\s+(\S+)\s*$", re.M),
    "submitted": re.compile(r"^\s+submitted:\s*(\S+)\s*$", re.M),
    "rank": re.compile(r"^\s+Rank:\s+(\d+)\s*$", re.M),
    "overall": re.compile(r"^\s+Overall:\s+" + NUM + r"\s*$", re.M),
    "pds": re.compile(r"^\s+Perturbation discrimination pds:\s+" + NUM + r"\s*$", re.M),
    "mse": re.compile(r"^\s+Expression accuracy mse:\s+" + NUM + r"\s*$", re.M),
    "nmae": re.compile(r"^\s+DE log-FC accuracy nmae:\s+" + NUM + r"\s*$", re.M),
    "fid": re.compile(r"^\s+DE direction fidelity fid:\s+" + NUM + r"\s*$", re.M),
    "reach": re.compile(r"^\s+DE direction reach reach:\s+" + NUM + r"\s*$", re.M),
    "jac": re.compile(r"^\s+DE significance overlap jac:\s+" + NUM + r"\s*$", re.M),
}


def parse_final_status(log: Path) -> dict:
    text = log.read_text()
    idx = text.rfind("FINAL_STATUS:")
    if idx < 0:
        sys.exit(f"[stop] {log} 无 FINAL_STATUS 块")
    block = text[idx:]
    out = {}
    for k, rx in FIELDS.items():
        m = rx.search(block)
        if not m:
            sys.exit(f"[stop] FINAL_STATUS 缺字段 {k}")
        out[k] = m.group(1)
    return {
        "entry": out["entry"], "model": out["model"], "status": out["status"],
        "submitted": out["submitted"], "rank": int(out["rank"]),
        "overall": float(out["overall"]),
        "metrics": {m: float(out[m]) for m in ("pds", "mse", "nmae", "fid", "reach", "jac")},
    }


def load_oa():
    spec = importlib.util.spec_from_file_location("official_alignment_live", TARGET)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def append_entry(family: str, stem: str, local: str, st: dict) -> bool:
    """幂等追加一条官方点; 返回是否发生了编辑。"""
    text = TARGET.read_text()
    key = f'("{family}", "{stem}")'
    if key in text:
        print(f"[append] {key} 已在表内 —— 幂等跳过")
        return False
    if st["entry"] in text:
        sys.exit(f"[stop] entry_id {st['entry']} 已存在于 official_alignment.py (异 stem), 需人工核查")
    m = re.search(r"OFFICIAL_POINTS = \{.*?^\}", text, re.S | re.M)
    if not m:
        sys.exit("[stop] 未定位到 OFFICIAL_POINTS 表尾")
    backup = Path(f"/tmp/official_alignment.py.bak.{time.strftime('%Y%m%d_%H%M%S')}")
    shutil.copy2(TARGET, backup)
    ov = st["metrics"]
    entry = (
        f'    {key}: dict(  # {st["model"]} entry {st["entry"]} submitted {st["submitted"]} '
        f'Rank {st["rank"]} Overall {st["overall"]:.6f}\n'
        f'        local="{local}", official={{"pds": {ov["pds"]:.4f}, "mse": {ov["mse"]:.1f}, '
        f'"nmae": {ov["nmae"]:.4f},\n'
        f'                                             "fid": {ov["fid"]:.4f}, "reach": {ov["reach"]:.4f}, '
        f'"jac": {ov["jac"]:.4f}}}),\n'
    )
    pos = m.end() - 1  # 收尾 '}' 前
    new_text = text[:pos] + entry + text[pos:]
    try:
        compile(new_text, str(TARGET), "exec")
    except SyntaxError as e:
        shutil.copy2(backup, TARGET)
        sys.exit(f"[abort] 编辑后语法校验失败, 已回滚: {e}")
    TARGET.write_text(new_text)
    print(f"[append] +1 点 → {key} (backup: {backup})")
    return True


def loo_errors(rows: list[dict], metrics: tuple[str, ...] | None = None) -> dict[str, tuple[float, float, int]]:
    """rows: fit_calib 的 points 行; 返回 {metric: (mean|err|, max|err|, n)}, n<3 跳过。"""
    names = sorted({r["metric"] for r in rows}) if metrics is None else metrics
    out = {}
    for m in names:
        pts = [(r["local_raw"], r["official_raw"]) for r in rows if r["metric"] == m]
        if len(pts) < 3:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        errs = []
        for i in range(len(xs)):
            keep = np.ones(len(xs), bool)
            keep[i] = False
            a, b = np.polyfit(xs[keep], ys[keep], 1)
            errs.append(abs(a * xs[i] + b - ys[i]))
        out[m] = (float(np.mean(errs)), float(np.max(errs)), len(pts))
    return out


def overall_level(args) -> None:
    """overall 级标定: local avg_score → official Overall, 5 个已发布点。"""
    mod = load_oa()
    xs, ys, labels, srcs = [], [], [], []
    for tag, official, lam, src in OVERALL_POINTS:
        la = mod.local_anchor(tag)
        if la is None:
            print(f"[warn] {tag} avg_score 缺失, overall 级拟合跳过该点")
            continue
        xs.append(la); ys.append(official); labels.append(f"λ{lam}"); srcs.append(src)
    x = np.array(xs); y = np.array(ys)
    a, b = np.polyfit(x, y, 1)
    pred = a * x + b
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot
    print("\n== overall 级拟合 (bizard, local avg_score → official Overall) ==")
    for lab, xi, yi, pi, sr in zip(labels, xs, ys, pred, srcs):
        print(f"  {lab:4s} local {xi:+.4f} → official {yi:+.4f}  (fit {pi:+.4f}, resid {yi - pi:+.4f})  [{sr}]")
    print(f"  official = {a:.6f}*local + {b:+.6f}   n={len(xs)}  R²={r2:.4f}")
    loo = loo_errors([{"metric": "overall", "local_raw": v, "official_raw": w}
                      for v, w in zip(xs, ys)])
    if loo:
        mean_e, max_e, n = loo["overall"]
        print(f"  LOO: mean|err| {mean_e:.4f}, max|err| {max_e:.4f} (n={n})")
    p = a * args.predict_local + b
    print(f"  预测 local={args.predict_local}: official 期望 = {p:+.4f}")
    # 敏感性: λ39 若按文档 0.0886
    y2 = y.copy(); y2[0] = 0.0886
    a2, b2 = np.polyfit(x, y2, 1)
    print(f"  [敏感性] 若 λ39=+0.0886 (文档值): official = {a2:.6f}*local + {b2:+.6f}, "
          f"local=0.3455 → {a2 * args.predict_local + b2:+.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="官方点回填: FINAL_STATUS → OFFICIAL_POINTS → refit")
    ap.add_argument("--log", required=True)
    ap.add_argument("--family", default="bizard")
    ap.add_argument("--stem", required=True, help="OFFICIAL_POINTS 键 stem, 如 20261015_bizard_v04_lam35")
    ap.add_argument("--local", required=True, help="本地 bench tag, 如 bzbiz10skip_lam35")
    ap.add_argument("--predict-local", type=float, default=0.3455,
                    help="overall 级映射对该本地分的官方期望预测 (默认 0.3455, v0.5 候选)")
    args = ap.parse_args()

    st = parse_final_status(Path(args.log))
    print("== FINAL_STATUS 解析 ==")
    print(f"  entry     {st['entry']}")
    print(f"  model     {st['model']}   status {st['status']}   rank {st['rank']}")
    print(f"  submitted {st['submitted']}")
    print(f"  Overall   {st['overall']:.9f}")
    for m, v in st["metrics"].items():
        print(f"  {m:6s}  {v:+.10f}")
    if st["status"] != "published":
        sys.exit("[stop] status 非 published, 不回填")

    # 追加前的旧系数 (来自上一次 fit 落盘的 json)
    old_json = json.loads(CALIB.read_text()) if CALIB.exists() else {}
    old_fam = old_json.get("family", {}).get(args.family, {})

    # 追加前旧映射对该提交 local 的 OOS 预测 (含该点则属 in-sample 回代)
    mod = load_oa()
    try:
        lr = mod.local_metric_anchors(args.local)
        old_preds = {m: f["a"] * v + f["b"] for m, v in lr.items()
                     if (f := old_fam.get(m))}
        if len(old_preds) >= 5:
            old_overall = float(np.mean(list(old_preds.values())))
            print(f"\n== 追加前旧映射对该点 (local={args.local}) 的预测 ==")
            for m in ("pds", "nmae", "fid", "reach", "jac"):
                if m in old_preds:
                    print(f"  {m:6s} pred {old_preds[m]:+.4f}")
            print(f"  overall_pred {old_overall:+.4f} vs 实际 {st['overall']:+.4f} "
                  f"(残差 {st['overall'] - old_overall:+.4f})")
    except FileNotFoundError as e:
        print(f"[skip] 本地 anchor 缺失: {e}")

    changed = append_entry(args.family, args.stem, args.local, st)

    mod = load_oa()
    fit = mod.fit_calib()
    new_fam = fit.get("family", {}).get(args.family, {})
    new_rows = [r for r in fit["points"] if r["family"] == args.family]
    old_rows = [r for r in old_json.get("points", []) if r["family"] == args.family]

    print(f"\n== {args.family} family per-metric 系数对比 (official = a·local + b) ==")
    print(f"  {'metric':6s} | {'old a':>10s} {'old b':>10s} {'n':>2s} | {'new a':>10s} {'new b':>10s} {'n':>2s} | {'Δa':>9s} {'Δb':>9s}")
    for m in ("pds", "nmae", "fid", "reach", "jac"):
        o, n_ = old_fam.get(m), new_fam.get(m)
        if not o and not n_:
            continue
        if o and n_:
            print(f"  {m:6s} | {o['a']:+10.6f} {o['b']:+10.6f} {o['n']:2d} | "
                  f"{n_['a']:+10.6f} {n_['b']:+10.6f} {n_['n']:2d} | "
                  f"{n_['a'] - o['a']:+.6f} {n_['b'] - o['b']:+.6f}")
        elif n_:
            print(f"  {m:6s} | {'—':>10s} {'—':>10s} {'—':>2s} | {n_['a']:+10.6f} {n_['b']:+10.6f} {n_['n']:2d} |")

    print("\n== leave-one-out 逐指标误差 (bizard family) ==")
    for name, rows in (("before", old_rows), ("after", new_rows)):
        loo = loo_errors(rows)
        if not loo:
            print(f"  {name:6s}: 点数不足 (n<3), LOO 不可算")
            continue
        parts = [f"{m}: mean {e0:.4f} / max {e1:.4f} (n={n})" for m, (e0, e1, n) in sorted(loo.items())]
        print(f"  {name:6s} (n_points={len(rows)}): " + "; ".join(parts))

    # 新拟合下 5 个已发布点的回代 (全部在拟合内 → in-sample, 仅作对照)
    print("\n== 新拟合回代: 5 个已发布点 per-metric 预测均值 vs 官方 Overall (in-sample) ==")
    for tag, official, lam, src in OVERALL_POINTS:
        try:
            lr = mod.local_metric_anchors(tag)
        except FileNotFoundError:
            print(f"  [skip] {tag} anchor 缺失")
            continue
        preds = {m: (f["a"] * v + f["b"]) for m, v in lr.items()
                 if (f := fit.get("family", {}).get(args.family, {}).get(m))}
        if len(preds) < 5:
            print(f"  [skip] {tag} 可预测指标不足 5")
            continue
        mean_pred = float(np.mean(list(preds.values())))
        print(f"  λ{lam:<3d} local {mod.local_anchor(tag):+.4f} → pred {mean_pred:+.4f} vs official {official:+.4f} (resid {official - mean_pred:+.4f})")

    overall_level(args)
    print(f"\n[done] changed={changed}, 表见 {TARGET.name}, 拟合落盘 {CALIB}")


if __name__ == "__main__":
    main()
