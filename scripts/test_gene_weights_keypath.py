#!/usr/bin/env python
"""gene_weights 键路径单元测试（独立脚本 + asserts，pytest 未安装）。

覆盖: Σw·Δ/Σw 数学、表外基因中性 w=1.0、w clip [0,1]（REV C4 上限）、
sources 子集、与 dataset_weights 乘法组合、mode 守卫、相对/绝对路径、report 块。
临时目录合成签名 + monkeypatch ROOT，不触碰真实 data/。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.models.l1_signature.make_predictions as mp  # noqa: E402


def write_sigs(tmp: Path):
    sig = tmp / "data/interim/signatures"
    sig.mkdir(parents=True)
    # dsA: T1g1=2.0 T1g2=1.0 T2g1=0.5 ; dsB: T1g1=-1.0 T1g2=0.4
    dsA = pd.DataFrame({"target_gene": ["T1", "T1", "T2"],
                        "gene": ["g1", "g2", "g1"],
                        "logfc": [2.0, 1.0, 0.5]})
    dsB = pd.DataFrame({"target_gene": ["T1", "T1"],
                        "gene": ["g1", "g2"],
                        "logfc": [-1.0, 0.4]})
    dsA.to_parquet(sig / "dsA.parquet", index=False)
    dsB.to_parquet(sig / "dsB.parquet", index=False)
    return dsA, dsB


def base_cfg(datasets, **extra):
    cfg = {"datasets": datasets, "lambda_generic": 0.15, "max_abs_delta": 4.0}
    cfg.update(extra)
    return cfg


def run(tmp, cfg, panel, targets):
    old = mp.ROOT
    mp.ROOT = tmp
    try:
        return mp.load_delta(cfg, panel, targets)
    finally:
        mp.ROOT = old


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) < tol


def main() -> int:
    checks = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        write_sigs(tmp)
        panel = pd.Index(["g1", "g2"])
        targets = ["T1", "T2"]

        # 1. no-key 基线（等权均值）
        real, rep = run(tmp, base_cfg(["dsA", "dsB"]), panel, targets)
        v = {(r.target_gene, r.gene): r.delta for r in real.itertuples()}
        assert abs(v[("T1", "g1")] - 0.5) < 1e-12 and abs(v[("T1", "g2")] - 0.7) < 1e-12
        assert abs(v[("T2", "g1")] - 0.5) < 1e-12
        assert "gene_weights" not in rep and rep["dataset_weights"] is None
        checks.append("no-key baseline: 等权均值 OK, report 无 gene_weights 键")

        # 2. gene_weights 全表（含 clip: 1.7→1.0, -0.2→0.0; 表外 (dsB,g?)→1.0）
        wt = pd.DataFrame({"source": ["dsA", "dsB", "dsA", "dsB"],
                           "gene": ["g1", "g1", "g2", "g2"],
                           "w": [0.5, 1.0, 1.7, -0.2]})
        wt.to_parquet(tmp / "wtab.parquet", index=False)
        cfg = base_cfg(["dsA", "dsB"],
                       gene_weights={"file": "wtab.parquet", "mode": "pre_significance"})
        real, rep = run(tmp, cfg, panel, targets)
        v = {(r.target_gene, r.gene): r.delta for r in real.itertuples()}
        # T1g1: (0.5·2 + 1·(-1))/1.5 = 0;  T1g2: (1.0·1 + 0.0·0.4)/1.0 = 1.0
        # T2g1: 单源加权均值不变 = 0.5
        assert close(v[("T1", "g1")], 0.0) and close(v[("T1", "g2")], 1.0)
        assert close(v[("T2", "g1")], 0.5)
        assert rep["gene_weights"]["table_rows"] == 4
        assert rep["gene_weights"]["sources"] == ["dsA", "dsB"]
        checks.append("Σw·Δ/Σw 加权 + clip[0,1] + 表外中性 + report 块 OK")

        # 3. sources 子集: 表含 dsA 行但 sources=[dsB] → dsA 权重 1.0
        wt2 = pd.DataFrame({"source": ["dsA", "dsB"], "gene": ["g1", "g1"],
                            "w": [0.1, 0.5]})
        wt2.to_parquet(tmp / "wtab2.parquet", index=False)
        cfg = base_cfg(["dsA", "dsB"],
                       gene_weights={"file": "wtab2.parquet", "sources": ["dsB"],
                                     "mode": "pre_significance"})
        real, _ = run(tmp, cfg, panel, targets)
        v = {(r.target_gene, r.gene): r.delta for r in real.itertuples()}
        # T1g1: (1·2.0 + 0.5·(-1))/1.5 = 1.0
        assert close(v[("T1", "g1")], 1.0)
        checks.append("sources 子集（未列源 w=1.0）OK")

        # 4. 与 dataset_weights 乘法组合: 有效权重 = ds_w × gene_w
        cfg = base_cfg(["dsA", "dsB"], dataset_weights={"dsA": 3.0},
                       gene_weights={"file": "wtab.parquet", "mode": "pre_significance"})
        real, _ = run(tmp, cfg, panel, targets)
        v = {(r.target_gene, r.gene): r.delta for r in real.itertuples()}
        # T1g1: dsA 有效 3.0·0.5=1.5, dsB 1.0 → (1.5·2 + 1·(-1))/2.5 = 0.8
        assert close(v[("T1", "g1")], 0.8)
        checks.append("dataset_weights × gene_weights 行级乘法组合 OK")

        # 5. mode 守卫
        try:
            run(tmp, base_cfg(["dsA"], gene_weights={"file": "wtab.parquet",
                                                     "mode": "post"}), panel, targets)
            raise AssertionError("mode!=pre_significance 应 raise")
        except ValueError:
            pass
        checks.append("mode 守卫 ValueError OK")

        # 6. 绝对路径
        real, rep = run(tmp, base_cfg(["dsA"], gene_weights={
            "file": str(tmp / "wtab.parquet"), "mode": "pre_significance"}),
            panel, targets)
        v = {(r.target_gene, r.gene): r.delta for r in real.itertuples()}
        # dsA 单源: 加权均值不变形 (0.5·2)/0.5=2.0, (1.7→1.0·1)/1=1.0
        assert close(v[("T1", "g1")], 2.0) and close(v[("T1", "g2")], 1.0)
        checks.append("绝对路径解析 OK")

        # 7. w=0 源剔除语义: (dsA,g1,w=0) → T1g1 只由 dsB 决定
        wt3 = pd.DataFrame({"source": ["dsA", "dsB"], "gene": ["g1", "g1"],
                            "w": [0.0, 1.0]})
        wt3.to_parquet(tmp / "wtab3.parquet", index=False)
        real, _ = run(tmp, base_cfg(["dsA", "dsB"], gene_weights={
            "file": "wtab3.parquet", "mode": "pre_significance"}), panel, targets)
        v = {(r.target_gene, r.gene): r.delta for r in real.itertuples()}
        assert close(v[("T1", "g1")], -1.0)
        checks.append("w=0 剔除源 OK")

    print("PASS — %d 组断言全过:" % len(checks))
    for c in checks:
        print("  -", c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
