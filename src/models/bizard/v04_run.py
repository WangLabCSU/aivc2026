"""Bizard v0.4: PBT 代际扩展 + 表征级变异深化（Expert J v0.4 落地）。

四件事（时间盒 150 min）:
  1) 加权 soft-rank 定标   --scan-agg + --bench --tag bz04_w{w}
     8 鸟池, heu_dual 权重 w ∈ {0, 0.05, 0.10, 0.25}, 其余 7 鸟均摊 (1-w)/7。
  2) 跨骨干杂交 Δ̂ 插值    --fusion-preds + --hybrid + --bench --tag bz04_hybf{w}
     hybrid = w·heu_dual + (1-w)·fusion_s2（v35 stage2 fused = 0.5·xgb_s2_35 + 0.5·mlp_s2_35_noemb,
     官方 fusion v3.5 同族）, w ∈ {0.2, 0.3, 0.5}。
  3) 第 5/6 骨干族        --confv2-preds / --l1sig-preds + --bench --tag bz04_{confv2,l1sig}
     confv2: dualsrc-deep 启发式 × L1.8 置信度 v2 w_j（clip 前, 含 generic 行）;
     l1sig: gwps 单源启发式（最小对照）。
  4) gen2 池 + heu 混编   --scan-agg --pool gen2 --w-heu 0.10（时间允许）

用法:
  python src/models/bizard/v04_run.py --scan-agg [--pool gen1] [--w-heu 0 0.05 0.10 0.25]
  python src/models/bizard/v04_run.py --fusion-preds [--contexts ...]
  python src/models/bizard/v04_run.py --hybrid [--w-heu 0.2 0.3 0.5]
  python src/models/bizard/v04_run.py --confv2-preds --l1sig-preds
  python src/models/bizard/v04_run.py --bench --tag bz04_w0.05 --delta scan_w0.05_Q_HepG2.parquet \
      --contexts Q_HepG2
  python src/models/bizard/v04_run.py --report
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
sys.path.insert(0, str(ROOT))

BENCH_OUT = ROOT / "outputs/metrics/local_bench"
V03 = ROOT / "data/interim/bizard/v03"
V04 = ROOT / "data/interim/bizard/v04"
V35 = ROOT / "data/interim/l15_fusion/v35"
LAM = 2.5
ALL_CTX = ["P_H1", "Q_HepG2", "R_Jurkat", "S_RPE1", "S_HCT116"]

GEN1 = ["heu_dual", "xgb_s235", "mlp_s235", "v02_b3r",
        "xgb_sigdrop", "mlp_sigsub", "xgb_rowboot", "mlp_mse"]
GEN2 = ["g2_c1", "g2_c2", "g2_c3", "g2_c4"]
HEU_W_TAG = {0.0: "w0", 0.05: "w0.05", 0.1: "w0.1", 0.25: "w0.25"}
HYB_W_TAG = {0.2: "hybf0.2", 0.3: "hybf0.3", 0.5: "hybf0.5"}

V03_CFG = ROOT / "configs/bizard/v03.yaml"


def _cfg03() -> dict:
    import src.models.bizard.swarm as swarm
    swarm.CFG_PATH = V03_CFG
    return swarm.load_cfg()


def pool_weights(pool: str, w_heu: float) -> dict[str, float]:
    if pool == "gen1":
        others = [b for b in GEN1 if b != "heu_dual"]
        return {"heu_dual": w_heu, **{b: (1 - w_heu) / len(others) for b in others}}
    # gen2: heu 插入 4 子代池
    return {"heu_dual": w_heu, **{b: (1 - w_heu) / len(GEN2) for b in GEN2}}


def _load_grid(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    return df[["target_gene", "gene"]], df["delta"].to_numpy(np.float32)


def _write_delta(base: pd.DataFrame, delta: np.ndarray, fp: Path) -> None:
    fp.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target_gene": base["target_gene"], "gene": base["gene"],
                  "delta": delta}).to_parquet(fp)


def _blend(src_dirs: list[Path], mix: dict[str, float], fp: Path, ctx: str) -> None:
    """Δ̂ 级线性插值（网格严格对齐断言）。父代 pred 依次在 src_dirs 中查找。"""
    if fp.exists():
        return
    base, acc = None, None
    for name, w in mix.items():
        path = next((d / f"pred_{name}_{ctx}.parquet" for d in src_dirs
                     if (d / f"pred_{name}_{ctx}.parquet").exists()), None)
        if path is None:
            raise FileNotFoundError(f"pred_{name}_{ctx} 在 {src_dirs} 均缺")
        g, d = _load_grid(path)
        if base is None:
            base, acc = g, w * d.astype(np.float64)
        else:
            assert len(g) == len(base) and (g["target_gene"].to_numpy()
                                            == base["target_gene"].to_numpy()).all() \
                and (g["gene"].to_numpy() == base["gene"].to_numpy()).all(), \
                f"{name}/{ctx} 网格不对齐"
            acc += w * d.astype(np.float64)
    _write_delta(base, np.clip(acc, -4.0, 4.0).astype(np.float32), fp)
    print(f"{fp.name}: rows={len(acc)} absmean={np.abs(acc).mean():.4f}", flush=True)


# ----------------------------------------------------------------- 1) w scan
def do_scan_agg(pool: str, ws: list[float], contexts: list[str]) -> None:
    src = [V03] if pool == "gen1" else [V03 / "gen2", V03]   # gen2 鸟在 gen2/, heu_dual 在 v03/
    for w in ws:
        mix = pool_weights(pool, w)
        tag = HEU_W_TAG[w]
        for ctx in contexts:
            fp = V04 / f"scan_{pool}_{tag}_{ctx}.parquet"
            if fp.exists():
                continue
            t0 = time.time()
            if w == 0.0:   # soft-rank 加权池 w=0 → 该鸟权重 0, 等价 7 鸟池
                mix = {k: v for k, v in mix.items() if v > 0}
            base, preds, names = None, [], []
            for name in mix:
                path = next((d / f"pred_{name}_{ctx}.parquet" for d in src
                             if (d / f"pred_{name}_{ctx}.parquet").exists()), None)
                if path is None:
                    raise FileNotFoundError(f"pred_{name}_{ctx} 在 {src} 均缺")
                g, d = _load_grid(path)
                if base is None:
                    base = g
                assert len(g) == len(base) and (g["target_gene"].to_numpy()
                                                == base["target_gene"].to_numpy()).all() \
                    and (g["gene"].to_numpy() == base["gene"].to_numpy()).all()
                preds.append(d)
                names.append(name)
            import src.models.bizard.swarm as swarm
            agg = swarm.aggregate(preds, method="soft_rank",
                                  weights=np.array([mix[n] for n in names]))
            _write_delta(base, agg, fp)
            print(f"{fp.name}: birds={len(preds)} w_heu={w} "
                  f"absmean={np.abs(agg).mean():.4f} ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------- 2) fusion 骨干
def do_fusion_preds(contexts: list[str]) -> None:
    """fusion_s2 鸟: alpha 0.5 = 0.5·xgb_s2_35 + 0.5·mlp_s2_35_noemb（v35 reuse, 与 v03 同链路）。"""
    import src.models.bizard.swarm as swarm
    cfg = _cfg03()
    fcfg = swarm.fusion_cfg(cfg)
    fcfg = json.loads(json.dumps(fcfg))
    fcfg["xgb"]["n_jobs"] = 24
    from src.models.l15_fusion.common import build_vocab
    from src.models.l15_fusion.predict_loco import predict_mlp, predict_xgb
    vocab = build_vocab(fcfg)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bdir = V35
    for ctx in contexts:
        fp = V04 / f"pred_fusion_s2_{ctx}.parquet"
        if fp.exists():
            print(f"{fp.name} exists, skip", flush=True)
            continue
        t0 = time.time()
        fp_feat = ROOT / cfg["v35_workdir"] / f"feat_bench_{ctx}.npz"
        if not fp_feat.exists():   # bench 特征回退 v01 缓存（v35 布局一致, 同 v03_run）
            fp_feat = ROOT / "data/interim/bizard/v01" / fp_feat.name
        z = np.load(fp_feat, allow_pickle=False)
        X = z["X"]
        tgt, gen = z["target"].astype(str), z["gene"].astype(str)
        px = predict_xgb(fcfg, bdir, X, 2, model_tag="s2_35")
        gpos = vocab.get_indexer(pd.Index(gen)).astype(np.int64)
        tpos = vocab.get_indexer(pd.Index(tgt)).astype(np.int64)
        pm = predict_mlp(fcfg, bdir, X, 2, gpos, tpos, dev, pt_tag="s2_35_noemb")
        delta = np.clip(np.nan_to_num(0.5 * px + 0.5 * pm, nan=0.0), -4, 4).astype(np.float32)
        del z, X
        _write_delta(pd.DataFrame({"target_gene": tgt, "gene": gen}), delta, fp)
        print(f"{fp.name}: rows={len(delta)} absmean={np.abs(delta).mean():.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)


def do_hybrid(ws: list[float], contexts: list[str]) -> None:
    for w in ws:
        for ctx in contexts:
            if not (V04 / f"pred_fusion_s2_{ctx}.parquet").exists():
                raise FileNotFoundError(f"fusion_s2/{ctx} 缺, 先 --fusion-preds")
            fp = V04 / f"pred_{HYB_W_TAG[w]}_{ctx}.parquet"
            _blend([V04, V03], {"heu_dual": w, "fusion_s2": 1 - w}, fp, ctx)


# ----------------------------------------------------------- 3) 骨干 5/6 preds
def _heu_grid(datasets: list[str], conf: bool) -> callable:
    """构造 grid Δ̂ 函数: dualsrc 启发式（复用 v03 缓存）× 可选 conf v2 w_j。"""
    import yaml
    import src.models.bizard.swarm as swarm
    swarm.CFG_PATH = V03_CFG
    cfg = swarm.load_cfg()
    hcfg = yaml.safe_load(open(ROOT / "configs/l1_signature/dualsrc-deep-confv2.yaml" if conf
                               else ROOT / cfg["heuristic_cfg"]))
    hcfg["datasets"] = datasets
    parts = {}
    for name in datasets:
        p = V03 / f"heu_agg_{name}.parquet"
        assert p.exists(), f"缺缓存 {p}"
        parts[name] = pd.read_parquet(p)
    gz = parts.get("gwps")
    generic = gz.groupby("gene", sort=False)["logfc"].mean() if gz is not None else None
    w = None
    if conf:
        wv = pd.read_parquet(ROOT / "data/interim/confidence_v2.parquet")
        w = pd.Series(wv["w"].to_numpy(np.float32), index=wv["gene"].astype(str))

    def grid_delta(target: np.ndarray, gene: np.ndarray) -> np.ndarray:
        # 与 v03_run.heuristic_delta 相同语义, 但 hcfg.datasets 受控 + clip 前 ×w_j
        tset, panel = set(np.unique(target.astype(str))), set(np.unique(gene.astype(str)))
        covered, parts_f = set(), []
        for g in parts.values():
            gf = g[g["target_gene"].isin(tset) & g["gene"].isin(panel)]
            covered |= set(gf["target_gene"])
            parts_f.append(gf.rename(columns={"logfc": "delta"})[
                ["target_gene", "gene", "delta"]])
        real = pd.concat(parts_f, ignore_index=True).groupby(
            ["target_gene", "gene"], sort=False)["delta"].mean().reset_index()
        uncovered = [t for t in tset if t not in covered]
        lam_g = float(hcfg.get("lambda_generic") or 0)
        if uncovered and generic is not None and lam_g > 0:
            gen = generic[generic.index.astype(str).isin(panel)]
            real = pd.concat([real, pd.DataFrame({
                "target_gene": np.repeat(uncovered, len(gen)),
                "gene": np.tile(gen.index.astype(str).to_numpy(), len(uncovered)),
                "delta": np.tile(gen.to_numpy(), len(uncovered))})], ignore_index=True)
        if w is not None:   # conf v2: clip 前 ×w_j（表外基因 0.7）
            real["delta"] = real["delta"].to_numpy() * real["gene"].astype(str).map(w).fillna(0.7).to_numpy()
        real["delta"] = real["delta"].clip(-float(hcfg["max_abs_delta"]),
                                           float(hcfg["max_abs_delta"]))
        key = pd.MultiIndex.from_arrays([target.astype(str), gene.astype(str)])
        dkey = pd.MultiIndex.from_arrays([real["target_gene"].astype(str),
                                          real["gene"].astype(str)])
        dser = pd.Series(real["delta"].to_numpy(np.float64), index=dkey)
        dser = dser[~dser.index.duplicated(keep="first")]
        return np.nan_to_num(dser.reindex(key).to_numpy(np.float32), nan=0.0)
    return grid_delta


def do_backbone_preds(name: str, datasets: list[str], conf: bool,
                      contexts: list[str], submit: bool = False) -> None:
    grid_delta = _heu_grid(datasets, conf)
    cfg = _cfg03()
    jobs = [(c, ROOT / cfg["v35_workdir"] / f"feat_bench_{c}.npz") for c in contexts]
    for ctx, fp_feat in jobs:
        if not fp_feat.exists():
            fp_feat = ROOT / "data/interim/bizard/v01" / fp_feat.name
        fp = V04 / f"pred_{name}_{ctx}.parquet"
        if fp.exists():
            print(f"{fp.name} exists, skip", flush=True)
            continue
        t0 = time.time()
        z = np.load(fp_feat, allow_pickle=False)
        dv = grid_delta(z["target"], z["gene"])
        _write_delta(pd.DataFrame({"target_gene": z["target"].astype(str),
                                   "gene": z["gene"].astype(str)}), dv, fp)
        del z
        print(f"{fp.name}: rows={len(dv)} absmean={np.abs(dv).mean():.4f} "
              f"covered={int((dv != 0).sum())} ({time.time()-t0:.0f}s)", flush=True)


# ------------------------------------------------------------------- bench
def bench(tag: str, delta_path: Path, contexts: list[str], lam: float = LAM) -> None:
    import src.models.bizard.bench_bird as bb
    outdir = BENCH_OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    per_ctx: dict[str, float] = {}
    per_metric: dict[str, dict] = {}
    sm = outdir / "bizard_summary.json"
    if sm.exists():
        prev = json.loads(sm.read_text())
        per_ctx.update(prev.get("per_ctx") or {})
        per_metric.update(prev.get("per_metric") or {})
    for ctx in contexts:
        if ctx in per_ctx:
            print(f"[{tag}/{ctx}] already {per_ctx[ctx]:.4f}, skip", flush=True)
            continue
        t0 = time.time()
        delta = pd.read_parquet(delta_path)
        pp = bb.build_pred(ctx, delta, lam, outdir, seed=0, n_cells=400, max_fold=8.0)
        print(f"[{tag}/{ctx}] pred built ({time.time()-t0:.0f}s)", flush=True)
        bb.run_celleval(ctx, pp, outdir)
        res = bb.score_ctx(tag, ctx)
        per_ctx[ctx] = res["overall"]
        per_metric[ctx] = res.get("per_metric") or {}
        print(f"[{tag}/{ctx}] overall={res['overall']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        bb.save_json({"tag": tag, "lam": lam, "per_ctx": dict(per_ctx),
                      "per_metric": per_metric}, sm)


# ------------------------------------------------------------------ report
def do_report() -> dict:
    out = {"w_scan_gen1": {}, "hybrid": {}, "backbone_new": {}, "gen2_mix": {},
           "refs": {}}
    for w, tag in HEU_W_TAG.items():
        p = BENCH_OUT / f"bz04_{tag}" / "bizard_summary.json"
        if p.exists():
            out["w_scan_gen1"][str(w)] = json.loads(p.read_text()).get("per_ctx")
    for w, tag in HYB_W_TAG.items():
        p = BENCH_OUT / f"bz04_{tag}" / "bizard_summary.json"
        if p.exists():
            out["hybrid"][str(w)] = json.loads(p.read_text()).get("per_ctx")
    for b in ("confv2", "l1sig"):
        p = BENCH_OUT / f"bz04_{b}" / "bizard_summary.json"
        if p.exists():
            out["backbone_new"][b] = json.loads(p.read_text()).get("per_ctx")
    p = BENCH_OUT / "bz04_g2heu_w0.1" / "bizard_summary.json"
    if p.exists():
        out["gen2_mix"]["0.1"] = json.loads(p.read_text()).get("per_ctx")
    for k, t in (("gen1_equal", "bz03_agg_lam2.5"), ("gen1_w0.25_v03", "bz03_aggw_lam2.5"),
                 ("gen2_agg", "bz03g2_agg_lam2.5"), ("v01_agg_sr", "bz01_agg_sr"),
                 ("g2_c3", "bz03g2_g2_c3")):
        p = BENCH_OUT / t / "bizard_summary.json"
        if p.exists():
            d = json.loads(p.read_text())
            out["refs"][k] = d.get("per_ctx") or {
                c: v["overall"] for c, v in (d.get("per_context") or {}).items()}
    fp = ROOT / "outputs/metrics/20261006_bizard_v04.json"
    fp.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-agg", action="store_true")
    ap.add_argument("--fusion-preds", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--confv2-preds", action="store_true")
    ap.add_argument("--l1sig-preds", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--pool", default="gen1", choices=["gen1", "gen2"])
    ap.add_argument("--w-heu", default="0,0.05,0.1,0.25")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--delta", default=None)
    ap.add_argument("--contexts", default="Q_HepG2")
    ap.add_argument("--lam", type=float, default=LAM)
    args = ap.parse_args()
    ctxs = args.contexts.split(",")
    ws = [float(x) for x in args.w_heu.split(",")]
    if args.scan_agg:
        do_scan_agg(args.pool, ws, ctxs)
    if args.fusion_preds:
        do_fusion_preds(ctxs)
    if args.hybrid:
        do_hybrid(ws, ctxs)
    if args.confv2_preds:
        do_backbone_preds("confv2", ["gwps", "hct116_all"], True, ctxs)
    if args.l1sig_preds:
        do_backbone_preds("l1sig", ["gwps"], False, ctxs)
    if args.bench:
        assert args.tag and args.delta
        bench(args.tag, V04 / args.delta, ctxs, args.lam)
    if args.report:
        do_report()


if __name__ == "__main__":
    main()
