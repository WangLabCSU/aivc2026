#!/usr/bin/env python
"""v2.2 ④source-LOPO 执行脚本（fid/源质量加权线，PI 放行顺序 ④；读数定义 = C1 冻结文档 §8）。

输入  data/interim/fid_srcquality/gene_source_votes.parquet     （③ 导出，7 源全量含 norman_all）
      data/interim/fid_srcquality/source_context_weights.parquet（② 冻结 ρ 表，sha256 锚定）
输出  data/interim/fid_srcquality/source_lopo_report.json

判据（C1 冻结，预注册，不得因读数调整）：
  - 投票池（REV C2）= {gwps, hct116_all, jurkat_all, k562_ess_all}；norman_all 剔除（导出仍保留）；
  - 留出源 rpe1_all / hepg2_all（basal ctx = rpe1 / hepg2），只作验证永不进共识；
  - eval 单元 = (target, gene)：留出源 |logFC| > 0.25 且 ≥1 投票源 |logFC| > 0.25（≥1 票）；
    加权与等权两法同分母；
  - 加权 Δ* = Σ ρ_used·Δ / Σ ρ_used（sig 投票源内；Σρ_used=0 → 等权均值 fallback，细化二）；
  - 读数 = 符号恢复率（consensus Δ 与留出源 logFC 同号比例；sign(Δ)=0 记不恢复并单独计数）；
  - PASS = 加权 ≥ 等权 + 2pp，且两留出源同时成立（AND 门）；任一不成立即 FAIL；
  - w（A.1 预注册）：agree = 同向 sig 源 ρ_used 和 / 全体 sig 源 ρ_used 和（fallback 单元按等权
    口径 = 同向源数/N，细化二）；w = clip(2·agree − 1, 0, 1)；N<3 → w′ = 0.7 + 0.3·w（β=0.7 冻结）；
  - 细化一：N = clip 前 sig 投票源数（与 ρ 无关），ρ_used=0 源计入 N、贡献 0 权重；
  - morphology E 条款（预声明）：两留出源的唯一正 ρ 投票源均为 hct116
    （rpe1↔hct116 0.5515 / hepg2↔hct116 0.6848；gwps/k562_ess/jurkat ρ_raw ≤ 0 → ρ_used=0），
    LOPO 实质 = "hct116 单源加权 vs 4 源等权"——预声明即检验对象，不构成偏差；
  - 披露义务（冻结 §10）：fallback 单元占比 / N 分布与软收缩触发面 / 逐源单源恢复率 /
    sig 构成中 ρ_used=0 源占比，全部量化入 report。

运行：source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026 && \
     python scripts/build_fid_srcquality_04_lopo.py            # 正式执行（需 ③ 产物在盘）
     python scripts/build_fid_srcquality_04_lopo.py --selftest # 合成数据单测（不读 ③ 产物）
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
FQ = f"{ROOT}/data/interim/fid_srcquality"
VOTES = f"{FQ}/gene_source_votes.parquet"
VOTES_META = f"{FQ}/gene_source_votes_build_meta.json"
WEIGHTS = f"{FQ}/source_context_weights.parquet"
OUT_JSON = f"{FQ}/source_lopo_report.json"

TAU = 0.25
BETA = 0.7
POOL = ["gwps", "hct116_all", "jurkat_all", "k562_ess_all"]   # REV C2 投票池
HOLDOUTS = {"rpe1": "rpe1_all", "hepg2": "hepg2_all"}          # basal ctx -> 留出源
# 冻结 ρ 读数（k/165 解析签名，1e-12 断言；负值经 clip 后 rho_used = 0）
KEY_RHO_USED = {
    ("hct116_all", "rpe1"): 91 / 165,     # 0.5515151515151515
    ("hct116_all", "hepg2"): 113 / 165,   # 0.6848484848484848
    ("gwps", "rpe1"): 0.0,                # raw −55/165
    ("gwps", "hepg2"): 0.0,               # raw −15/165
    ("jurkat_all", "rpe1"): 0.0,          # raw −141/165
    ("jurkat_all", "hepg2"): 0.0,         # raw −129/165
    ("k562_ess_all", "rpe1"): 0.0,        # raw −45/165
    ("k562_ess_all", "hepg2"): 0.0,       # raw −5/165
}
KEY_RHO_RAW = {
    ("hct116_all", "rpe1"): 91 / 165,
    ("hct116_all", "hepg2"): 113 / 165,
    ("gwps", "rpe1"): -55 / 165,
    ("gwps", "hepg2"): -15 / 165,
    ("jurkat_all", "rpe1"): -141 / 165,
    ("jurkat_all", "hepg2"): -129 / 165,
    ("k562_ess_all", "rpe1"): -45 / 165,
    ("k562_ess_all", "hepg2"): -5 / 165,
}


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rho() -> dict:
    """② 冻结表 → {(source, context): (rho_raw, rho_used)}，并断言关键读数。"""
    w = pq.read_table(WEIGHTS).to_pandas()
    w = w[w["rho_raw"].notna()]
    rho = {}
    for _, r in w.iterrows():
        rho[(r["source"], r["context"])] = (float(r["rho_raw"]), float(r["rho_used"]))
    for k, v in KEY_RHO_RAW.items():
        assert abs(rho[k][0] - v) < 1e-12, f"ρ_raw 冻结读数不符: {k} {rho[k][0]} != {v}"
        assert abs(rho[k][1] - KEY_RHO_USED[k]) < 1e-12, f"ρ_used 冻结读数不符: {k} {rho[k][1]}"
    return {k: v[1] for k, v in rho.items()}


def compute_units(m: pd.DataFrame) -> pd.DataFrame:
    """逐单元 A.1 计算。m 列 = [target_gene, gene, source, logfc, lfc_h, rho]；
    返回 unit 级表（index = (target,gene)）：N / sum_rho / dw / de / lfc_h / fallback / agree / w0 / w。"""
    keys = ["target_gene", "gene"]
    gb = m.groupby(keys)
    N = gb.size().astype(np.float64)
    sum_rho = gb["rho"].sum()
    sum_rho_lfc = gb["rho_lfc"].sum()
    sum_lfc = gb["logfc"].sum()
    lfc_h = gb["lfc_h"].first()
    pos = sum_rho > 0
    dw_raw = sum_rho_lfc / sum_rho.where(pos, 1.0)          # Σρ·Δ/Σρ（Σρ>0）
    dw = dw_raw.where(pos, sum_lfc / N)                     # fallback（细化二）：等权均值
    de = sum_lfc / N                                        # 等权法
    unit = pd.DataFrame({
        "N": N, "sum_rho": sum_rho, "dw": dw, "de": de, "lfc_h": lfc_h, "fallback": ~pos,
    })
    # 逐行同向指示（对 dw / de 各一套），回聚合得 agree
    m2 = m.merge(unit.reset_index(), on=keys, how="left", validate="many_to_one")
    sg = np.sign(m2["logfc"].astype(np.float64).to_numpy())
    m2 = m2.assign(
        num_w=np.where(sg == np.sign(m2["dw"].to_numpy(np.float64)),
                       m2["rho"].astype(np.float64).to_numpy(), 0.0),
        num_e=(sg == np.sign(m2["de"].astype(np.float64).to_numpy())).astype(np.float64),
    )
    gb2 = m2.groupby(keys)
    num_w = gb2["num_w"].sum().reindex(unit.index)
    num_e = gb2["num_e"].sum().reindex(unit.index)
    agree = np.where(unit["fallback"].to_numpy(),
                     (num_e / unit["N"]).to_numpy(np.float64),
                     (num_w / sum_rho).to_numpy(np.float64))
    w0 = np.clip(2.0 * agree - 1.0, 0.0, 1.0)
    w = np.where((unit["N"] < 3).to_numpy(), BETA + (1.0 - BETA) * w0, w0)  # N<3 软收缩
    unit["agree"] = agree
    unit["w0"] = w0
    unit["w"] = w
    return unit


def selftest() -> int:
    """合成 4 单元断言：加权/回退/收缩/单源/等权全路径。"""
    rows = [
        # A: gwps(+1, ρ0) + hct116(+0.5, ρ0.6) → dw=0.5, de=0.75, agree=1, w0=1, N=2<3 → w=1.0
        ("T1", "G1", "gwps", 1.0, 1.0, 0.0),
        ("T1", "G1", "hct116_all", 0.5, 1.0, 0.6),
        # B: gwps(+1, ρ0) + jurkat(−1, ρ0) → 全 0 fallback, dw=de=0.0, agree=0, w0=0 → w=0.7
        ("T2", "G2", "gwps", 1.0, 1.0, 0.0),
        ("T2", "G2", "jurkat_all", -1.0, 1.0, 0.0),
        # C: 单 hct116(−0.8, ρ0.5), N=1 → dw=−0.8, agree=1, w0=1 → w=1.0（细化一预期行为）
        ("T3", "G3", "hct116_all", -0.8, -1.0, 0.5),
        # D: hct116(+1, ρ0.6) + gwps(+1, ρ0) + jurkat(−1, ρ0), N=3 → dw=1.0, de=1/3, agree=1, w=1.0
        ("T4", "G4", "hct116_all", 1.0, 1.0, 0.6),
        ("T4", "G4", "gwps", 1.0, 1.0, 0.0),
        ("T4", "G4", "jurkat_all", -1.0, 1.0, 0.0),
    ]
    m = pd.DataFrame(rows, columns=["target_gene", "gene", "source", "logfc", "lfc_h", "rho"])
    m["rho_lfc"] = m["rho"] * m["logfc"]
    u = compute_units(m)
    exp = {
        ("T1", "G1"): dict(N=2.0, sum_rho=0.6, dw=0.5, de=0.75, fallback=False, agree=1.0, w=1.0),
        ("T2", "G2"): dict(N=2.0, sum_rho=0.0, dw=0.0, de=0.0, fallback=True, agree=0.0, w=0.7),
        ("T3", "G3"): dict(N=1.0, sum_rho=0.5, dw=-0.8, de=-0.8, fallback=False, agree=1.0, w=1.0),
        ("T4", "G4"): dict(N=3.0, sum_rho=0.6, dw=1.0, de=1.0 / 3.0, fallback=False, agree=1.0, w=1.0),
    }
    for k, e in exp.items():
        for c, v in e.items():
            got = u.loc[k, c]
            ok = bool(got) == v if isinstance(v, bool) else abs(float(got) - v) < 1e-12
            assert ok, f"selftest {k}.{c}: got {got} expect {v}"
    rec_w = float((np.sign(u["dw"].to_numpy(np.float64)) == np.sign(u["lfc_h"].to_numpy(np.float64))).mean())
    rec_e = float((np.sign(u["de"].to_numpy(np.float64)) == np.sign(u["lfc_h"].to_numpy(np.float64))).mean())
    assert abs(rec_w - 0.75) < 1e-12 and abs(rec_e - 0.75) < 1e-12, f"recovery {rec_w}/{rec_e}"
    print("[selftest] 4 单元 A.1 全路径断言通过（加权/回退/收缩/单源/等权；rec_w=rec_e=0.75）")
    return 0


def run_holdout(ctx: str, src: str, sig: pd.DataFrame, rho: dict) -> dict:
    """单留出源 LOPO：eval 单元构建 + 两法符号恢复率 + §10 披露量化。"""
    h = sig.loc[sig["source"] == src, ["target_gene", "gene", "logfc"]].rename(columns={"logfc": "lfc_h"})
    assert not h.duplicated(["target_gene", "gene"]).any(), f"{src}: (target,gene) 重复"
    v = sig.loc[sig["source"].isin(POOL)]
    m = h.merge(v, on=["target_gene", "gene"], how="inner")
    m["rho"] = m["source"].map({s: rho[(s, ctx)] for s in POOL}).astype(np.float64)
    m["rho_lfc"] = m["rho"] * m["logfc"].astype(np.float64)
    unit = compute_units(m)
    D = len(unit)
    assert D > 0, f"{ctx}: eval 单元为空"

    sg_h = np.sign(unit["lfc_h"].to_numpy(np.float64))
    rec_w = float((np.sign(unit["dw"].to_numpy(np.float64)) == sg_h).mean())
    rec_e = float((np.sign(unit["de"].to_numpy(np.float64)) == sg_h).mean())
    n_dw_zero = int((np.sign(unit["dw"].to_numpy(np.float64)) == 0).sum())
    n_de_zero = int((np.sign(unit["de"].to_numpy(np.float64)) == 0).sum())

    # E 条款数值确认：除 hct116 外全部投票源 ρ_used=0 → fallback ⟺ hct116 不 sig
    hct = m.loc[m["source"] == "hct116_all", ["target_gene", "gene"]].drop_duplicates()
    n_hct_units = len(hct)
    n_fb = int(unit["fallback"].sum())
    assert n_fb == D - n_hct_units, \
        f"{ctx}: fallback 单元数 {n_fb} != 非hct116单元数 {D - n_hct_units}（E 条款数验失败）"

    # 非门控诊断（冻结 §8）：逐源单源恢复率 + 该源投票单元上的等权/加权基线。
    # de/dw 是 unit 级聚合列 → 回联到该源的 (target,gene) 行（sig 行单源内唯一 → 行=单元）。
    per_src = {}
    unit_kd = unit.reset_index()[["target_gene", "gene", "de", "dw"]]
    for s in POOL:
        ms = m.loc[m["source"] == s]
        if len(ms) == 0:
            per_src[s] = {"units": 0}
            continue
        j = ms.merge(unit_kd, on=["target_gene", "gene"], how="left", validate="many_to_one")
        assert len(j) == len(ms), f"{ctx}/{s}: unit 回联行数漂移 {len(j)} != {len(ms)}"
        assert j["de"].notna().all() and j["dw"].notna().all(), \
            f"{ctx}/{s}: unit 回联未命中键（de/dw NaN）"
        sg = np.sign(j["logfc"].astype(np.float64).to_numpy())
        sg_hh = np.sign(j["lfc_h"].astype(np.float64).to_numpy())
        per_src[s] = {
            "units": int(len(ms)),
            "recovery_single": float((sg == sg_hh).mean()),
            "equal_baseline_same_units": float((np.sign(j["de"].to_numpy(np.float64)) == sg_hh).mean()),
            "weighted_same_units": float((np.sign(j["dw"].to_numpy(np.float64)) == sg_hh).mean()),
        }

    # 目标级投票分布（对照冻结 §9 预注册 {2:38, 3:623, 4:1732}）
    vt = v.loc[v["target_gene"].isin(set(h["target_gene"].unique()))]
    vdist = vt.groupby("target_gene")["source"].nunique().value_counts().sort_index()

    res = {
        "holdout_source": src,
        "context": ctx,
        "holdout_sig_units": int(len(h)),
        "eval_units_ge1vote": int(D),
        "dropped_no_vote": int(len(h) - D),
        "recovery_weighted": rec_w,
        "recovery_equal": rec_e,
        "margin_pp": 100.0 * (rec_w - rec_e),
        "pass_ge2pp": bool(rec_w - rec_e >= 0.02 - 1e-12),
        "n_zero_delta_weighted": n_dw_zero,
        "n_zero_delta_equal": n_de_zero,
        "fallback_units": n_fb,
        "fallback_frac": n_fb / D,
        "hct116_sig_units": int(n_hct_units),
        "N_distribution": {str(int(k)): int(c) for k, c in
                           unit["N"].value_counts().sort_index().items()},
        "n_units_N_lt3": int((unit["N"] < 3).sum()),
        "frac_N_lt3": float((unit["N"] < 3).mean()),
        "rho_used_zero_voter_rows_frac": float((m["rho"] == 0.0).mean()),
        "units_with_any_rho0_voter_frac": float(
            m.loc[m["rho"] == 0.0].groupby(["target_gene", "gene"]).ngroups / D),
        "w_mean": float(unit["w"].mean()),
        "frac_w_at_beta_floor": float(((unit["N"] < 3) & (unit["w0"] == 0.0)).mean()),
        "per_source_single_source_diagnostics": per_src,
        "target_level_vote_distribution_holdout_universe": {
            str(int(k)): int(c) for k, c in vdist.items()},
    }
    print(f"[holdout {ctx}] holdout_sig={len(h):,} eval_units={D:,} dropped={len(h) - D:,} "
          f"(fallback={n_fb:,} = {n_fb / D:.2%}, N<3 = {res['frac_N_lt3']:.2%})", flush=True)
    print(f"[holdout {ctx}] recovery weighted={rec_w:.4f} equal={rec_e:.4f} "
          f"margin={res['margin_pp']:+.2f}pp pass={res['pass_ge2pp']}", flush=True)
    return res


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()

    meta3 = json.load(open(VOTES_META))
    votes_sha = sha256_of(VOTES)
    assert votes_sha == meta3["output_sha256"], "③ votes parquet sha256 与 build meta 不符"
    w_sha = sha256_of(WEIGHTS)
    rho = load_rho()

    sig = pq.read_table(VOTES, columns=["target_gene", "gene", "source", "logfc"],
                        filters=[("sig", "=", True)]).to_pandas()
    assert (sig["logfc"].abs() > TAU).all(), "加载行含 |logFC| ≤ τ（sig 列与门控不一致）"
    per_src = sig["source"].value_counts().to_dict()
    print(f"[load] sig 行加载: total={len(sig):,} per_source={per_src}", flush=True)
    for need in ["rpe1_all", "hepg2_all"] + POOL:
        assert per_src.get(need, 0) > 0, f"{need}: sig 行为空"
    assert "norman_all" not in POOL and "rpe1_all" not in POOL and "hepg2_all" not in POOL

    results = {ctx: run_holdout(ctx, src_, sig, rho) for ctx, src_ in HOLDOUTS.items()}
    gate_pass = all(r["pass_ge2pp"] for r in results.values())

    report = {
        "built": "2026-10-14",
        "executor": "cb-expert",
        "script": "scripts/build_fid_srcquality_04_lopo.py",
        "inputs": {
            "gene_source_votes.parquet": {"sha256": votes_sha, "build_meta_rows": meta3["rows_total"],
                                          "build_meta_sig_rows": meta3["sig_rows_total"]},
            "source_context_weights.parquet": {"sha256": w_sha},
        },
        "frozen_refs": {
            "module_spectra_z_frozen.csv.sha256": sha256_of(f"{FQ}/module_spectra_z_frozen.csv"),
            "rho_9ctx_frozen.csv.sha256": sha256_of(f"{FQ}/rho_9ctx_frozen.csv"),
            "refs/essentialome_targets.csv.sha256": sha256_of(f"{FQ}/refs/essentialome_targets.csv"),
        },
        "pre_registered": {
            "pool_C2": POOL,
            "holdouts": HOLDOUTS,
            "tau": TAU,
            "beta": BETA,
            "gate": "PASS = 加权符号恢复率 ≥ 等权 + 2pp，且 rpe1/hepg2 两留出源同时成立（AND）",
            "rho_used_rule": "rho_used = max(rho_raw, 0)，永不翻号；Σρ_used=0 → 等权 fallback（细化二）",
            "N_counting": "N = clip 前 sig 投票源数（细化一）；N<3 → w′ = 0.7 + 0.3·w",
            "morphology_E_clause": "两留出源唯一正 ρ 投票源均为 hct116（0.5515/0.6848）→ LOPO 实质 = "
                                   "hct116 单源加权 vs 4 源等权；预声明即检验对象，非偏差",
        },
        "holdout_results": results,
        "gate": {"pass": bool(gate_pass),
                 "rule": "weighted ≥ equal + 2pp both holdouts (AND)"},
        "disclosure_notes": [
            "eval 单元级读数；冻结 §9 预注册 {2:38,3:623,4:1732} 为目标级对照列（见各 holdout 的 "
            "target_level_vote_distribution_holdout_universe）",
            "norman_all 按 REV C2 剔出投票池（导出保留全行）；rpe1_all/hepg2_all 为留出源不进共识",
            "N 计数按细化一：clip 前 sig 源数；ρ_used=0 源计入 N 贡献 0 权",
            "fallback 单元 Δ* = 等权均值，w 按等权口径 A.1（细化二）",
        ],
    }
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[write] {OUT_JSON}")
    print(f"[GATE] LOPO 判据: {'PASS' if gate_pass else 'FAIL'} "
          f"(rpe1 margin {results['rpe1']['margin_pp']:+.2f}pp / "
          f"hepg2 margin {results['hepg2']['margin_pp']:+.2f}pp, 需双源 ≥ +2pp)")
    print("BUILD 04 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
