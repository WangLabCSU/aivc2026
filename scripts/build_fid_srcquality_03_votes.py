#!/usr/bin/env python
"""v2.2 ③gene_source_votes.parquet 构建脚本（fid/源质量加权线，C1 冻结后按 PI 放行顺序执行）。

产出 data/interim/fid_srcquality/gene_source_votes.parquet：
  - 7 源全导出（含 norman_all）：gwps / hct116_all / hepg2_all / jurkat_all /
    k562_ess_all / norman_all / rpe1_all，schema (target_gene, gene, source, logfc, fdr, sig, dir)；
  - sig 门控 = |logFC| > τ=0.25（冻结文档 §6；源 parquet 无 FDR 列，fdr = NaN 占位列，
    FDR<0.05 分支数据面不可计算——PI 认可 fallback，双档披露，非口径选择）；
  - dir = sign(logfc) ∈ {-1, 0, +1}（int8；logfc=0 → 0）；
  - **norman_all 行保留在导出中**（PI 放行顺序："7 源全导出含 norman"），
    LOPO 投票池按 REV C2 在 ④ 执行时剔除（池 = {gwps, hct116_all, jurkat_all, k562_ess_all}），
    本脚本不做池过滤；
  - 短名 hepg2/jurkat/k562_ess/rpe1.parquet 为 2280 字节空占位文件，数据版为 *_all；
    tahoe_p12 / 短名 norman 不属于 7 源线，不导出。

唯一性披露（2026-10-14 PI 裁定）：全行 (target,gene) 唯一性不成立为已知数据面现象——
hct116_all 源签名系 109 批次采集合并（其 build meta n_batches=109），含非相邻重复行且重复行
logfc 有差异（rg0 实测 1,165 重复行 = 542 键×2 + 27 键×3，max spread 0.0928），重复集中在
非 sig 段（sig 行全量核查零重复）。下游 ④ LOPO 仅经 filters=[("sig","=",True)] 消费 sig 行，
非 sig 重复行不进入任何计算；故本脚本唯一性断言收缩至 sig 行，全行重复计数入 meta 披露。

--verify-only：对在盘输出重跑全部断言并补写 build meta（不重建；用于首跑构建段完整落盘
而验证段 crash 后的收尾——输出字节与首跑一致，构建段不重跑）。

排序：source, target_gene, gene 升序（逐源读入后排序再写，供 ④ LOPO 与联表稳定）。

运行：source /data/home/wsx/miniconda3/etc/profile.d/conda.sh && conda activate aivc2026 && \
     python scripts/build_fid_srcquality_03_votes.py                # 正式构建（全量重写+验证）
     python scripts/build_fid_srcquality_03_votes.py --verify-only  # 仅验证在盘输出+补写 meta
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = "/data/resource/wanglab/aivc/aivc2026"
FQ = f"{ROOT}/data/interim/fid_srcquality"
SIG = f"{ROOT}/data/interim/signatures"
OUT_PARQUET = f"{FQ}/gene_source_votes.parquet"
OUT_META = f"{FQ}/gene_source_votes_build_meta.json"
REFS = f"{FQ}/refs"

TAU = 0.25
# 源名 → 签名 parquet 文件（source 名 = 文件名去 .parquet，与 ② build_meta naming_note 一致）
SOURCES = {
    "gwps":        f"{SIG}/gwps.parquet",
    "hct116_all":  f"{SIG}/hct116_all.parquet",
    "hepg2_all":   f"{SIG}/hepg2_all.parquet",
    "jurkat_all":  f"{SIG}/jurkat_all.parquet",
    "k562_ess_all": f"{SIG}/k562_ess_all.parquet",
    "norman_all":  f"{SIG}/norman_all.parquet",
    "rpe1_all":    f"{SIG}/rpe1_all.parquet",
}
# 清点实测行数（2026-10-14 全量清点；hct116_all 6.7 亿行用 target_gene 列扫描测得）
EXPECT_ROWS = {
    "gwps": 81_162_977,
    "hct116_all": 670_122_948,
    "hepg2_all": 22_989_288,
    "jurkat_all": 21_220_160,
    "k562_ess_all": 17_577_592,
    "norman_all": 2_284_813,
    "rpe1_all": 20_905_429,
}
CHUNK = 20_000_000  # hct116_all 3.97 GB 分块读


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_source(path: str, src: str) -> pd.DataFrame:
    """读单源签名 parquet（hct116_all 分块），加 source/fdr/sig/dir 列，排序返回。"""
    pf = pq.ParquetFile(path)
    n = pf.metadata.num_rows
    parts = []
    for batch in pf.iter_batches(batch_size=CHUNK, columns=["target_gene", "gene", "logfc"]):
        parts.append(batch.to_pandas())
    df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    assert len(df) == n == EXPECT_ROWS[src], f"{src}: rows {len(df)} != expect {EXPECT_ROWS[src]}"
    del parts

    df["source"] = src
    df["fdr"] = np.float32(np.nan)                      # 数据面无 FDR 列 → NaN 占位（双档披露）
    df["sig"] = df["logfc"].abs() > TAU                 # 冻结门控 |logFC| > 0.25
    df["dir"] = np.sign(df["logfc"]).astype("int8")     # logfc=0 → 0

    df = df.sort_values(["source", "target_gene", "gene"], kind="stable",
                        ignore_index=True, na_position="last")
    return df[["target_gene", "gene", "source", "logfc", "fdr", "sig", "dir"]]


def main() -> int:
    verify_only = "--verify-only" in sys.argv
    # 输入完整性预检（冻结/引用资产 sha256 入 meta）
    frozen_refs = {
        "module_spectra_z_frozen.csv.sha256": sha256_of(f"{FQ}/module_spectra_z_frozen.csv"),
        "rho_9ctx_frozen.csv.sha256": sha256_of(f"{FQ}/rho_9ctx_frozen.csv"),
        "source_context_weights.parquet.sha256": sha256_of(f"{FQ}/source_context_weights.parquet"),
        "refs/comp300_targets.csv.sha256": sha256_of(f"{REFS}/comp300_targets.csv"),
        "refs/essentialome_targets.csv.sha256": sha256_of(f"{REFS}/essentialome_targets.csv"),
    }

    schema = pa.schema([
        ("target_gene", pa.string()),
        ("gene", pa.string()),
        ("source", pa.string()),
        ("logfc", pa.float32()),
        ("fdr", pa.float32()),
        ("sig", pa.bool_()),
        ("dir", pa.int8()),
    ])

    total = 0
    per_source = {}
    if verify_only:
        # --verify-only：首跑构建段已完整落盘（7 源全写完，验证段 hct116_all 断言处 crash），
        # 输出字节与首跑一致，不重建；直接进入下方断言 + meta 补写。
        pf0 = pq.ParquetFile(OUT_PARQUET)
        assert pf0.metadata.num_rows == sum(EXPECT_ROWS.values()), \
            f"在盘输出总行数 {pf0.metadata.num_rows} != {sum(EXPECT_ROWS.values())}"
        assert pf0.schema_arrow == schema, f"schema 漂移: {pf0.schema_arrow}"
        total = int(pf0.metadata.num_rows)
        print(f"[verify-only] 跳过构建段，在盘输出 rows={total:,}，进入断言 + meta 补写", flush=True)
        del pf0
    else:
        writer = pq.ParquetWriter(OUT_PARQUET, schema, compression="snappy")
        try:
            for src, path in SOURCES.items():
                df = read_source(path, src)
                tbl = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
                writer.write_table(tbl)
                n_sig = int(df["sig"].sum())
                per_source[src] = {"rows": len(df), "sig_rows": n_sig}
                total += len(df)
                print(f"[src] {src:<12} rows={len(df):,} sig={n_sig:,} ({n_sig / len(df):.1%})", flush=True)
                del df, tbl
        finally:
            writer.close()

    # ---- 断言（落盘后重读验证，不信任内存态）----
    pf = pq.ParquetFile(OUT_PARQUET)
    assert pf.metadata.num_rows == total == sum(EXPECT_ROWS.values()), \
        f"总行数 {pf.metadata.num_rows} != {sum(EXPECT_ROWS.values())}"
    assert pf.schema_arrow == schema, f"schema 漂移: {pf.schema_arrow}"

    total_sig = 0
    targets_union = set()
    dup_stats = {}        # 全行重复披露（唯一性裁定见 docstring；下游 ④ 仅消费 sig 行）
    out_sig_by_src = {}
    for src in SOURCES:
        # 从输出文件按源切片验证（sig/dir 是本脚本派生列，不存在于源签名 parquet）
        t = pq.read_table(OUT_PARQUET, columns=["target_gene", "gene", "logfc", "fdr", "sig", "dir"],
                          filters=[("source", "=", src)])
        d = t.to_pandas()
        # sig/dir 与 logfc 全量重算一致（向量化，无抽样）
        assert np.array_equal(d["sig"].values, (d["logfc"].abs() > TAU).values), f"{src}: sig 不一致"
        assert np.array_equal(d["dir"].values, np.sign(d["logfc"].astype(np.float64)).astype("int8").values), \
            f"{src}: dir 不一致"
        assert d["fdr"].isna().all(), f"{src}: fdr 应全 NaN"
        assert d["logfc"].notna().all(), f"{src}: logfc 不应有 NaN"
        # 唯一性断言收缩至 sig 行（PI 裁定 2026-10-14：非 sig 重复 = 多批次合并痕迹，下游 ④ 不消费）
        n_dup_sig = int(d.loc[d["sig"]].duplicated(["target_gene", "gene"]).sum())
        assert n_dup_sig == 0, f"{src}: sig 行 (target,gene) 重复 {n_dup_sig} 处"
        dup_stats[src] = {
            "full_row_dup_rows": int(d.duplicated(["target_gene", "gene"]).sum()),
            "sig_row_dups": n_dup_sig,
        }
        total_sig += int(d["sig"].sum())
        out_sig_by_src[src] = int(d["sig"].sum())
        targets_union.update(d["target_gene"].unique())
        del t, d
        print(f"[check] {src:<12} sig/dir 一致、无 NaN、sig 行唯一"
              f"（全行重复 {dup_stats[src]['full_row_dup_rows']}，披露入 meta）", flush=True)

    comp300 = set(pd.read_csv(f"{REFS}/comp300_targets.csv")["target_gene"])
    # RAB3B 披露式放行（2026-10-14 PI 裁定）：RAB3B 在 7 源签名 parquet 均无行（vocab.json 在,
    # essentialome 无、④ LOPO 不受影响）→ C1 冻结 §9 预注册 "1 靶 0 票" 实证命中。
    # 缺失名单 ⊆ 预注册集合 → 放行并入 meta 披露；超出预注册 → 硬 fail。
    PRE_REG_ZERO_VOTE = {"RAB3B"}
    comp300_missing = comp300 - targets_union
    assert comp300_missing <= PRE_REG_ZERO_VOTE, \
        f"comp300 缺失靶超出预注册 {sorted(comp300_missing)}"
    if comp300_missing:
        print(f"[check] comp300 缺失靶（披露式放行）: {sorted(comp300_missing)}", flush=True)

    # 输出侧逐源行数校验（只读 source 列，字典编码，秒级）
    out_counts = pq.read_table(OUT_PARQUET, columns=["source"]).to_pandas()["source"].value_counts().to_dict()
    assert out_counts == {s: EXPECT_ROWS[s] for s in SOURCES}, f"输出逐源行数不符: {out_counts}"
    print(f"[check] 输出侧逐源行数 == 清点值: {out_counts}", flush=True)

    # per_source 统一由输出侧回填（构建态与 --verify-only 态产物 meta 逐键一致）
    per_source = {s: {"rows": EXPECT_ROWS[s], "sig_rows": out_sig_by_src[s]} for s in SOURCES}

    meta = {
        "built": "2026-10-14",
        "builder": "cb-expert",
        "script": "scripts/build_fid_srcquality_03_votes.py",
        "frozen_refs": frozen_refs,
        "sig_gate": "|logFC| > 0.25 (tau=0.25 冻结; 无 FDR 列 → fdr=NaN 占位, PI 认可 fallback 双档披露)",
        "sources": per_source,
        "rows_total": int(total),
        "sig_rows_total": int(total_sig),
        "n_targets_union": len(targets_union),
        "uniqueness_note": "全行 (target,gene) 唯一性不成立为已知数据面现象（hct116_all 109 批次合并痕迹, "
                           "重复集中非 sig 段）；唯一性断言收缩至 sig 行（逐源 n_dup_sig==0 硬断言）；"
                           "下游 ④ LOPO 仅经 filters=[('sig','=',True)] 消费 sig 行",
        "full_row_dup_disclosure": dup_stats,
        "naming_note": "source 名 = 签名 parquet 文件名去 .parquet; 短名 hepg2/jurkat/k562_ess/rpe1.parquet 为空占位, 数据版为 *_all",
        "lopo_pool_note": "norman_all 全行保留于本导出; REV C2 投票池 {gwps,hct116_all,jurkat_all,k562_ess_all} 在 ④ LOPO 执行时剔除",
        "comp300_missing_disclosure": {
            "missing_targets": sorted(comp300_missing),
            "pre_registered_zero_vote": sorted(PRE_REG_ZERO_VOTE),
            "note": "comp300 缺失靶 = RAB3B（7 源签名 parquet 均无行; vocab.json 有 → 建模侧无缺口; "
                    "essentialome 无 → ④ LOPO 不受影响），命中 C1 冻结 §9 预注册 '1 靶 0 票'。"
                    "另注: data/raw/controls/pert_counts.csv 与 comp300_targets.csv 对称差 550 "
                    "（交集仅 25）——pert_counts 非竞赛 300 清单副本, 是不同 artifact, "
                    "comp300_targets.csv 权威维持。实测竞赛 300 靶投票分布 262 靶 2 票 / 37 靶 1 票 / "
                    "1 靶 0 票 vs §9 预注册 255/44/1（2/1 票界差 7 靶, 0 票数吻合）, 偏差待 ⑤ 纪要披露",
        },
        "row_order": "source, target_gene, gene 升序",
        "output_sha256": None,
    }
    meta["output_sha256"] = sha256_of(OUT_PARQUET)
    with open(OUT_META, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[write] {OUT_PARQUET} rows={total:,} sig={total_sig:,} targets_union={len(targets_union)}")
    print(f"[check] output sha256 = {meta['output_sha256']}")
    print("BUILD 03 PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
