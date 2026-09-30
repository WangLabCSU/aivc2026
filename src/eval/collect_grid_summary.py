"""汇总 L1.3 网格各 run 的 P_H1 六指标 → grid_summary.json + grid_summary.md。

用法: python src/eval/collect_grid_summary.py [tag ...]   # 缺省=全部 g13_* + 基线
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "outputs/metrics/local_bench"
SHORT = {"pds_cosine": "pds", "expr_mse_unbiased": "mse", "de_wilcoxon_lfc_nmae": "nmae",
         "de_wilcoxon_direction_fidelity_yield_raw": "fid",
         "de_wilcoxon_direction_reach_raw": "reach", "de_wilcoxon_sig_jaccard": "jac"}
ORDER = ["pds", "mse", "nmae", "fid", "reach", "jac"]
# 展示基线: L1.2 (λ=1.5 无稀疏) 与 generic-response 官方基线（本地 P_H1）
BASELINES = [("l11_lam1.5", "L1.2 lam1.5 no-sparse (baseline)")]


def cfg_of(tag: str) -> dict:
    import yaml
    for cand in [BENCH / tag / "summary.json"]:
        if cand.exists():
            s = json.loads(cand.read_text())
            mc = s.get("model_config", "")
            cfg = yaml.safe_load(open(ROOT / mc)) if mc and (ROOT / mc).exists() else {}
            lam = cfg.get("lambda_real", s["per_context"].get("P_H1", {}).get("lambda_real", "?"))
            return {"lam": lam, "top_k": cfg.get("top_k"), "stress_alpha": cfg.get("stress_alpha", 0.0),
                    "cfg": mc}
    return {}


def collect(tags: list[tuple[str, str]]) -> tuple[dict, list[str]]:
    rows, lines_md = {}, []
    for tag, label in tags:
        p = BENCH / tag / "summary.json"
        if not p.exists():
            lines_md.append(f"| {label} | pending |" + " |" * 6)
            rows[tag] = None
            continue
        s = json.loads(p.read_text())
        pc = s["per_context"].get("P_H1", {})
        vals = {SHORT[m]: round(float(pc[m]), 4) for m in SHORT if m in pc}
        rows[tag] = {"label": label, **{k: vals.get(k) for k in ORDER}}
        cells = " | ".join(f"{vals.get(k, float('nan')):.4f}" if vals.get(k) is not None
                           else "-" for k in ORDER)
        lines_md.append(f"| {label} | {cells} |")
    return rows, lines_md


def main() -> None:
    tags = []
    for a in sys.argv[1:]:
        t, _, lbl = a.partition(":")
        tags.append((t, lbl or t))
    if not tags:
        for t, lbl in BASELINES + [("l11_lam1.0", "L1.1 lam1.0 no-sparse")]:
            tags.append((t, lbl))
        for d in sorted(BENCH.glob("g13*_*.json.parent")) + sorted(
                p for p in BENCH.glob("g13*") if p.is_dir()):
            tag = d.name
            cfg = cfg_of(tag)
            if cfg:
                lbl = (f"lam{cfg['lam']} K{cfg['top_k']} a{cfg['stress_alpha']}"
                       .replace("KNone", "K-").replace(".0 ", " ").replace(".0", "")
                       .replace("a0.0", "a0"))
            else:
                lbl = tag
            tags.append((tag, lbl))
    rows, lines = collect(tags)
    out = {"per_run_P_H1": rows}
    (BENCH / "grid_summary.json").write_text(json.dumps(out, indent=2))
    header = ("| run | pds | mse | nmae | fid | reach | jac |\n|---|---|---|---|---|---|---|")
    (BENCH / "grid_summary.md").write_text(
        "# L1.3 grid (P_H1 raw, cell_sampling=true, seed=0)\n\n" + header + "\n"
        + "\n".join(lines) + "\n")
    print((BENCH / "grid_summary.md").read_text())


if __name__ == "__main__":
    main()
