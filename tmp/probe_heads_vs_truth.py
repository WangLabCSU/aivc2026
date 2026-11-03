# iter023 v2: 用正确的 100 目标集 (Q_HepG2 targets.json) 重跑各头 vs truth
import pandas as pd, numpy as np, json

truth = pd.read_parquet("tmp/q_hepg2_truth_lfc_top50.parquet")
t100 = json.load(open("data/interim/local_bench/Q_HepG2/targets.json"))
print("truth targets:", truth.target.nunique(), "t100:", len(t100), "coverage:", truth.target.isin(t100).mean())

bz = pd.read_parquet("data/interim/bizard/v02/agg_delta_submit_A.parquet")

cols = ["target_gene", "gene", "logfc"]
sigs = [pd.read_parquet(f"data/interim/signatures/{n}.parquet", columns=cols).rename(columns={"logfc": "delta"}) for n in ["gwps", "hct116_all"]]
real = pd.concat(sigs).groupby(["target_gene", "gene"], sort=False)["delta"].mean().reset_index()
del sigs

def agree_vs_truth(head, name):
    m = truth.merge(head, left_on=["target", "gene"], right_on=["target_gene", "gene"], suffixes=("_t", ""))
    hcol = "delta" if "delta" in m.columns else "delta_h"
    both = (np.abs(m.lfc) > 0.1) & (np.abs(m[hcol]) > 1e-6)
    if both.sum() == 0:
        return {"name": name, "pairs": 0}
    ok = (np.sign(m.lfc[both]) == np.sign(m[hcol][both])).to_numpy()
    r = float(ok.mean())
    w = np.abs(m.lfc[both]).to_numpy()
    wr = float((w * ok).sum() / w.sum())
    return {"name": name, "pairs": int(both.sum()), "sign_agree": round(r, 4), "weighted": round(wr, 4)}

out = [agree_vs_truth(bz, "bizard_v02_top50"), agree_vs_truth(real, "dualsrcdeep_top50")]
print(json.dumps(out, indent=1))

# top-k: 按 |delta| 每 target 取 k, 测 truth-sign 命中率
def topk_probe(head, name):
    h = head[head.target_gene.isin(t100)]
    res = {}
    for k in [500, 1000, 2000]:
        topk = h.reindex(h.delta.abs().sort_values(ascending=False).index).groupby("target_gene").head(k)
        m = truth.merge(topk, left_on=["target", "gene"], right_on=["target_gene", "gene"])
        both = np.abs(m.lfc) > 0.1
        r = (np.sign(m.lfc[both]) == np.sign(m.delta[both])).mean() if both.sum() else None
        res[k] = {"pairs": int(both.sum()), "sign_agree": round(float(r), 4) if r is not None else None}
    print(f"topk_{name}:", json.dumps(res))
    return res

resb = topk_probe(bz, "bizard")
resd = topk_probe(real, "dualsrcdeep")

json.dump({"head_vs_truth_top50": out, "topk_bizard": resb, "topk_dualsrcdeep": resd},
          open("tmp/delta_heads_vs_truth.json", "w"), indent=1)
print("DONE")
