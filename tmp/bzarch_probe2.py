# Probe 2 (decision): vcc2025 跨 Flex 组 per-target logFC 相关性
# 判据: spearman ~0.9 → Flex 组≈重复, 学不到 context-specific 成分 (ECRD 前提弱)
#        ~0.5-0.7 → 有可学的 context-specific 成分 (ECRD 继续)
# 方法: 前 40 个 (按出现顺序) 在三组均有 ≥50 cells 的 target, 各组伪批量
#       log2((pm+0.5)/(ntc_same_group+0.5)), 两两 spearman + cosine
import h5py, numpy as np, scipy.sparse as sp
from scipy.stats import spearmanr

rng = np.random.default_rng(0)
f = h5py.File("data/public/vcc2025/adata_Training.h5ad", "r")
obs = f["obs"]
def cats(key):
    return np.array([c.decode() if isinstance(c, bytes) else str(c)
                     for c in obs[key]["categories"][:]])
bcat, tcat = cats("batch"), cats("target_gene")
bcodes = obs["batch"]["codes"][:]
tcodes = obs["target_gene"]["codes"][:]

# map batch -> flex group
gcode = np.array([b.split("_")[0] for b in bcat])  # Flex_1..3 groups per batch category
batch_grp = np.array([b.split("_")[0] + "_" + b.split("_")[1] for b in bcat])

indptr = f["X"]["indptr"][:]
data = f["X"]["data"][:]
indices = f["X"]["indices"][:]

def rows_sum(idx):
    if len(idx) == 0:
        return np.zeros(18080)
    seg = np.concatenate([np.arange(indptr[i], indptr[i+1]) for i in idx])
    out = np.zeros(18080)
    np.add.at(out, indices[seg], data[seg])
    return out

# cells per (group, target) counts — vectorized via bincount on (tcode*4+gnum[bcodes])
gnum_b = np.array([int(bcat[c].split("_")[1]) for c in bcodes])  # per-cell group 1/2/3
pair = tcodes.astype(np.int64) * 4 + gnum_b
counts = np.bincount(pair, minlength=len(tcat)*4).reshape(len(tcat), 4)

ntc = list(tcat).index("non-targeting")
# candidate targets: >=50 cells in all 3 groups, not NTC
ok = (counts[:, 1] >= 50) & (counts[:, 2] >= 50) & (counts[:, 3] >= 50)
ok[ntc] = False
cand = np.where(ok)[0][:40]
print("candidate targets (40 of", ok.sum(), "eligible):", len(cand))
print("ntc cells per group:", counts[ntc, 1:], "| flex sizes:",
      (gnum_b == 1).sum(), (gnum_b == 2).sum(), (gnum_b == 3).sum())

# NTC pseudo-bulk per group (all NTC cells capped 20000)
ntc_prof = {}
for g in (1, 2, 3):
    idx = np.where((tcodes == ntc) & (gnum_b == g))[0]
    if len(idx) > 20000:
        idx = rng.choice(idx, 20000, replace=False)
    ntc_prof[g] = rows_sum(idx) / len(idx)
    print(f"NTC grp{g}: {len(idx)} cells")

lfcs = {}  # tcode -> {g: lfc vector}
for ti in cand:
    lfcs[ti] = {}
    for g in (1, 2, 3):
        idx = np.where((tcodes == ti) & (gnum_b == g))[0]
        pm = rows_sum(idx) / len(idx)
        lfcs[ti][g] = np.log2((pm + 0.5) / (ntc_prof[g] + 0.5))
np.save("tmp/bzarch_probe2_lfcs.npy", np.array([[lfcs[ti][g] for g in (1,2,3)] for ti in cand]))

sp12, sp13, sp23, co12, co13, co23 = [], [], [], [], [], []
for ti in cand:
    a, b, c = lfcs[ti][1], lfcs[ti][2], lfcs[ti][3]
    sp12.append(spearmanr(a, b).statistic); sp13.append(spearmanr(a, c).statistic); sp23.append(spearmanr(b, c).statistic)
    co12.append(np.corrcoef(a, b)[0,1]); co13.append(np.corrcoef(a, c)[0,1]); co23.append(np.corrcoef(b, c)[0,1])
m = lambda v: np.nanmedian(v)
print(f"spearman medians: F1-F2 {m(sp12):.4f}  F1-F3 {m(sp13):.4f}  F2-F3 {m(sp23):.4f}")
print(f"pearson  medians: F1-F2 {m(co12):.4f}  F1-F3 {m(co13):.4f}  F2-F3 {m(co23):.4f}")
# 残差分析: 每靶减去三组均值后的组间残差相关 (context-specific 成分是否可复现)
r12, r13, r23 = [], [], []
for ti in cand:
    X = np.stack([lfcs[ti][1], lfcs[ti][2], lfcs[ti][3]])  # 3x18080
    R = X - X.mean(axis=0, keepdims=True)
    r12.append(spearmanr(R[0], R[1]).statistic); r13.append(spearmanr(R[0], R[2]).statistic); r23.append(spearmanr(R[1], R[2]).statistic)
print(f"residual spearman medians (context-specific reproducibility): "
      f"F1-F2 {m(r12):.4f}  F1-F3 {m(r13):.4f}  F2-F3 {m(r23):.4f}")
print("DECISION: spearman~0.9 & residual~0 → replicate noise, ECRD premise weak;")
print("          spearman 0.5-0.7 & residual >0.2 → learnable context component, ECRD go")
f.close()
