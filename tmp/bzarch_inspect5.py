# The make-or-break probe for the learned-architecture route:
# can we train "context A → context B/C transfer" on vcc2025 Training/Validation
# and does that look like the current challenge (context transfer within same cell line)?
import h5py, numpy as np, collections, json
# vcc2025: what is the batch structure? Flex_1_xx = perturbation batches? 48 batches over 3 contexts?
# Check: is vcc2025 single-context (one cell line, all batches same condition)?
f = h5py.File("data/public/vcc2025/adata_Training.h5ad", "r")
obs = f["obs"]
# guide_id includes non-targeting?
cats = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["guide_id"]["categories"][:]]
ntc = [c for c in cats if "targeting" in c.lower() or c.lower() in ("ntc", "control", "safe_targeting")]
print("guide_id total:", len(cats), "ntc-like:", ntc[:5], "count ntc-like:", len(ntc))
tg = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["target_gene"]["categories"][:]]
print("target_gene ntc-like:", [c for c in tg if "targeting" in c.lower()][:3])
# batch naming: Flex_1_01..48 — any context info encoded?
bcats = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["batch"]["categories"][:]]
print("batches:", bcats)
f.close()
# Check the vcc2025 "context" structure: pert_counts_Training columns
pc = pd.read_csv("data/public/vcc2025/pert_counts_Training.csv")
print(pc.head(3))
print("median_umi range:", pc["median_umi_per_cell"].min(), pc["median_umi_per_cell"].max())
