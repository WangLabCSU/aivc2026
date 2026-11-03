import pandas as pd, h5py, numpy as np
# KEY QUESTION: what do the "contexts" A/B/C in the current challenge look like in vcc2025 terms?
# vcc2025: Flex_1/2/3 = 3 contexts? cells 221273/150 perts ≈ 1475 cells/pert →
# 221273 = 150 perts * ~1400 + NTC. 48 batches / 3 Flex groups = 16 per group.
# If Flex_1/2/3 were contexts like A/B/C, then per (pert,batch) cells = 4339/16 ≈ 271...
# Actually check: does each Flex group contain the SAME 150 targets? That's the context-transfer structure.
f = h5py.File("data/public/vcc2025/adata_Training.h5ad", "r")
obs = f["obs"]
bcat = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["batch"]["categories"][:]]
bcodes = obs["batch"]["codes"][:]
tcat = [c.decode() if isinstance(c, bytes) else str(c) for c in obs["target_gene"]["categories"][:]]
tcodes = obs["target_gene"]["codes"][:]
# per Flex group: which targets and how many NTC cells
for grp in ("Flex_1", "Flex_2", "Flex_3"):
    m = np.array([bcat[c].startswith(grp) for c in bcodes])
    tgs = set(tcat[c] for c in tcodes[m])
    ntc_cells = int(m[tcat == [t for t in tgs if "targeting" in t.lower()][0]].sum()) if any("targeting" in t.lower() for t in tgs) else 0
    print(f"{grp}: cells={m.sum()}, targets={len(tgs)}, has_ntc={'non-targeting' in tgs}")
# per-batch targets: does each batch have a unique target or shared?
m12 = np.array([bcat[c] == "Flex_1_01" for c in bcodes])
print("Flex_1_01 targets:", sorted(set(tcat[c] for c in tcodes[m12])))
m13 = np.array([bcat[c] == "Flex_1_02" for c in bcodes])
print("Flex_1_02 targets:", sorted(set(tcat[c] for c in tcodes[m13])))
m21 = np.array([bcat[c] == "Flex_2_01" for c in bcodes])
print("Flex_2_01 targets:", sorted(set(tcat[c] for c in tcodes[m21])))
f.close()
