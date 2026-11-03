import pandas as pd, json
# The critical overlap probe: does vcc2025 cover the CURRENT challenge's 300 targets?
cur = pd.read_csv("data/raw/controls/pert_counts.csv")["target_gene"].astype(str)
print("current challenge targets:", len(cur), cur.nunique())
sp = {}
for s in ("Training", "Validation", "Test"):
    pc = pd.read_csv(f"data/public/vcc2025/pert_counts_{s}.csv")["target_gene"].astype(str)
    sp[s] = set(pc)
    ov = set(cur) & set(pc)
    print(f"vcc2025 {s}: {len(pc)} targets, overlap with current-300: {len(ov)}")
allv = sp["Training"] | sp["Validation"] | sp["Test"]
print("vcc2025 total unique targets:", len(allv), "overlap with current-300:", len(allv & set(cur)))
print("current-300 NOT in vcc2025:", len(set(cur) - allv))
# manifest of controls
try:
    m = json.load(open("data/raw/controls/manifest.json"))
    print("controls manifest:", json.dumps(m, indent=1)[:1500])
except Exception as e:
    print("manifest err:", e)
