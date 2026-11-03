"""Verify manifests of the two pre-built bundles (gwps, k562_ess)."""
import json

OUT = "/data/resource/wanglab/aivc/aivc2026/outputs/metrics/local_bench/bundles"

for ctx in ["gwps", "k562_ess"]:
    m = json.load(open(f"{OUT}/{ctx}/manifest.json"))
    print(f"=== {ctx} ===")
    print(json.dumps(m, indent=1, default=str)[:1500])
    print()
