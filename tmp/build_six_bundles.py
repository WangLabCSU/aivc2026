"""Six-context option_b pre-build: real bundles for gwps + k562_ess.

Inline 2-step cell-eval2 recipe (same as bench_score.build_bundle) but WITHOUT
build_bundle() itself — its _cfg_for reads configs/local_bench/contexts.yaml,
which does not contain gwps/k562_ess and is committed read-only.
Control = "non-targeting" for both new contexts.
Cached: skips if outputs/metrics/local_bench/bundles/{ctx}/manifest.json exists.
"""
import subprocess
from pathlib import Path

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
ENV_BIN = Path("/data/home/wsx/miniconda3/envs/aivc2026/bin")
BENCH = ROOT / "data/interim/local_bench"
OUT_DIR = ROOT / "outputs/metrics/local_bench"
CFG = ["--preset", "vcc2026", "--pert-col", "target_gene",
       "--input-type", "counts", "--set", "de.backend=pdex",
       "--control", "non-targeting"]


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


for ctx in ["gwps", "k562_ess"]:
    bundle = OUT_DIR / "bundles" / ctx
    if (bundle / "manifest.json").exists():
        print(f"[bundle] {ctx}: exists, skip", flush=True)
        continue
    ref = BENCH / ctx / "ref.h5ad"
    build = OUT_DIR / "bundle_build"
    build.mkdir(parents=True, exist_ok=True)
    run([str(ENV_BIN / "cell-eval2"), "baseline", "-ar", str(ref), *CFG,
         "-o", str(build / f"{ctx}_baseline"),
         "--save-pred", str(build / f"{ctx}_baseline_pred.h5ad")])
    run([str(ENV_BIN / "cell-eval2"), "prep-real-bundle", "--real", str(ref),
         "--baseline", str(build / f"{ctx}_baseline_pred.h5ad"),
         "-o", str(bundle), "--id", f"{ctx}_local", "--force", *CFG])
    print(f"[bundle] {ctx}: built", flush=True)
