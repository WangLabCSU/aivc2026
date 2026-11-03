"""Six-context option_b pre-build: refs for gwps + k562_ess (delta-choice-independent).

Uses src/eval/local_bench.prepare_reference with hand-built ccfg dicts —
configs/local_bench/contexts.yaml (committed read-only) is NOT touched.
Cached: skips if data/interim/local_bench/{ctx}/ref.h5ad already exists.
"""
import sys

sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")

from src.eval.local_bench import prepare_reference

CCFGS = {
    "gwps": {
        "h5ad": "data/public/scperturb/Replogle2022_K562_gwps.h5ad",
        "pert_col": "gene",
        "control": "non-targeting",
        "n_perts": 100,
        "seed": 0,
    },
    "k562_ess": {
        "h5ad": "data/public/scperturb/Replogle2022_K562_essential.h5ad",
        "pert_col": "gene",
        "control": "non-targeting",
        "n_perts": 100,
        "seed": 0,
    },
}

for ctx, ccfg in CCFGS.items():
    ref = prepare_reference(ctx, ccfg, n_cells=400, n_ntc=2000)
    print(f"[prebuild] {ctx}: {ref}", flush=True)
