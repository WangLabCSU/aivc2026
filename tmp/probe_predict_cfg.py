"""Verify cfg["predict"] keys the run() path reads (n_cells_per_pert, max_fold)."""
import sys
sys.path.insert(0, "/data/resource/wanglab/aivc/aivc2026")
from src.models.bizard.swarm import load_cfg
cfg = load_cfg()
p = cfg.get("predict", {})
print("predict keys:", sorted(p.keys()))
print("n_cells_per_pert:", p.get("n_cells_per_pert"))
print("max_fold:", p.get("max_fold"))
# also confirm seed key convention used by build_pred default
print("predict.seed:", p.get("seed"))
