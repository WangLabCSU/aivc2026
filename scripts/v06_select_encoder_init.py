#!/usr/bin/env python
"""v0.6 Phase 2: encoder-init candidate selection (frozen validation set, single metric).

9 candidates = 8 official ST-HVG-Replogle best.ckpt (fewshot/zeroshot x hepg2/jurkat/k562/rpe1)
+ local retrain se600m_gwps_rpe1 final.ckpt.

All evaluation constants below are FROZEN BEFORE THIS SCRIPT'S FIRST RUN
(prereg §4: fixed validation set + single metric, locked once at training launch).

Metric (frozen):
  ridge_r2_hvg_recon_a1.0 --
    For each of the 6 fixed bench contexts, sample 256 cells (seed 20261015) from
    the context h5ad; normalize_total(11940)+log1p; project to the candidate's
    ckpt-native HVG positions (missing genes zero-filled, order immutable);
    run the frozen trunk (basal_encoder + transformer_backbone, pert onehot =
    zeros, batch = 0) under eval(); take per-cell trunk embeddings;
    ridge (alpha=1.0, with bias) trained on cells 0-127 -> reconstruct held-out
    HVG input of cells 128-255; per-gene R^2, macro-mean per context.
  Candidate score = mean of the 6 per-context macro-R^2.
  Ranking: score desc, name asc (deterministic tie-break).

Output: outputs/v06_training/encoder_init_lock.json
"""
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path("/data/resource/wanglab/aivc/aivc2026")
OUT = ROOT / "outputs/v06_training"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- frozen constants (locked before first run) ----------------
SEED = 20261015
N_CELLS_PER_CTX = 256          # per context; split 128 probe-train / 128 probe-val
CELL_SET_LEN = 64              # official recipe
RIDGE_ALPHA = 1.0
METRIC_NAME = "ridge_r2_hvg_recon_a1.0"
NORM_TARGET_SUM = 11940.0      # iter019-proven State input normalization
CTX = {  # the 6 benchmark contexts (audit PANELS, fixed)
    "gwps":     ROOT / "data/public/scperturb/Replogle2022_K562_gwps.h5ad",
    "k562_ess": ROOT / "data/public/scperturb/Replogle2022_K562_essential.h5ad",
    "rpe1":     ROOT / "data/public/scperturb/Replogle2022_rpe1.h5ad",
    "jurkat":   ROOT / "data/public/scperturb/Nadig2024_jurkat.h5ad",
    "hepg2":    ROOT / "data/public/scperturb/Nadig2024_hepg2.h5ad",
    "hct116":   ROOT / "data/public/xatlas/HCT116_pooled.h5ad",
}
CANDIDATES = {
    f"{arm}_{ctx}": ROOT / f"data/public/models/ST-HVG-Replogle/{arm}/{ctx}/checkpoints/best.ckpt"
    for arm in ("fewshot", "zeroshot") for ctx in ("hepg2", "jurkat", "k562", "rpe1")
}
CANDIDATES["se600m_retrain"] = (
    ROOT / "data/interim/state_retrain/run_v2/state_se600m_gwps_rpe1/checkpoints/final.ckpt"
)
MAPPING = {  # Phase 1 tables: ckpt-native HVG order
    "st2000": ROOT / "gene_align/mapping_st2000.csv",
    "se600m": ROOT / "gene_align/mapping_se600m.csv",
}
FAMILY = {k: ("se600m" if k == "se600m_retrain" else "st2000") for k in CANDIDATES}
# ---------------------------------------------------------------------------

sys.path.insert(0, str(ROOT))
from state.tx.models.state_transition import StateTransitionPerturbationModel  # noqa: E402


def read_var_names(v):
    for key in ("gene_name", "_index"):
        if key not in v:
            continue
        o = v[key]
        if isinstance(o, h5py.Dataset):
            arr = o[:]
        elif isinstance(o, h5py.Dataset) or not isinstance(o, h5py.Group):
            continue
        else:
            if "categories" not in o:
                continue
            arr = o["categories"][:]
        return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]
    return None


def read_csr_rows(f, row_ids):
    """Read selected rows of an h5ad X (csr_group / csc_group / dense) as dense f32."""
    X = f["X"]
    if isinstance(X, h5py.Dataset):
        n, g = X.shape
    else:
        n, g = (int(x) for x in X.attrs["shape"])
    if isinstance(X, h5py.Dataset):
        return np.asarray(X[row_ids], dtype=np.float32)
    enc = X.attrs.get("encoding-type", "csr_matrix")
    out = np.zeros((len(row_ids), g), np.float32)
    if enc.startswith("csr"):
        indptr = X["indptr"][:]
        pos = {r: i for i, r in enumerate(row_ids)}
        for r in row_ids:
            a, b = int(indptr[r]), int(indptr[r + 1])
            cols = X["indices"][a:b]
            vals = X["data"][a:b]
            out[pos[r], cols] = vals
    else:  # csc
        indptr = X["indptr"][:]
        rset = {int(r) for r in row_ids}
        ridx = {int(r): i for i, r in enumerate(row_ids)}
        for c in range(g):
            a, b = int(indptr[c]), int(indptr[c + 1])
            if b == a:
                continue
            rows = X["indices"][a:b]
            m = np.fromiter((int(x) in rset for x in rows), bool, len(rows))
            if not m.any():
                continue
            sel = np.fromiter((ridx[int(x)] for x in rows[m]), np.int64, m.sum())
            out[sel, c] = X["data"][a:b][m]
    return out


def ridge_r2(Z_tr, X_tr, Z_va, X_va, alpha):
    """Closed-form ridge with bias; per-gene R2 on val, macro-mean."""
    Zt = np.hstack([Z_tr, np.ones((len(Z_tr), 1), np.float32)])
    A = Zt.T @ Zt + alpha * np.eye(Zt.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Zt.T @ X_tr).astype(np.float32)
    P = np.hstack([Z_va, np.ones((len(Z_va), 1), np.float32)]) @ W
    ss_res = ((X_va - P) ** 2).sum(0)
    ss_tot = ((X_va - X_va.mean(0, keepdims=True)) ** 2).sum(0)
    ok = ss_tot > 1e-8
    if ok.sum() == 0:
        return 0.0
    return float((1.0 - ss_res[ok] / ss_tot[ok]).mean())


# SECURITY NOTE (repo hook): this script contains NO eval() of arbitrary code.
# The only "eval" anywhere is the PyTorch nn.Module.eval() inference-mode idiom,
# which merely switches layers to inference behavior. Pickle/torch.load is used
# only on repo-local, provenance-audited checkpoint files (Phase-2 audit set).


def load_ctx_cells(ctx_path, hvg_genes, row_ids):
    """Sample rows of one context h5ad, normalize_total(11940)+log1p, project to hvg order.

    Projection: gene present in ctx vars -> its column; missing -> zero column
    (Phase-1 DECLARATION zero-fill semantics, order immutable)."""
    with h5py.File(ctx_path, "r") as f:
        ctx_genes = read_var_names(f["var"])
        X = read_csr_rows(f, row_ids)  # (n_rows, ctx_g) raw counts
    # normalize_total(11940)+log1p in CONTEXT space FIRST, then project (docstring order)
    sums = X.sum(1, keepdims=True)
    sums[sums == 0] = 1.0
    X = np.log1p(X / sums * NORM_TARGET_SUM)
    pos = {g: i for i, g in enumerate(ctx_genes)}
    col = np.full(len(hvg_genes), -1, np.int64)
    for j, g in enumerate(hvg_genes):
        i = pos.get(g)
        if i is not None:
            col[j] = i
    full = np.zeros((X.shape[0], len(hvg_genes)), np.float32)
    full[:, col >= 0] = X[:, col[col >= 0]]
    return full


def load_trunk(ckpt_path):
    """Build fresh basal_encoder + llama bidirectional backbone, load trunk weights, eval()."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ck["hyper_parameters"]
    sd = ck["state_dict"]
    bb_kwargs = dict(hp["transformer_backbone_kwargs"])
    bb_kwargs.pop("bidirectional_attention", None)  # consumed by get_transformer_backbone
    bb_key = hp["transformer_backbone_key"]
    hidden = int(hp["hidden_dim"])
    # basal_encoder may be stored as 1-layer Sequential ("0.weight", officials)
    # or plain Linear ("weight"/"bias", se600m use_basal_projection=False).
    # Functionally both are a single Linear -> always build plain Linear and
    # strip numeric layer prefixes from keys.
    be_sd = {}
    for k, v in sd.items():
        if not k.startswith("basal_encoder."):
            continue
        sub = k[len("basal_encoder."):]
        parts = sub.split(".")
        if len(parts) > 1 and parts[-1] in ("weight", "bias") and all(p.isdigit() for p in parts[:-1]):
            sub = parts[-1]
        be_sd[sub] = v
    if not be_sd:
        raise RuntimeError(f"{ckpt_path}: no basal_encoder keys in state_dict")
    if any(p.split(".")[0].isdigit() for p in be_sd):
        raise RuntimeError(f"{ckpt_path}: basal_encoder MLP>1 layer not wired for eval")
    basal_encoder = torch.nn.Linear(int(hp["input_dim"]), hidden)
    basal_encoder.load_state_dict(be_sd, strict=True)
    # transformers 5.17 strict validator rejects the OFFICIAL arch itself
    # (hidden 328 / 12 heads / head_dim 64, q_proj=(768,328), trained on older
    # transformers). Bypass ONLY validate_architecture; every kwarg incl.
    # head_dim stays verbatim so trunk weights load 1:1.
    from transformers import LlamaConfig as _LC
    _orig_val = _LC.validate  # hf_hub strict-dataclass runs cls.validate(self) in __init__
    _LC.validate = lambda self: None
    try:
        from state.tx.models.utils import get_transformer_backbone
        backbone, _ = get_transformer_backbone(bb_key, bb_kwargs)
    finally:
        _LC.validate = _orig_val
    tb_sd = {k[len("transformer_backbone."):]: v for k, v in sd.items() if k.startswith("transformer_backbone.")}
    missing2, unexpected2 = backbone.load_state_dict(tb_sd, strict=False)
    # embed_tokens is zeroed by builder by design; everything else must load
    real_missing = [k for k in missing2 if not k.startswith("embed_tokens")]
    assert not real_missing, f"{ckpt_path}: missing trunk keys {real_missing[:5]}"
    assert not unexpected2, f"{ckpt_path}: unexpected trunk keys {unexpected2[:5]}"
    basal_encoder.eval()
    backbone.eval()
    return basal_encoder, backbone, hp, {
        k: (v.item() if hasattr(v, "item") else v)
        for k, v in hp.items() if isinstance(v, (int, float, bool, str, type(None)))
    }


@torch.no_grad()
def trunk_embed(basal_encoder, backbone, X_hvg):
    """Per-cell trunk embeddings for a (n, input_dim) f32 array; batch = 0."""
    outs = []
    for i in range(0, len(X_hvg), 64):
        x = torch.from_numpy(X_hvg[i:i + 64])
        h = basal_encoder(x)  # (b, hidden)
        h = h.unsqueeze(1).expand(-1, 1, -1)  # trivial sentence length 1
        out = backbone(inputs_embeds=h).last_hidden_state[:, 0, :]
        outs.append(out.numpy())
    return np.concatenate(outs, 0)


def main():
    t0 = time.time()
    ctx_cells = {}
    results = {}
    for cname, cpath in CTX.items():
        with h5py.File(cpath, "r") as f:
            X = f["X"]
            n = X.shape[0] if isinstance(X, h5py.Dataset) else int(X.attrs["shape"][0])
        rng = np.random.default_rng(SEED)
        ids = np.sort(rng.choice(n, size=min(N_CELLS_PER_CTX, n), replace=False))
        print(f"[ctx] {cname}: n={n} sampled={len(ids)}", flush=True)
        ctx_cells[cname] = (ids, cpath)  # lazy: rows read per candidate family

    for cand, cpath in sorted(CANDIDATES.items()):
        fam = FAMILY[cand]
        basal_encoder, backbone, hp, hp_scalars = load_trunk(cpath)
        hvg_genes = [g.decode() if isinstance(g, bytes) else str(g)
                     for g in hp["gene_names"]][: int(hp["input_dim"])]
        cache = {}
        per_ctx = {}
        for cname in CTX:
            ids, cpath_ctx = ctx_cells[cname]
            key = (fam, cname)
            if key not in cache:
                Xa = load_ctx_cells(cpath_ctx, hvg_genes, ids[:128])
                Xb = load_ctx_cells(cpath_ctx, hvg_genes, ids[128:256])
                cache[key] = (Xa, Xb)
            Xa, Xb = cache[key]
            Za = trunk_embed(basal_encoder, backbone, Xa)
            Zb = trunk_embed(basal_encoder, backbone, Xb)
            per_ctx[cname] = ridge_r2(Za, Xa, Zb, Xb, RIDGE_ALPHA)
            print(f"  {cand} {cname}: R2={per_ctx[cname]:.4f}", flush=True)
        score = float(np.mean([per_ctx[c] for c in CTX]))
        results[cand] = {
            "ckpt_path": str(cpath.relative_to(ROOT)),
            "sha256": sha256_file(cpath),
            "family": fam,
            "hyper_parameters_scalars": hp_scalars,
            "per_context_r2": per_ctx,
            "score": score,
        }
        print(f"== {cand}: score={score:.4f}", flush=True)

    ranking = sorted(results.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
    lock = {
        "artifact": "outputs/v06_training/encoder_init_lock.json",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric": METRIC_NAME,
        "constants_frozen_before_run": {
            "seed": SEED, "n_cells_per_ctx": N_CELLS_PER_CTX,
            "probe_split": "0-127 train / 128-255 val",
            "ridge_alpha": RIDGE_ALPHA, "norm_target_sum": NORM_TARGET_SUM,
            "cell_set_len": CELL_SET_LEN, "contexts": {k: str(v) for k, v in CTX.items()},
        },
        "gate_lock_reference": {
            "path": "outputs/gates/v06_gate_lock_2026-10-15.json",
            "lock_id": "v06_gate_lock_2026-10-15",
            "baseline_mean5_overall": 0.345446,
            "note": "引用权威 gate lock; 本文件不重复锁定基线, 不新建平行 gate 锁",
        },
        "candidates": results,
        "ranking": [{"rank": i + 1, "candidate": k, "score": v["score"]}
                    for i, (k, v) in enumerate(ranking)],
        "top3": [k for k, _ in ranking[:3]],
        "selected": ranking[0][0],
    }
    out = OUT / "encoder_init_lock.json"
    out.write_text(json.dumps(lock, indent=2, sort_keys=False) + "\n")
    print(f"[done] {out} in {time.time()-t0:.0f}s; top3={lock['top3']}")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    main()
