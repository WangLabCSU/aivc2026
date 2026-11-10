# cell_eval2

Evaluation of perturbation-prediction models at single-cell resolution: a GPU-accelerated 
streamlined reimplementation of [`cell-eval`](https://github.com/ArcInstitute/cell-eval).

- **[Tutorial](docs/tutorial.md)** — a runnable tour: metric selection, caching, CPU/GPU, streaming.
- **[Metrics reference](docs/metrics.md)** — the definition of every metric in the catalog.
- **[DE tables](docs/de_tables.md)** — what "unthresholded" means, the DE columns and row order,
  and the `score_cellstream(write_de=True)` output contract.
- **[Ground-truth DE cache](docs/ground_truth_cache.md)** — the content-addressed, immutable cache
  of the real-DE table on the `score_cellstream` path: identity, on-disk format, policies, CLI.
- **[vcc2026 metric specification](docs/vcc2026_metrics/)** — what the 2026 competition scores, and
  how. Start here if you are entering: the
  [full document](docs/vcc2026_metrics/vcc2026-metrics.pdf) carries the reasoning, the
  [abridged edition](docs/vcc2026_metrics/vcc2026-metrics-brief.pdf) is definitions and measured
  values only.
- **[v1 parity notes](docs/parity/v1-parity-divergences.md)** — every accepted behavioural
  difference from upstream `cell-eval`, and why each is left alone.

## Install

Python >= 3.11. The base install is **CPU-only** and needs no GPU, no CUDA and nothing private:

```bash
pip install cell-eval2
cell-eval2 --help
```

| extra | adds | when you need it |
|---|---|---|
| *(none)* | — | everything except GPU DE and out-of-core input |
| `[gpu]` | `cupy-cuda12x`, `nvidia-nvcomp-cu12<6` | CUDA pseudobulk / distance / rank kernels |
| `[gpudge]` | [`gpudge`](https://github.com/ArcInstitute/gpudge) | GPU differential expression |
| `[scale]` | [`cellstream`](https://github.com/ArcInstitute/cellstream) | scoring `.csad` archives out of core |

Combine them: `pip install 'cell-eval2[gpu,gpudge,scale]'`.

### With and without a GPU

- **No GPU** — everything works except GPU DE. Expression, PDS and delta metrics run on NumPy; DE
  runs on a CPU engine (`pdex`, or `scanpy` if pdex is absent).
- **With a GPU** — add `[gpu]` for the kernels and `[gpudge]` for DE. `de.backend="auto"` then
  selects `gpudge`.
- ⚠️ `auto` is **tiered, not a silent fallback**. On a host where a CUDA device is visible but no
  GPU DE engine is installed it **raises**: quietly computing DE on a CPU engine there would move
  every DE number with no signal. With no CUDA device it falls back and warns once. Set
  `de.backend` explicitly (`"pdex"`, `"scanpy"`) to pin an engine and silence it.
- ⚠️ **`[gpudge]` pulls `torch`, and the wheel must match your driver.** A default-index torch can
  be unusable on a driver where cupy works fine. Install with
  `uv pip install --torch-backend=auto 'cell-eval2[gpudge]'` (uv picks the build for the detected
  driver), or name the index yourself:
  `pip install 'cell-eval2[gpudge]' --extra-index-url https://download.pytorch.org/whl/cu126`.

### Optional dependencies

Both are public Arc packages on PyPI. Both are optional; neither is needed for the CPU path.

- **[`cellstream`](https://github.com/ArcInstitute/cellstream)** (`[scale]`, `>=0.9.1`) — the
  `SHPK` archive format and its readers, for scoring datasets too large to hold in memory.
  cell_eval2 reads its **cell layout** (`.csad`, one row group per perturbation).
  ⚠️ Prebuilt wheels cover **CPython 3.11 and 3.12 only**; on 3.13 the resolver falls through to
  the source distribution and needs a Rust toolchain, so prefer 3.11/3.12 for this extra.
- **[`gpudge`](https://github.com/ArcInstitute/gpudge)** (`[gpudge]`, pinned exactly; see `pyproject.toml`) — the GPU
  differential-expression engine. It computes fold changes and p-values on-GPU, and is what
  `de.backend="auto"` selects when a CUDA device is present. Kept separate from `[gpu]` because it
  pulls `torch`, which most users of the kernels do not want.

## Quickstart

```python
from cell_eval2 import compute_metrics, EvalConfig

# expression / PDS metrics need only the (pred, real) pair
df = compute_metrics("pred.h5ad", "real.h5ad", metrics=["expr_mae", "pds_cosine"])

# a metric profile
cfg = EvalConfig(metrics="de", pert_col="target_gene", control="non-targeting",
                 input_type="counts")
df = compute_metrics("pred.h5ad", "real.h5ad", config=cfg)

# the competition: the preset carries every vcc2026 parameter
from dataclasses import replace
cfg = replace(EvalConfig.from_preset("vcc2026"), pert_col="target_gene")
df = compute_metrics("pred.h5ad", "real.h5ad", config=cfg)
```

```bash
# a metric profile
cell-eval2 run -ap pred.h5ad -ar real.h5ad \
  --profile de --pert-col target_gene --control non-targeting --input-type counts -o out/

# the competition: --preset carries every vcc2026 parameter, not just the metric list
cell-eval2 run -ap pred.h5ad -ar real.h5ad --preset vcc2026 --pert-col target_gene -o out/
```

`--preset` (`vcc2026`, `v1`, `v2`, `cell-eval-0.7.6`) and `--config` set the config *base*;
explicit flags and `--set KEY.PATH=VALUE` override it.

## Metrics

Four classes, normally selected by profile rather than one at a time:

- **Expression error** (`expr_*`) — how far the predicted profile sits from the measured one,
  including the sampling-noise-corrected forms the 2026 competition scores.
- **Perturbation discrimination** (`pds_*`) — is a predicted profile closer to *its own* measured
  profile than to any other perturbation's? A rank, so it measures separability rather than
  distance. `pds_l1` / `pds_l2` / `pds_cosine` differ only in the distance used.
-