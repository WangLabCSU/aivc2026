#!/bin/bash
# iter043-63: v0.20 gen→prep 链; gen 完成后自动 prep (vcc 0.3.0 语法)
set -u
export DBUS_SESSION_BUS_ADDRESS=
export TMPDIR=/data/resource/wanglab/aivc/aivc2026/tmp/vcc_prep
VCC=/data/home/wsx/miniconda3/envs/aivc2026/bin/vcc
cd /data/resource/wanglab/aivc/aivc2026
H5=outputs/submissions/20261025_bizard_v20_final_hykens_nogeneric.h5ad
while kill -0 2748548 2>/dev/null; do sleep 60; done
echo "[chain $(date -u +%H:%M)] gen done, starting prep"
timeout 3600 "$VCC" prep "$H5" -g data/raw/controls_final/gene_names.csv --perts data/raw/controls_final/pert_counts.csv
echo "[chain $(date -u +%H:%M)] prep exit=$?"
ls -la outputs/submissions/20261025_bizard_v20_final_hykens_nogeneric.prep.vcc 2>/dev/null || echo "PREP_MISSING"
