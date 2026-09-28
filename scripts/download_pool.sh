#!/bin/bash
# 数据池后台下载：VCC2025 (GCS) + scPerturb (Zenodo)
# 用法: nohup bash scripts/download_pool.sh <gcs|zenodo> > outputs/logs/download_<source>.log 2>&1 &
set -uo pipefail
SRC=${1:?usage: download_pool.sh <gcs|zenodo>}
ROOT=/data/resource/wanglab/aivc/aivc2026
mkdir -p "$ROOT/data/public/vcc2025" "$ROOT/data/public/scperturb" "$ROOT/outputs/logs"

dl() { # dl <url> <dest>
  local url=$1 dest=$2
  if [ -s "$dest.done" ]; then echo "[skip] $dest"; return 0; fi
  echo "[$(date +%F\ %T)] GET $url"
  wget -c -q --tries=5 --timeout=60 -O "$dest.part" "$url" && mv "$dest.part" "$dest" && touch "$dest.done" \
    && echo "[$(date +%F\ %T)] OK  $dest ($(du -h "$dest" | cut -f1))" \
    || echo "[$(date +%F\ %T)] FAIL $dest"
}

if [ "$SRC" = gcs ]; then
  B=https://storage.googleapis.com/arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025
  D=$ROOT/data/public/vcc2025
  dl $B/validation/adata_Validation.h5ad        $D/adata_Validation.h5ad
  dl $B/validation/pert_counts_Validation.csv   $D/pert_counts_Validation.csv
  dl $B/test/adata_Test.h5ad                    $D/adata_Test.h5ad
  dl $B/test/pert_counts_Test.csv               $D/pert_counts_Test.csv
  dl $B/train/adata_Training.h5ad               $D/adata_Training.h5ad
  dl $B/train/pert_counts_Training.csv          $D/pert_counts_Training.csv
  dl $B/gene_names.csv                          $D/gene_names.csv
else
  Z="https://zenodo.org/records/13350497/files"
  D=$ROOT/data/public/scperturb
  dl "$Z/AdamsonWeissman2016_GSM2406681_10X010.h5ad?download=1"      $D/AdamsonWeissman2016.h5ad
  dl "$Z/NormanWeissman2019_filtered.h5ad?download=1"                $D/NormanWeissman2019.h5ad
  dl "$Z/ReplogleWeissman2022_rpe1.h5ad?download=1"                  $D/Replogle2022_rpe1.h5ad
  dl "$Z/ReplogleWeissman2022_K562_essential.h5ad?download=1"        $D/Replogle2022_K562_essential.h5ad
  dl "$Z/NadigOConner2024_hepg2.h5ad?download=1"                     $D/Nadig2024_hepg2.h5ad
  dl "$Z/NadigOConner2024_jurkat.h5ad?download=1"                    $D/Nadig2024_jurkat.h5ad
  dl "$Z/ReplogleWeissman2022_K562_gwps.h5ad?download=1"             $D/Replogle2022_K562_gwps.h5ad
fi
echo "[$(date +%F\ %T)] ALL DONE ($SRC)"
