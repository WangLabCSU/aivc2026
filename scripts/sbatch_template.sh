#!/bin/bash
#SBATCH --job-name=aivc2026
#SBATCH --partition=gpu            # 按集群实际分区调整
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/data/resource/wanglab/aivc/aivc2026/outputs/logs/%x-%j.out

# 用法: sbatch scripts/sbatch_template.sh <python 脚本及其参数>
set -euo pipefail
cd /data/resource/wanglab/aivc/aivc2026

source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026

echo "== host=$(hostname) start=$(date) =="
srun python "$@"
echo "== done=$(date) =="
