# iter024 第二轮: signature 链 top_k A/B — bench twin lam70mf14 × top_k ∈ {500,1000,2000}
# 对照 = dualdeep_lam70mf14_Q (无 top_k, pds 0.8156 nmae 0.9830 fid 0.5870 reach 0.4217 jac 0.0762)
# 每档 ~10-15 min (pred 构建 + cell-eval2 + score), 顺序跑, 判据: fid/reach/jac 结构性改善且 nmae 不塌
source /data/home/wsx/miniconda3/etc/profile.d/conda.sh
conda activate aivc2026
export TMPDIR=/data/resource/wanglab/aivc/tmpdir
cd /data/resource/wanglab/aivc/aivc2026

for K in 500 1000 2000; do
  sed "s/^output_stem:.*/output_stem: 20261016_dualsrcdeep_lam70mf14/" \
      configs/l1_signature/bench-dualdeep-lam70-mf14.yaml \
    | sed "/^lambda_real:/a top_k: ${K}" > tmp/bench_topk${K}.yaml
  echo "== top_k=${K} config =="; cat tmp/bench_topk${K}.yaml
  python src/eval/local_bench.py --model-config tmp/bench_topk${K}.yaml \
    --contexts Q_HepG2 --tag dualdeep_lam70mf14_topk${K} \
    > tmp/bench_topk${K}.log 2>&1
  echo "[topk${K} rc=$?] $(date -u)"
done
echo "ALL DONE $(date -u)"
