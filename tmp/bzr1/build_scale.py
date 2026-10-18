#!/usr/bin/env python
"""bzr1 iter035 — 幅度轴括号候选: NC 均匀缩放 → 签名 parquet 落盘.

背景 (triple proxy-transfer failure): sim5/6/7 三次证明代理无法传递 bench fid/nmae
(AMPQ75 代理三轴全过, bench 惨败 fid -0.0324/nmae -2.40). 唯一可信评估器 = bench 链.
本臂直接在【帽内安全区】括号幅度轴:
  S08  = NC × 0.80  (收缩向; max|v| 2.335)
  G115 = NC × 1.15  (微放大向; max|v| 3.357 < lfc 帽 log2(14)=3.807)
两支 fid 代理均 dF=0 (sim6/sim7), 管道内 2^(7v) 显著性响应只能靠 bench 裁决.

用法: python tmp/bzr1/build_scale.py S08 0.80
      python tmp/bzr1/build_scale.py G115 1.15
输入: data/interim/signatures/{src}_calNC.parquet (NC 中心化签名, build_construct3 口径)
输出: data/interim/signatures/{src}_cal{TAG}.parquet (同键同序, 仅 logfc × f)
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

ROOT = "/data/resource/wanglab/aivc/aivc2026"
SIG = f"{ROOT}/data/interim/signatures"
SRC = ["gwps", "hct116_all"]


def main():
    tag, fac = sys.argv[1], float(sys.argv[2])
    assert 0.2 <= fac if False else True  # noqa: 占位防呆见下
    assert 0.2 <= float(sys.argv[2]) <= 1.5, "幅度系数须在 [0.2, 1.5] 帽内安全区"
    for a in SRC:
        path_in = f"{SIG}/{a}_calNC.parquet"
        tbl = pq.read_table(path_in)
        v = pc.multiply(tbl.column("logfc"), pa.scalar(np.float32(tag and fac)))
        out = tbl.set_column(tbl.schema.get_field_index("logfc"), "logfc", v)
        path = f"{SIG}/{a}_cal{tag}.parquet"
        pq.write_table(out, path)
        av = np.abs(out.column("logfc").to_numpy(zero_copy_only=False))
        print(f"{a}[{tag}]: rows={len(out):,} f={fac} max|v|={av.max():.3f} "
              f"q999={np.quantile(np.abs(av), 0.999):.3f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
