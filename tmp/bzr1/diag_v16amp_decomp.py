#!/usr/bin/env python
"""iter037-续7 — bzr1v16hy8amp (0.0876) vs HY8-family 分量分解 + heur 公式反推.

问题: amp_max_5src 在 sim8c d8 +0.0037 首选, 为何 bench heur 掉到 0.0876 (< v11nc 0.0859? 不, >0.0859 但 < HY8 0.0915)?
分量: pds 0.5092 (HY12 0.5083 邻域), nmae -0.0559, fid -0.0563, reach 0.1272, jac 0.0016.
对照: v10hcx 0.0896 (pds 0.596 / nmae -0.1093 / fid -0.072 / reach 0.1218)
      v11nc  0.0859 (pds 0.5314 / nmae -0.0846 / fid -0.0601 / reach 0.1284)
      bzr1v16hy8amp 0.0876 (pds 0.5092 / nmae -0.0559 / fid -0.0563 / reach 0.1272)
nmae 更好 (-0.0559 vs -0.0846), pds 更差 (0.5092 vs 0.5314)。
"""
import json
cands = {
    "v10hcx": "tmp/pred_v10hcx.json",
    "v11nc": "tmp/pred_v11nc.json",
    "bzr1v16hy8amp": "tmp/pred_bzr1v16hy8amp.json",
    "v09bsg100": "tmp/pred_v09bsg100.json",
    "v09bsg50": "tmp/pred_v09bsg50.json",
    "v09cmg75": "tmp/pred_v09cmg75.json",
    "v09cmg90": "tmp/pred_v09cmg90.json",
}
import os
for k, p in cands.items():
    if not os.path.exists(p):
        continue
    d = json.load(open(p))
    print(f"{k:16s} overall={d['overall_pred']:.4f} pds={d['pds']:.4f} nmae={d['nmae']:.4f} "
          f"fid={d['fid']:.4f} reach={d['reach']:.4f} jac={d['jac']:.4f}")
