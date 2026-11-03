"""Verify delta column schema for all 7 mapped delta files (read-only)."""
import pyarrow.parquet as pq

V03 = "/data/resource/wanglab/aivc/aivc2026/data/interim/bizard/v03"
FILES = {
    "agg_delta_Q_HepG2": f"{V03}/agg_delta_Q_HepG2.parquet",
    "agg_delta_R_Jurkat": f"{V03}/agg_delta_R_Jurkat.parquet",
    "agg_delta_S_RPE1": f"{V03}/agg_delta_S_RPE1.parquet",
    "agg_delta_P_H1": f"{V03}/agg_delta_P_H1.parquet",
    "gen2/pred_g2_c3_S_HCT116": f"{V03}/gen2/pred_g2_c3_S_HCT116.parquet",
    "gen2/pred_g2_c3_submit_A": f"{V03}/gen2/pred_g2_c3_submit_A.parquet",
    "heu_agg_gwps": f"{V03}/heu_agg_gwps.parquet",
}
for name, p in FILES.items():
    sch = pq.read_schema(p)
    print(f"{name:28s} {sch.names}")
