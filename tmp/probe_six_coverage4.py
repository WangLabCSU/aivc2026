"""Probe 4: schemas of candidate delta sources."""
import pandas as pd
import pyarrow.parquet as pq

d = pq.read_table("data/interim/bizard/v03/heu_agg_gwps.parquet").slice(0, 5).to_pandas()
print(d)
print(d.dtypes)

d2 = pd.read_parquet("data/interim/bizard/v03/agg_delta_P_H1.parquet").head(3)
print(d2)

d3 = pd.read_parquet("data/interim/bizard/v03/gen2/pred_g2_c3_submit_A.parquet").head(3)
print(d3)
