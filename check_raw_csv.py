"""Compare loop_20150101_20151231.csv (long format) with speed_matrix_2015.
Run from the project root: python check_raw_csv.py"""
from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
COLS = ["ts", "cab", "dir", "speed", "volume", "occ"]

mat = pd.read_pickle(RAW / "speed_matrix_2015")
mat.index = pd.to_datetime(mat.index)
mat.columns = [str(c) for c in mat.columns]
V = mat.to_numpy(dtype="float32")
T, N = V.shape
observed = np.zeros((T, N), dtype=bool)

rows = matched = 0
csv_sensors, unmatched_sensors = set(), set()
abs_diff_sum, n_close, n_cmp = 0.0, 0, 0
ranges = {c: [np.inf, -np.inf] for c in ("speed", "volume", "occ")}

for ch in pd.read_csv(RAW / "loop_20150101_20151231.csv", header=None, names=COLS,
                      dtype={"cab": str, "dir": str}, chunksize=2_000_000):
    rows += len(ch)
    ch["sensor"] = ch["dir"].str.strip() + ch["cab"].str.strip()
    csv_sensors.update(ch["sensor"].unique())
    ti = mat.index.get_indexer(pd.to_datetime(ch["ts"], format="%Y-%m-%d %H:%M:%S.%f"))
    ci = mat.columns.get_indexer(ch["sensor"])
    ok = (ti >= 0) & (ci >= 0)
    unmatched_sensors.update(ch.loc[ci < 0, "sensor"].unique())
    matched += ok.sum()
    observed[ti[ok], ci[ok]] = True
    d = np.abs(V[ti[ok], ci[ok]] - ch["speed"].to_numpy()[ok])
    d = d[~np.isnan(d)]
    abs_diff_sum += d.sum(); n_close += (d <= 0.5).sum(); n_cmp += len(d)
    for c in ranges:
        ranges[c] = [min(ranges[c][0], ch[c].min()), max(ranges[c][1], ch[c].max())]
    print(f"  read {rows:,} rows", end="\r")

cov = observed.mean(0)
print(f"\nCSV rows:                  {rows:,}")
print(f"Matrix cells (T x N):      {T:,} x {N} = {T*N:,}")
print(f"CSV sensors:               {len(csv_sensors)}  (not in matrix: {len(unmatched_sensors)})")
print(f"Matrix cells with CSV row: {observed.mean():.2%}  -> the rest exists only in the matrix (imputed)")
print(f"Sensor coverage min/median/max: {cov.min():.2%} / {np.median(cov):.2%} / {cov.max():.2%}")
print(f"Sensors with coverage < 90%: {(cov < 0.9).sum()}")
print(f"Speed |matrix - csv|: mean {abs_diff_sum / max(n_cmp, 1):.3f}, share within 0.5 mph {n_close / max(n_cmp, 1):.2%}")
print("Ranges:", {k: [round(float(a), 3), round(float(b), 3)] for k, (a, b) in ranges.items()})
print("Example unmatched sensors:", sorted(unmatched_sensors)[:10])
pd.DataFrame({"sensor": mat.columns, "csv_coverage": cov}).sort_values("csv_coverage") \
  .to_csv("outputs/raw_csv_coverage.csv", index=False)
np.save("outputs/raw_observed_mask.npy", observed)
print("Saved outputs/raw_csv_coverage.csv and outputs/raw_observed_mask.npy")
