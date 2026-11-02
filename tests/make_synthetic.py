"""Create a small synthetic dataset with the same file formats, for a smoke test of the pipeline.
Usage:  python tests/make_synthetic.py  &&  SRP_RAW_DIR=tests/synth SRP_OUT_DIR=tests/synth_out python run_all.py"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
d = Path(__file__).resolve().parent / "synth"
d.mkdir(exist_ok=True)

idx = pd.date_range("2015-01-01", "2015-12-31 23:55", freq="5min")
T, N = len(idx), 24
cols = [f"i005es{16000 + 50 * k:05d}" if k < 12 else f"d405es{1000 + 50 * k:05d}" for k in range(N)]

# hourly rain (UTC), more in winter
hours = pd.date_range("2015-01-01 08:00", "2016-01-01 07:00", freq="h", tz="UTC")
month = hours.tz_convert("America/Los_Angeles").month
p_start = np.where(np.isin(month, [10, 11, 12, 1, 2, 3]), 0.025, 0.006)
rain = np.zeros(len(hours))
k = 0
while k < len(hours):
    if rng.random() < p_start[k]:
        L = rng.integers(2, 12)
        rain[k:k + L] = rng.gamma(1.2, 1.5, size=min(L, len(hours) - k))
        k += L
    k += 1
rows = []
for h, r in zip(hours, rain):
    ts = (h - pd.Timedelta(minutes=7)).strftime("%Y-%m-%dT%H:%M:%S")
    aa = f"01,{int(round(r * 10)):04d},9,5" if r >= 0.25 else ""
    rows.append({"STATION": "72793024233", "DATE": ts, "SOURCE": 7, "REPORT_TYPE": "FM-15", "AA1": aa})
    if rng.random() < 0.05:
        rows.append({"STATION": "72793024233", "DATE": ts.replace(":53:", ":20:"), "SOURCE": 7,
                     "REPORT_TYPE": "FM-16", "AA1": "01,9999,9,9"})
pd.DataFrame(rows).to_csv(d / "72793024233.csv", index=False)

# rain at 5-min resolution in local naive time
rain_local = pd.Series(rain, index=hours.tz_convert("America/Los_Angeles").tz_localize(None))
rain_local = rain_local[~rain_local.index.duplicated()].reindex(idx, method="ffill").fillna(0).to_numpy()

tod = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
wkday = idx.dayofweek.to_numpy() < 5
peak = wkday * (12 * np.exp(-((tod - 8) ** 2) / 1.5) + 15 * np.exp(-((tod - 17) ** 2) / 2))
sens_peak = rng.uniform(0.3, 1.5, N)
corridor = np.array([0] * 12 + [1] * 12)
common = np.zeros((T, 2))
for t in range(1, T):  # corridor-level AR(1) shocks -> correlated errors
    common[t] = 0.97 * common[t - 1] + rng.normal(0, 0.8, 2)
rain_eff = (3 + 1.5 * np.sqrt(rain_local))[:, None] * (rain_local > 0)[:, None] * rng.uniform(0.5, 1.5, N)
idio = rng.normal(0, 1.5, (T, N))
V = 62 - peak[:, None] * sens_peak - rain_eff * (1 + peak[:, None] / 10) + common[:, corridor] + idio
V = np.clip(np.round(V), 5, 75).astype("float32")
V[20000:50000, 5] = V[20000, 5]  # a stuck sensor that validation should drop
pd.DataFrame(V, index=idx, columns=cols).to_pickle(d / "speed_matrix_2015")

A = np.eye(N)
for i in range(N - 1):
    if corridor[i] == corridor[i + 1]:
        A[i, i + 1] = A[i + 1, i] = 1
np.save(d / "Loop_Seattle_2015_A.npy", A)
pd.DataFrame({"milepost": cols}).to_csv(d / "nodes_loop_mp_list.csv", index=False)
print("synthetic data written to", d)
