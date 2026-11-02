"""Step 1: validate the Seattle loop speed matrix (timestamps, sensors, missing/imputed values)."""
import re

import numpy as np
import pandas as pd

from src.common import C, log, out, save_json, load_speed_raw


def flatline_fraction(x, min_run):
    """Fraction of samples that sit inside runs of identical values of length >= min_run."""
    x = np.asarray(x)
    same = np.r_[False, x[1:] == x[:-1]]
    run_id = np.cumsum(~same)
    run_len = np.bincount(run_id)[run_id]
    return float((run_len >= min_run).mean())


def main():
    df = load_speed_raw()
    idx = df.index
    rep = {"n_timestamps": len(idx), "n_sensors": df.shape[1],
           "start": idx.min(), "end": idx.max()}

    # --- timestamps
    full = pd.date_range(idx.min().floor("D"), idx.max().ceil("D") - pd.Timedelta("5min"), freq="5min")
    diffs = pd.Series(idx[1:] - idx[:-1]).value_counts().head(5)
    rep["step_counts"] = {str(k): int(v) for k, v in diffs.items()}
    rep["missing_timestamps"] = int(len(full.difference(idx)))
    rep["unexpected_timestamps"] = int(len(idx.difference(full)))

    # Local-vs-UTC heuristic: free flow should be fastest around 03-05 local, not 10-12.
    wk = df[idx.dayofweek < 5].mean(axis=1)
    by_hour = wk.groupby(wk.index.hour).mean()
    slow, fast = int(by_hour.idxmin()), int(by_hour.idxmax())
    looks_local = (slow in (6, 7, 8, 9, 15, 16, 17, 18)) and (fast in (0, 1, 2, 3, 4, 5, 20, 21, 22, 23))
    rep["hourly_mean_speed_weekdays"] = by_hour.round(2).to_dict()
    rep["tz_verdict"] = (f"index looks LOCAL (slowest {slow}h, fastest {fast}h)" if looks_local else
                         f"CHECK: slowest {slow}h, fastest {fast}h is not a local-time pattern -> maybe UTC; "
                         "set TRAFFIC_INDEX_IS_UTC")
    rep["slowest_weekday_hours"] = by_hour.nsmallest(3).index.tolist()

    # --- values
    v = df.to_numpy()
    rep["nan_values"] = int(np.isnan(v).sum())
    rep["nonpositive_values"] = int((v <= 0).sum())
    rep["values_above_90mph"] = int((v > 90).sum())
    rep["share_integer_values"] = float(np.mean(np.isclose(v[~np.isnan(v)], np.round(v[~np.isnan(v)]))))

    # --- per-sensor quality
    rows = []
    for c in df.columns:
        x = df[c].to_numpy()
        m = re.match(r"^([di])(\d{3})(..)(\d{5})$", c)
        rows.append({
            "sensor": c,
            "direction": m.group(1) if m else None,
            "route": m.group(2) if m else None,
            "milepost": int(m.group(4)) / 100 if m else None,
            "nan_frac": float(np.isnan(x).mean()),
            "flatline_frac": flatline_fraction(np.nan_to_num(x, nan=-1), C.FLATLINE_STEPS),
            "mean": float(np.nanmean(x)), "p05": float(np.nanpercentile(x, 5)),
            "p95": float(np.nanpercentile(x, 95)),
        })
    q = pd.DataFrame(rows)
    q["keep"] = (q.flatline_frac <= C.MAX_FLATLINE_FRAC) & (q.nan_frac < 0.05)
    q.to_csv(out("sensor_quality.csv"), index=False)
    keep = q.loc[q.keep, "sensor"].tolist()
    save_json(keep, "sensors_keep.json")
    rep["sensors_kept"] = len(keep)
    rep["sensors_dropped"] = q.loc[~q.keep, "sensor"].tolist()
    rep["median_flatline_frac"] = float(q.flatline_frac.median())

    # --- adjacency / node list
    a_path = C.RAW_DIR / C.ADJ_FILE
    if a_path.exists():
        A = np.load(a_path)
        rep["adjacency_shape"] = list(A.shape)
        rep["adjacency_matches_sensors"] = A.shape[0] == df.shape[1]
        rep["adjacency_symmetric"] = bool(np.allclose(A, A.T))
        off = (A > 0).astype(int)
        np.fill_diagonal(off, 0)
        rep["adjacency_degree_min_median_max"] = [int(off.sum(1).min()), float(np.median(off.sum(1))), int(off.sum(1).max())]
    n_path = C.RAW_DIR / C.NODES_FILE
    if n_path.exists():
        nodes = pd.read_csv(n_path)
        rep["nodes_rows"] = len(nodes)
        rep["nodes_columns"] = list(nodes.columns)
        first_col = nodes.iloc[:, -1].astype(str).tolist()
        rep["nodes_order_matches_columns"] = first_col == list(df.columns)

    rep["notes"] = [
        "Zenodo record 3258904 states the matrix was quality-controlled and IMPUTED by STAR Lab; "
        "true outages are not marked. flatline_frac is only a proxy for imputed/stuck periods.",
        "Terms of use: README asks to cite Cui et al. (2018, 2019); check the licence on the Zenodo/download page.",
    ]
    save_json(rep, "01_validation_report.json")
    log.info("Timestamps: %s -> %s, %d rows, missing %d", rep["start"], rep["end"], len(idx), rep["missing_timestamps"])
    log.info("Sensors kept %d / %d; NaN values %d; integer share %.2f", len(keep), df.shape[1], rep["nan_values"], rep["share_integer_values"])
    log.info("TZ heuristic: %s", rep["tz_verdict"])


if __name__ == "__main__":
    main()
