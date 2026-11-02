"""Step 2: NOAA ISD hourly precipitation for Sea-Tac, rain episodes, leakage-safe as-of features."""
import urllib.request

import numpy as np
import pandas as pd

from src.common import C, log, out, save_json, load_speed


def get_isd():
    p = C.RAW_DIR / C.WEATHER_FILE
    if not p.exists() and C.WEATHER_AUTO_DOWNLOAD:
        log.info("Downloading %s", C.ISD_URL)
        try:
            urllib.request.urlretrieve(C.ISD_URL, p)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Download failed ({e}). Download manually:\n  {C.ISD_URL}\nand save as {p}")
    if not p.exists():
        raise SystemExit(f"Missing {p}; download {C.ISD_URL}")
    return pd.read_csv(p, low_memory=False)


def parse_aa(s):
    """ISD AA1 'period_hours,depth(mm*10),condition,quality' -> (hours, mm)."""
    if not isinstance(s, str) or "," not in s:
        return np.nan, np.nan
    parts = s.split(",")
    try:
        hrs, depth = int(parts[0]), int(parts[1])
    except ValueError:
        return np.nan, np.nan
    if depth == 9999 or hrs == 99:
        return np.nan, np.nan
    return hrs, depth / 10.0


def hourly_precip(raw):
    raw = raw[raw["REPORT_TYPE"].astype(str).str.strip() == "FM-15"].copy()
    raw["obs_utc"] = pd.to_datetime(raw["DATE"], utc=True)
    aa_cols = [c for c in ("AA1", "AA2", "AA3", "AA4") if c in raw.columns]
    p = np.full(len(raw), np.nan)
    for col in aa_cols:  # take the first precip group with a 1-hour accumulation period
        parsed = raw[col].map(parse_aa)
        hrs = parsed.map(lambda x: x[0]).to_numpy(dtype=float)
        mm = parsed.map(lambda x: x[1]).to_numpy(dtype=float)
        take = np.isnan(p) & (hrs == 1)
        p[take] = mm[take]
    raw["p_mm"] = p
    n_missing = int(raw["p_mm"].isna().sum())
    raw["p_mm"] = raw["p_mm"].fillna(0.0)  # FM-15 without a 1-h precip group => no measurable precip
    raw["obs"] = raw["obs_utc"].dt.tz_convert(C.LOCAL_TZ).dt.tz_localize(None)
    w = raw[["obs", "p_mm"]].drop_duplicates("obs").sort_values("obs").set_index("obs")
    return w, n_missing


def episodes_from(w):
    wet = w["p_mm"] >= C.WET_HOUR_MM
    t = w.index[wet]
    if len(t) == 0:
        return pd.DataFrame(columns=["start", "end", "total_mm", "peak_mm_h", "wet_hours"])
    groups, cur = [], [t[0]]
    for a in t[1:]:
        if (a - cur[-1]) <= pd.Timedelta(hours=C.EPISODE_MERGE_GAP_H + 1):
            cur.append(a)
        else:
            groups.append(cur)
            cur = [a]
    groups.append(cur)
    rows = []
    for g in groups:
        start, end = g[0] - pd.Timedelta(hours=1), g[-1]  # obs at HH:53 covers the previous hour
        seg = w.loc[(w.index > start) & (w.index <= end), "p_mm"]
        rows.append({"start": start, "end": end, "total_mm": float(seg.sum()),
                     "peak_mm_h": float(seg.max()), "wet_hours": int((seg >= C.WET_HOUR_MM).sum()),
                     "duration_h": (end - start) / pd.Timedelta(hours=1)})
    ep = pd.DataFrame(rows)
    ep.index.name = "episode_id"
    return ep


def main():
    w, n_missing = hourly_precip(get_isd())
    ep = episodes_from(w)
    ep.to_csv(out("02_rain_episodes.csv"))

    # as-of features: an obs becomes usable at obs_time + latency
    w = w.copy()
    w["p3"] = w["p_mm"].rolling("3h", closed="right").sum()
    w["p24"] = w["p_mm"].rolling("24h", closed="right").sum()
    w["last_wet"] = pd.Series(w.index.where(w["p_mm"] >= C.WET_HOUR_MM), index=w.index).ffill()
    w["avail"] = w.index + pd.Timedelta(minutes=C.WEATHER_LATENCY_MIN)

    tidx = load_speed().index
    grid = pd.DataFrame({"t": tidx})
    f = pd.merge_asof(grid, w.reset_index()[["avail", "p_mm", "p3", "p24", "last_wet"]],
                      left_on="t", right_on="avail", direction="backward")
    feat = pd.DataFrame(index=tidx)
    feat["p1_asof"] = f["p_mm"].fillna(0).to_numpy()
    feat["p3_asof"] = f["p3"].fillna(0).to_numpy()
    feat["p24_asof"] = f["p24"].fillna(0).to_numpy()
    hsr = (f["t"] - f["last_wet"]) / pd.Timedelta(hours=1)
    feat["hours_since_rain_asof"] = hsr.fillna(999).clip(upper=999).to_numpy()

    # evaluation-only labels (actual state, NOT features)
    feat["wet_now"] = False
    feat["rain_affected"] = False
    feat["episode_id"] = -1
    for i, r in ep.iterrows():
        m = (tidx > r.start) & (tidx <= r.end)
        feat.loc[m, "wet_now"] = True
        feat.loc[m, "episode_id"] = i
        feat.loc[(tidx > r.start) & (tidx <= r.end + pd.Timedelta(hours=C.RAIN_AFFECTED_AFTER_H)), "rain_affected"] = True
    feat.to_pickle(out("02_weather_5min.pkl"))

    monthly = ep.groupby(pd.to_datetime(ep["start"]).dt.month).agg(
        episodes=("total_mm", "size"), total_mm=("total_mm", "sum")) if len(ep) else pd.DataFrame()
    rep = {"hourly_obs": len(w), "obs_without_1h_precip_group": n_missing,
           "annual_precip_mm": float(w["p_mm"].sum()), "episodes": len(ep),
           "episodes_ge_5mm": int((ep.total_mm >= 5).sum()) if len(ep) else 0,
           "episodes_ge_10mm": int((ep.total_mm >= 10).sum()) if len(ep) else 0,
           "share_5min_wet": float(feat["wet_now"].mean()),
           "episodes_by_month": monthly.to_dict(orient="index"),
           "notes": ["Single station (Sea-Tac). Distance to sensors varies; treat as area-level rain.",
                     "Hourly obs at HH:53 accumulate the previous hour; features use only obs available at forecast time."]}
    save_json(rep, "02_weather_report.json")
    log.info("Annual precip %.0f mm, %d episodes (%d >= 5 mm), wet share %.3f",
             rep["annual_precip_mm"], rep["episodes"], rep["episodes_ge_5mm"], rep["share_5min_wet"])


if __name__ == "__main__":
    main()
