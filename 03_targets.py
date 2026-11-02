"""Step 3: leakage-safe splits, dry-weather reference speed, per-sensor shortfall,
network shortfall S(t), warning threshold, distinct events, and Gao et al. (2024) resilience.

v2 drift fix
------------
ref_i(t) = profile_i[time-of-week(t)] * level_i(day, period-of-day)
  profile : static dry-TRAIN median per (sensor, 5-min time-of-week slot), as in v1
  level   : trailing median of daily dry medians of V/profile over the previous
            LEVEL_WINDOW_DAYS days, per sensor and period of day, shifted by one day.
            Only past data enter, so it is available at forecast time.
The level absorbs slow seasonal drift (summer vs winter), so the threshold on S means the
same thing in train, calib and test. Set REF_MODE = "static" in config.py to reproduce v1.

Outputs
-------
03_ref_slot_sensor.npy   static profile (used as features by step 4; same as v1)
03_ref_t.npy             time-varying reference, T x N float32 (used for the target in step 5)
03_network_target.pkl    S, split, wet_now, rain_affected, event_id
03_targets_report.json   threshold, event rates by split/month, distinct event counts
"""
import numpy as np
import pandas as pd

from src.common import C, log, out, save_json, load_speed, time_of_week, SLOTS_PER_WEEK

REF_MODE = getattr(C, "REF_MODE", "rolling")              # "rolling" or "static"
LEVEL_WINDOW_DAYS = getattr(C, "LEVEL_WINDOW_DAYS", 14)
LEVEL_MIN_DAYS = getattr(C, "LEVEL_MIN_DAYS", 3)
LEVEL_CLIP = getattr(C, "LEVEL_CLIP", (0.70, 1.15))
DAY_PERIODS = getattr(C, "DAY_PERIODS", [(0, 6), (6, 10), (10, 15), (15, 20), (20, 24)])
EVENT_MERGE_GAP_STEPS = getattr(C, "EVENT_MERGE_GAP_STEPS", 6)   # exceedances <= 30 min apart = one event


def make_splits(idx, wx):
    """Chronological splits; calib/test start after an embargo and outside rain-affected periods."""
    ep = pd.read_csv(out("02_rain_episodes.csv"), parse_dates=["start", "end"])
    lab = pd.Series("none", index=idx)
    bounds = {}
    for name, (a, b) in C.SPLITS.items():
        start = pd.Timestamp(a)
        end = pd.Timestamp(b) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        if name != "train":
            start = start + pd.Timedelta(hours=C.EMBARGO_HOURS)
            hit = ep[(ep.start < start) & (ep.end + pd.Timedelta(hours=C.RAIN_AFFECTED_AFTER_H) >= start)]
            if len(hit):
                start = hit.end.max() + pd.Timedelta(hours=C.RAIN_AFFECTED_AFTER_H, minutes=5)
        lab[(idx >= start) & (idx <= end)] = name
        bounds[name] = [str(start), str(end)]
    return lab, bounds


def dry_reference(V, tow, mask_ref, mask_fallback):
    """Median speed per (time-of-week slot, sensor) from dry TRAIN data; fallback to all-weather TRAIN."""
    N = V.shape[1]
    ref = np.full((SLOTS_PER_WEEK, N), np.nan, dtype="float32")
    cnt = np.zeros(SLOTS_PER_WEEK, dtype=int)
    order = np.argsort(tow, kind="stable")
    bounds = np.searchsorted(tow[order], np.arange(SLOTS_PER_WEEK + 1))
    for s in range(SLOTS_PER_WEEK):
        rows = order[bounds[s]:bounds[s + 1]]
        r_ref = rows[mask_ref[rows]]
        cnt[s] = len(r_ref)
        use = r_ref if cnt[s] >= C.MIN_REF_COUNT else rows[mask_fallback[rows]]
        if len(use):
            ref[s] = np.nanmedian(V[use], axis=0)
    return ref, cnt


def rolling_level(V, prof_t, idx, dry):
    """Per sensor and period of day: trailing median of daily dry medians of V/profile (past days only)."""
    T, N = V.shape
    ratio = (V / prof_t).astype("float32")
    ratio[~dry] = np.nan
    day = idx.normalize()
    days = pd.DatetimeIndex(day.unique())
    day_pos = days.get_indexer(day)
    hour = idx.hour.to_numpy()
    level_t = np.ones((T, N), dtype="float32")
    for lo, hi in DAY_PERIODS:
        m = (hour >= lo) & (hour < hi)
        daily = pd.DataFrame(ratio[m]).groupby(day_pos[m]).median().reindex(range(len(days)))
        lev = (daily.rolling(LEVEL_WINDOW_DAYS, min_periods=LEVEL_MIN_DAYS).median()
               .shift(1).ffill().fillna(1.0).clip(*LEVEL_CLIP).to_numpy(dtype="float32"))
        level_t[m] = lev[day_pos[m]]
    return level_t


def event_ids(S, thr, merge_gap):
    """Runs of S > thr (gaps <= merge_gap steps merged) -> event id per step, -1 outside events."""
    ids = np.full(len(S), -1, dtype=int)
    eid, last = -1, -10 ** 9
    for i in np.flatnonzero(S > thr):
        if i - last > merge_gap + 1:
            eid += 1
        ids[i] = eid
        last = i
    return ids


def gao_resilience(df, ref_t, wx, ep, vfree, split):
    """Approximation of Gao, Hu & Wang (2024) Eqs. (1)-(5) per rain episode and sensor.

    F(t)  = 1 if V/V_free >= rho                                   (Eq. 1)
    r_hat = mean F over the rain window                             (Eq. 2)
    r0    = mean F over the same weekly window in up to 4 previous dry weeks (initial reliability)
    C     = 1 if T_rec <= mean T_rec else mean T_rec / T_rec        (Eqs. 3-4, as described in the text)
    R     = (n_rain * r_hat + n_rec * C) / ((n_rain + n_rec) * r0) (Eq. 5, convex combination / initial)
    """
    V = df.to_numpy()
    idx = df.index
    F = (V / vfree[None, :] >= C.GAO_RHO).astype("float32")
    affected = wx["rain_affected"].to_numpy()
    rows = []
    max_rec = C.GAO_RECOVERY_MAX_H * 12
    for eid, e in ep.iterrows():
        a = idx.searchsorted(e.start, side="right")
        b = idx.searchsorted(e.end, side="right")
        if b - a < 3 or b + max_rec >= len(idx):
            continue
        r_hat = F[a:b].mean(0)
        prev = []
        for k in range(1, 5):
            sa, sb = a - k * 2016, b - k * 2016
            if sa >= 0 and not affected[sa:sb].any():
                prev.append(F[sa:sb].mean(0))
        if not prev:
            continue
        r0 = np.clip(np.mean(prev, 0), 1e-3, None)
        ok = V[b:b + max_rec] >= (1 - C.GAO_RECOVERY_TOL) * ref_t[b:b + max_rec]
        run3 = ok[:-2] & ok[1:-1] & ok[2:]
        T = np.where(run3.any(0), run3.argmax(0), max_rec).astype(float) + 1  # steps
        rows.append(dict(episode_id=eid, split=split.iloc[a], start=e.start, end=e.end,
                         total_mm=e.total_mm, peak_mm_h=e.peak_mm_h, n_rain=b - a,
                         r_hat=r_hat, r0=r0, T_rec=T))
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    Tbar = np.mean(np.vstack([r["T_rec"] for r in rows]), 0)
    long, sysrows = [], []
    for r in rows:
        Crec = np.where(r["T_rec"] <= Tbar, 1.0, Tbar / r["T_rec"])
        n_rec = r["T_rec"]
        R = (r["n_rain"] * r["r_hat"] + n_rec * Crec) / ((r["n_rain"] + n_rec) * r["r0"])
        for j, s in enumerate(df.columns):
            long.append(dict(episode_id=r["episode_id"], sensor=s, r_hat=r["r_hat"][j], r0=r["r0"][j],
                             T_rec_min=5 * r["T_rec"][j], C_rec=Crec[j], R=R[j]))
        sysrows.append(dict(episode_id=r["episode_id"], split=r["split"], start=r["start"], end=r["end"],
                            total_mm=r["total_mm"], peak_mm_h=r["peak_mm_h"], R_mean=float(R.mean()),
                            R_p10=float(np.percentile(R, 10)), share_R_below_0_8=float((R < 0.8).mean())))
    return pd.DataFrame(long), pd.DataFrame(sysrows)


def main():
    df = load_speed()
    wx = pd.read_pickle(out("02_weather_5min.pkl")).reindex(df.index)
    idx = df.index
    V = df.to_numpy()
    tow = time_of_week(idx)
    split, bounds = make_splits(idx, wx)
    sp = split.to_numpy()
    dry = ~wx["rain_affected"].to_numpy().astype(bool)
    wet_now = wx["wet_now"].to_numpy().astype(bool)
    is_train = sp == "train"

    prof, cnt = dry_reference(V, tow, is_train & dry, is_train)
    prof_t = prof[tow]
    if REF_MODE == "rolling":
        level_t = rolling_level(V, prof_t, idx, dry)
        ref_t = (prof_t * level_t).astype("float32")
    else:
        level_t = None
        ref_t = prof_t.astype("float32")

    short = np.clip(1.0 - V / ref_t, 0, None)        # per-sensor relative shortfall d_i(t)
    S = short.mean(1)                                 # network shortfall S(t)
    thr = float(np.quantile(S[is_train], C.SHORTFALL_THRESHOLD_Q))
    eids = event_ids(S, thr, EVENT_MERGE_GAP_STEPS)

    np.save(out("03_ref_slot_sensor.npy"), prof)
    np.save(out("03_ref_t.npy"), ref_t)
    pd.DataFrame({"S": S, "split": sp, "wet_now": wet_now, "rain_affected": ~dry, "event_id": eids},
                 index=idx).to_pickle(out("03_network_target.pkl"))

    # Gao-style link efficiency & resilience
    night = (idx.hour < 5) & is_train & dry
    vfree = np.nanquantile(V[night], C.GAO_VFREE_QUANTILE, axis=0)
    ep = pd.read_csv(out("02_rain_episodes.csv"), index_col=0, parse_dates=["start", "end"])
    link, system = gao_resilience(df, ref_t, wx, ep, vfree, split)
    link.to_csv(out("03_gao_link_resilience.csv"), index=False)
    system.to_csv(out("03_gao_system_resilience.csv"), index=False)

    exceed = S > thr
    month = pd.Series(exceed).groupby(idx.month).mean()
    ev = pd.DataFrame({"eid": eids, "split": sp, "wet": wet_now})[eids >= 0].groupby("eid").first()
    rep = {"ref_mode": REF_MODE,
           "split_bounds": bounds,
           "split_rows": split.value_counts().to_dict(),
           "slots_with_dry_fallback": int((cnt < C.MIN_REF_COUNT).sum()),
           "threshold_S": thr, "threshold_quantile": C.SHORTFALL_THRESHOLD_Q,
           "event_rate": {k: float(exceed[sp == k].mean()) for k in ("train", "calib", "test")},
           "event_rate_dry": {k: float(exceed[(sp == k) & dry].mean()) for k in ("train", "calib", "test")},
           "event_rate_wet": {k: float(exceed[(sp == k) & wet_now].mean()) for k in ("train", "calib", "test")},
           "event_rate_by_month": {int(k): round(float(v), 4) for k, v in month.items()},
           "distinct_events": ev["split"].value_counts().to_dict(),
           "distinct_events_starting_wet": ev[ev["wet"]]["split"].value_counts().to_dict(),
           "gao_rho": C.GAO_RHO, "gao_episodes_scored": int(len(system))}
    if level_t is not None:
        lv = pd.Series(level_t.mean(1)).groupby(idx.month).mean()
        rep["mean_level_by_month"] = {int(k): round(float(v), 4) for k, v in lv.items()}
    if len(system):
        rep["gao_corr_R_vs_total_mm"] = float(system[["R_mean", "total_mm"]].corr().iloc[0, 1])
    save_json(rep, "03_targets_report.json")

    r4 = lambda d: {k: round(v, 4) for k, v in d.items()}  # noqa: E731
    log.info("Ref mode: %s; threshold S = %.4f", REF_MODE, thr)
    log.info("Event rate     train/calib/test: %s", r4(rep["event_rate"]))
    log.info("  dry only                     : %s", r4(rep["event_rate_dry"]))
    log.info("  wet only                     : %s", r4(rep["event_rate_wet"]))
    log.info("Event rate by month: %s", rep["event_rate_by_month"])
    log.info("Distinct events: %s (starting wet: %s)", rep["distinct_events"], rep["distinct_events_starting_wet"])


if __name__ == "__main__":
    main()
