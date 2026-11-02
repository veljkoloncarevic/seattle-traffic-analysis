"""Step 5: do cross-sensor forecast-error dependencies matter for network-level warnings?

Variants for P(S(t+h) > threshold):
  point          deterministic 0/1 from the point forecast (no uncertainty)
  independent    per-sensor residuals drawn independently (dependence destroyed = 'shuffled')
  joint          whole calibration residual vectors (dependence preserved)
  joint_weather  whole vectors from the pool matching the rain state known at origin
  lowrank_gauss  Gaussian with Ledoit-Wolf shrinkage covariance of calibration residuals
  direct         classifier on the network target, trained on TRAIN only (raw)
  direct_cal     same classifier, Platt-recalibrated on the CALIB split (fair comparison:
                 the residual-based variants also take their spread from CALIB)

v2 additions
  - target uses the time-varying reference 03_ref_t.npy (drift fix in step 3)
  - strata: all / wet / dry / onset / onset_wet  (onset = S(t) <= threshold at the origin)
  - distinct event counts per stratum (consecutive windows of one event are not independent)
  - day-block bootstrap 95% CIs for Brier differences vs 'joint' AND vs 'direct_cal'
"""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.common import C, log, out, load_json, load_speed, load_adjacency, load_speed_raw, time_of_week

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

VARIANTS = ["point", "independent", "joint", "joint_weather", "lowrank_gauss", "direct", "direct_cal"]
REFERENCES = ["joint", "direct_cal"]
BOOT_STRATA = ["all", "wet", "onset", "onset_wet"]


def network_S(v, ref):
    return np.clip(1.0 - v / ref, 0, None).mean(-1)


def ece(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def scores(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    r = dict(n=len(y), positives=int(y.sum()), event_rate=float(y.mean()), brier=float(np.mean((p - y) ** 2)),
             logloss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), ece=ece(p, y))
    if 0 < y.sum() < len(y):
        r["auc"] = float(roc_auc_score(y, p))
        r["avg_precision"] = float(average_precision_score(y, p))
    return r


def residual_diagnostics(E, wet, A):
    rows = []
    for name, m in (("all", np.ones(len(E), bool)), ("wet", wet), ("dry", ~wet)):
        if m.sum() < 30:
            continue
        X = E[m]
        X = X[~np.isnan(X).any(1)]
        Cm = np.corrcoef(X, rowvar=False)
        off = ~np.eye(len(Cm), dtype=bool)
        r = dict(stratum=name, n=len(X), mean_offdiag_corr=float(np.nanmean(Cm[off])))
        if A is not None:
            adj = (A > 0) & off
            r["mean_corr_adjacent"] = float(np.nanmean(Cm[adj]))
            r["mean_corr_non_adjacent"] = float(np.nanmean(Cm[off & ~adj]))
        ev = np.linalg.eigvalsh(np.cov(X, rowvar=False))[::-1]
        r["pc1_variance_share"] = float(ev[0] / ev.sum())
        r["pc1_5_variance_share"] = float(ev[:5].sum() / ev.sum())
        r["variance_inflation_sum"] = float(np.var(X.sum(1)) / np.var(X, 0).sum())
        rows.append(r)
    return rows


def sample_probs(vhat, ref, thr, sampler, K, rng):
    p = np.empty(len(vhat))
    for a in range(0, len(vhat), C.MC_CHUNK):
        vh, rf = vhat[a:a + C.MC_CHUNK], ref[a:a + C.MC_CHUNK]
        e = sampler(len(vh), K, rng, a)                        # (B, K, N)
        S = network_S(vh[:, None, :] + e, rf[:, None, :])     # (B, K)
        p[a:a + C.MC_CHUNK] = (S > thr).mean(1)
    return p


def day_block_bootstrap(diff, days, reps, rng):
    ud = np.unique(days)
    per_day = pd.Series(diff).groupby(days).agg(["sum", "count"]).loc[ud]
    s, c = per_day["sum"].to_numpy(), per_day["count"].to_numpy()
    draws = rng.integers(0, len(ud), size=(reps, len(ud)))
    boot = s[draws].sum(1) / c[draws].sum(1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)), len(ud)


def direct_model(h, S, thr, split, wxf, idx):
    """Network-level classifier: features at origin -> 1{S(t+h) > thr}. Trained on TRAIN only."""
    tod = (idx.hour * 60 + idx.minute).to_numpy() / 1440
    lagged = [np.r_[np.full(L, S[0]), S[:-L]] for L in (1, 3, 6, 12)]
    F = np.column_stack([S] + lagged +
                        [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod), idx.dayofweek.to_numpy(),
                         wxf[["p1_asof", "p3_asof", "p24_asof", "hours_since_rain_asof"]].to_numpy()])
    y = np.r_[S[h:] > thr, np.zeros(h, bool)]
    tr = np.arange(12, len(S) - h)
    tr = tr[(split[tr] == "train") & (split[tr + h] == "train")]
    if lgb is not None:
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=100,
                               verbose=-1, random_state=C.SEED)
    else:
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=C.SEED)
    m.fit(F[tr], y[tr])
    return lambda t: m.predict_proba(F[t])[:, 1]


def platt(p_cal, y_cal):
    """Logistic recalibration on the logit of the raw probability; identity if CALIB has one class."""
    if y_cal.min() == y_cal.max():
        log.warning("CALIB has a single class; direct_cal = direct")
        return lambda p: p
    z = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))[:, None]  # noqa: E731
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z(p_cal), y_cal)
    return lambda p: lr.predict_proba(z(p))[:, 1]


def main():
    rng = np.random.default_rng(C.SEED)
    df = load_speed()
    idx = df.index
    ref_path = out("03_ref_t.npy")
    if ref_path.exists():
        ref_all = np.load(ref_path, mmap_mode="r")
    else:
        log.warning("03_ref_t.npy missing (old step 3?) -> using static slot reference")
        ref_all = np.load(out("03_ref_slot_sensor.npy"))[time_of_week(idx)]
    tgt = pd.read_pickle(out("03_network_target.pkl"))
    S_all = tgt["S"].to_numpy()
    split = tgt["split"].to_numpy()
    wet_all = tgt["wet_now"].to_numpy().astype(bool)
    event_all = tgt["event_id"].to_numpy() if "event_id" in tgt else np.full(len(tgt), -1)
    thr = load_json("03_targets_report.json")["threshold_S"]
    A = load_adjacency(list(load_speed_raw().columns), list(df.columns))
    wxf = pd.read_pickle(out("02_weather_5min.pkl")).reindex(idx)
    rain_state_all = (wxf["p3_asof"].to_numpy() > 0)        # known at origin

    diag_rows, metric_rows, boot_rows, count_rows = [], [], [], []
    for hname, h in C.HORIZONS.items():
        cal = np.load(out(f"04_forecast_{hname}_calib.npz"))
        tst = np.load(out(f"04_forecast_{hname}_test.npz"))
        E = (cal["vtrue"] - cal["vhat"]).astype("float32")
        E = np.where(np.isnan(E), 0, E)
        for r in residual_diagnostics(E, wet_all[cal["t"] + h], A):
            diag_rows.append(dict(horizon=hname, **r))

        t = tst["t"]
        vhat = tst["vhat"]
        ref = np.asarray(ref_all[t + h])
        y = (S_all[t + h] > thr).astype(float)
        n_cal, N = E.shape
        cal_state = rain_state_all[cal["t"]]
        tst_state = rain_state_all[t]
        lw = LedoitWolf().fit(E)
        L = np.linalg.cholesky(lw.covariance_ + 1e-6 * np.eye(N)).astype("float32")
        mu = lw.location_.astype("float32")

        def s_indep(B, K, g, a):
            return E[g.integers(0, n_cal, size=(B, K, N)), np.arange(N)]

        def s_joint(B, K, g, a):
            return E[g.integers(0, n_cal, size=(B, K))]

        pools = {s: np.flatnonzero(cal_state == s) for s in (False, True)}

        def s_joint_wx(B, K, g, a):
            o = np.empty((B, K, N), dtype="float32")
            for b in range(B):
                pool = pools[bool(tst_state[a + b])]
                pool = pool if len(pool) >= C.MIN_WEATHER_POOL else np.arange(n_cal)
                o[b] = E[pool[g.integers(0, len(pool), size=K)]]
            return o

        def s_gauss(B, K, g, a):
            z = g.standard_normal((B, K, N), dtype="float32")
            return mu + z @ L.T

        probs = {"point": (network_S(vhat, ref) > thr).astype(float)}
        for name, smp in (("independent", s_indep), ("joint", s_joint),
                          ("joint_weather", s_joint_wx), ("lowrank_gauss", s_gauss)):
            probs[name] = sample_probs(vhat, ref, thr, smp, C.MC_SAMPLES, rng)
            log.info("h=%s %s done", hname, name)

        direct = direct_model(h, S_all, thr, split, wxf, idx)
        probs["direct"] = direct(t)
        t_cal = cal["t"]
        recal = platt(direct(t_cal), (S_all[t_cal + h] > thr).astype(int))
        probs["direct_cal"] = recal(probs["direct"])

        wet = wet_all[t + h]
        onset = S_all[t] <= thr
        strata = {"all": np.ones(len(y), bool), "wet": wet, "dry": ~wet,
                  "onset": onset, "onset_wet": onset & wet}
        ev = event_all[t + h]
        for sname, m in strata.items():
            pos = m & (y == 1)
            count_rows.append(dict(horizon=hname, stratum=sname, origins=int(m.sum()), positive_windows=int(pos.sum()),
                                   distinct_events=int(len(np.unique(ev[pos & (ev >= 0)]))),
                                   days=int(len(np.unique(pd.DatetimeIndex(idx[t[m]]).normalize())))))
            if m.sum() == 0:
                continue
            for name in VARIANTS:
                metric_rows.append(dict(horizon=hname, variant=name, stratum=sname, **scores(probs[name][m], y[m])))

        days = pd.DatetimeIndex(idx[t]).normalize().to_numpy()
        for refname in REFERENCES:
            b_ref = (probs[refname] - y) ** 2
            for other in VARIANTS:
                if other in (refname, "point"):
                    continue
                for sname in BOOT_STRATA:
                    m = strata[sname]
                    if m.sum() == 0 or y[m].sum() == 0:
                        continue
                    d = ((probs[other] - y) ** 2 - b_ref)[m]
                    mean, lo, hi, nd = day_block_bootstrap(d, days[m], C.BOOTSTRAP_REPS, rng)
                    boot_rows.append(dict(horizon=hname, reference=refname, variant=other, stratum=sname,
                                          brier_diff=mean, ci95_low=lo, ci95_high=hi, days=nd,
                                          verdict=("worse" if lo > 0 else "better" if hi < 0 else "n.s.")))
        pd.DataFrame({"t": idx[t], "y": y, "wet": wet, "onset": onset, "S0": S_all[t], "event_id": ev,
                      **probs}).to_csv(out(f"05_probs_{hname}.csv"), index=False)

    pd.DataFrame(diag_rows).to_csv(out("05_residual_dependence.csv"), index=False)
    mt = pd.DataFrame(metric_rows)
    mt.to_csv(out("05_warning_metrics.csv"), index=False)
    bt = pd.DataFrame(boot_rows)
    bt.to_csv(out("05_bootstrap_brier.csv"), index=False)
    ct = pd.DataFrame(count_rows)
    ct.to_csv(out("05_event_counts.csv"), index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 500)
    print("\nResidual dependence (calibration residuals):\n", pd.DataFrame(diag_rows).round(3).to_string(index=False))
    print("\nTest sample sizes (distinct_events = separate exceedance events, the real sample size):\n",
          ct.to_string(index=False))
    show = mt[mt.stratum.isin(["all", "wet", "onset", "onset_wet"])]
    cols = ["horizon", "stratum", "variant", "positives", "brier", "ece", "auc", "avg_precision"]
    print("\nWarning skill (test):\n", show[cols].round(4).to_string(index=False))
    print("\nBrier difference = variant - reference (positive = variant worse); day-block bootstrap 95% CI:\n",
          bt.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
