"""Step 4: baseline forecasters (persistence, dry-reference, LightGBM) at 15/30 min.
Saves calib/test forecasts and residual matrices (origin x sensor) for step 5."""
import numpy as np
import pandas as pd

from src.common import (C, log, out, load_speed, load_speed_raw, load_json, load_adjacency,
                        row_normalised_neighbours, time_of_week)

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

WX_COLS = ["p1_asof", "p3_asof", "p24_asof", "hours_since_rain_asof"]


class Panel:
    """Holds aligned arrays and builds the long feature matrix for (origin, sensor) rows."""

    def __init__(self):
        df = load_speed()
        self.idx = df.index
        self.V = df.to_numpy()
        self.N = self.V.shape[1]
        ref = np.load(out("03_ref_slot_sensor.npy"))
        self.tow = time_of_week(self.idx)
        self.ref_t = ref[self.tow]
        self.ratio = self.V / self.ref_t
        A = load_adjacency(list(load_speed_raw().columns), list(df.columns))
        W = row_normalised_neighbours(A)
        self.nbr_ratio = self.ratio @ W.T if W is not None else np.repeat(self.ratio.mean(1, keepdims=True), self.N, 1)
        self.net_ratio = self.ratio.mean(1)
        self.wx = pd.read_pickle(out("02_weather_5min.pkl")).reindex(self.idx)
        self.tgt = pd.read_pickle(out("03_network_target.pkl"))
        self.split = self.tgt["split"].to_numpy()

    def origins(self, name, h, stride):
        maxlag = max(C.LAGS)
        t = np.arange(maxlag, len(self.idx) - h)
        t = t[(self.split[t] == name) & (self.split[t + h] == name)]
        return t[::stride]

    def features(self, t, h):
        N, n = self.N, len(t)
        cols, names = [], []
        for L in C.LAGS:
            cols.append(self.V[t - L]); names.append(f"v_lag{L}")
        cols.append(self.ratio[t]); names.append("ratio_now")
        cols.append(self.ref_t[t + h]); names.append("ref_target")
        cols.append(self.ref_t[t + h] - self.ref_t[t]); names.append("ref_change")
        cols.append(self.nbr_ratio[t]); names.append("nbr_ratio")
        cols.append(self.nbr_ratio[t] - self.nbr_ratio[t - 3]); names.append("nbr_ratio_d15")
        cols.append(np.repeat(self.net_ratio[t, None], N, 1)); names.append("net_ratio")
        tt = self.idx[t + h]
        tod = (tt.hour * 60 + tt.minute).to_numpy() / 1440.0
        for nm, val in (("tod_sin", np.sin(2 * np.pi * tod)), ("tod_cos", np.cos(2 * np.pi * tod)),
                        ("dow", tt.dayofweek.to_numpy())):
            cols.append(np.repeat(val[:, None], N, 1)); names.append(nm)
        for c in WX_COLS:
            cols.append(np.repeat(self.wx[c].to_numpy()[t, None], N, 1)); names.append(c)
        cols.append(np.repeat(np.arange(N)[None, :], n, 0)); names.append("sensor")
        X = np.stack([c.reshape(-1) for c in cols], 1).astype("float32")
        return X, names


def fit_lgbm(panel, h):
    t_tr = panel.origins("train", h, C.TRAIN_STRIDE)
    cut = int(len(t_tr) * 0.9)
    X, names = panel.features(t_tr, h)
    y = (panel.V[t_tr + h] - panel.V[t_tr]).reshape(-1)
    ok = ~np.isnan(y)
    n_fit = cut * panel.N
    fit, val = ok.copy(), ok.copy()
    fit[n_fit:] = False
    val[:n_fit] = False
    if lgb is not None:
        m = lgb.LGBMRegressor(**C.LGBM_PARAMS, random_state=C.SEED)
        m.fit(X[fit], y[fit], eval_set=[(X[val], y[val])], feature_name=names,
              categorical_feature=["sensor"], callbacks=[lgb.early_stopping(50, verbose=False)])
    else:
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, early_stopping=True, random_state=C.SEED)
        m.fit(X[fit | val], y[fit | val])
    log.info("h=%d: trained on %d rows", h, int(fit.sum()))
    return m, names


def predict(panel, m, t, h, chunk=2000):
    out_ = np.empty((len(t), panel.N), dtype="float32")
    for a in range(0, len(t), chunk):
        tt = t[a:a + chunk]
        X, _ = panel.features(tt, h)
        out_[a:a + chunk] = panel.V[tt] + m.predict(X).reshape(len(tt), panel.N)
    return out_


def main():
    panel = Panel()
    metrics = []
    for hname, h in C.HORIZONS.items():
        m, names = fit_lgbm(panel, h)
        if lgb is not None:
            imp = pd.Series(m.feature_importances_, index=names).sort_values(ascending=False)
            imp.to_csv(out(f"04_feature_importance_{hname}.csv"))
        for split in ("calib", "test"):
            t = panel.origins(split, h, C.EVAL_STRIDE)
            vtrue = panel.V[t + h]
            preds = {"lgbm": predict(panel, m, t, h), "persistence": panel.V[t],
                     "dry_reference": panel.ref_t[t + h]}
            np.savez_compressed(out(f"04_forecast_{hname}_{split}.npz"), t=t, vtrue=vtrue,
                                vhat=preds["lgbm"], vpers=preds["persistence"])
            wet = panel.wx["wet_now"].to_numpy()[t + h]
            for model, vh in preds.items():
                e = vtrue - vh
                for stratum, mask in (("all", np.ones_like(wet)), ("wet", wet), ("dry", ~wet)):
                    if mask.sum() == 0:
                        continue
                    em = e[mask]
                    metrics.append(dict(horizon=hname, split=split, model=model, stratum=stratum,
                                        n_origins=int(mask.sum()), MAE=float(np.nanmean(np.abs(em))),
                                        RMSE=float(np.sqrt(np.nanmean(em ** 2)))))
        log.info("h=%s done", hname)
    mt = pd.DataFrame(metrics)
    mt.to_csv(out("04_point_metrics.csv"), index=False)
    print(mt[mt.split == "test"].pivot_table(index=["horizon", "model"], columns="stratum", values="MAE").round(3))


if __name__ == "__main__":
    main()
