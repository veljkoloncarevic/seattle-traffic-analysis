import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.common import out, load_json

thr = load_json("03_targets_report.json")["threshold_S"]
S = pd.read_pickle(out("03_network_target.pkl"))["S"]
V = ["point", "independent", "joint", "joint_weather", "lowrank_gauss", "direct"]

for h in ("15min", "30min"):
    p = pd.read_csv(out(f"05_probs_{h}.csv"), parse_dates=["t"])
    p["S0"] = S.reindex(p["t"]).to_numpy()
    for stratum, q in (("all", p), ("wet", p[p["wet"]])):
        q = q[q["S0"] <= thr]                     # not in an event at forecast time
        y = q["y"].to_numpy()
        print(f"\n{h} {stratum}: origins {len(q)}, onsets {int(y.sum())}")
        if 0 < y.sum() < len(y):
            for v in V:
                pr = q[v].to_numpy()
                print(f"  {v:14s} brier {np.mean((pr - y) ** 2):.4f}  "
                      f"auc {roc_auc_score(y, pr):.3f}  ap {average_precision_score(y, pr):.3f}")