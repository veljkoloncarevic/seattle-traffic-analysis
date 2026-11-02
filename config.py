"""Central configuration. Edit values here; every script imports this file."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_DIR = Path(os.environ.get("SRP_RAW_DIR", ROOT / "data" / "raw"))
OUT_DIR = Path(os.environ.get("SRP_OUT_DIR", ROOT / "outputs"))

# ---- Input files (copy into data/raw/) ---------------------------------
SPEED_FILE = "speed_matrix_2015"            # pickled pandas DataFrame (time x milepost)
ADJ_FILE = "Loop_Seattle_2015_A.npy"        # adjacency matrix, same column order
NODES_FILE = "nodes_loop_mp_list.csv"       # milepost list, same order (optional)
WEATHER_FILE = "72793024233.csv"            # NOAA ISD / Global-Hourly, Sea-Tac 2015
ISD_URL = "https://www.ncei.noaa.gov/data/global-hourly/access/2015/72793024233.csv"
WEATHER_AUTO_DOWNLOAD = True

# ---- Time handling -----------------------------------------------------
# NOAA timestamps are UTC. The loop matrix index is naive; 01_validate prints a
# heuristic verdict on whether it is local or UTC. Set this accordingly.
TRAFFIC_INDEX_IS_UTC = False
LOCAL_TZ = "America/Los_Angeles"
WEATHER_LATENCY_MIN = 10        # an hourly obs is usable this many minutes after its timestamp

# ---- Rain episodes -----------------------------------------------------
WET_HOUR_MM = 0.25              # hour counts as wet if 1-h precip >= this (0.01 in = 0.254 mm)
EPISODE_MERGE_GAP_H = 2         # merge wet hours separated by <= this many dry hours
RAIN_AFFECTED_AFTER_H = 3       # hours after an episode still treated as "not dry"

# ---- Chronological splits (inclusive dates) ----------------------------
SPLITS = {
    "train": ("2015-01-01", "2015-08-31"),
    "calib": ("2015-09-01", "2015-10-31"),
    "test":  ("2015-11-01", "2015-12-31"),
}
EMBARGO_HOURS = 24              # dropped at the start of calib/test; boundaries also pushed out of rain episodes

# ---- Target definition -------------------------------------------------
MIN_REF_COUNT = 5               # min dry samples per (sensor, time-of-week slot), else all-weather fallback
SHORTFALL_THRESHOLD_Q = 0.95    # warning threshold = this quantile of network shortfall S on TRAIN

# ---- Data quality ------------------------------------------------------
FLATLINE_STEPS = 12             # identical value for >= 12 steps (1 h) = suspected stuck/imputed
MAX_FLATLINE_FRAC = 0.20        # sensors above this flatline fraction are excluded

# ---- Forecasting -------------------------------------------------------
HORIZONS = {"15min": 3, "30min": 6, "60min": 12}   # steps of 5 min
LAGS = [0, 1, 2, 3, 6, 12]
TRAIN_STRIDE = 6                # use every 6th origin for training (memory)
EVAL_STRIDE = 3                 # evaluate every 3rd origin in calib/test
LGBM_PARAMS = dict(n_estimators=600, learning_rate=0.05, num_leaves=63,
                   min_child_samples=200, subsample=0.8, subsample_freq=1,
                   colsample_bytree=0.8, verbose=-1)

# ---- Dependence test ---------------------------------------------------
MC_SAMPLES = 200
MC_CHUNK = 100
BOOTSTRAP_REPS = 1000
MIN_WEATHER_POOL = 50
SEED = 42

# ---- Gao et al. (2024, TRD 126:104000) link-reliability resilience -----
# Eq.(1): F=1 if V/V_free >= rho. rho is a policy choice in the paper -> agree a value with Prof. Gao.
GAO_RHO = 0.75
GAO_VFREE_QUANTILE = 0.85       # V_free = this quantile of dry night-time (00-05h) train speeds
GAO_RECOVERY_TOL = 0.10         # recovered when V >= (1-tol)*dry reference for 3 consecutive steps
GAO_RECOVERY_MAX_H = 3
