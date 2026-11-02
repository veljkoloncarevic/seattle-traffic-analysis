import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

C.OUT_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("srp")

STEPS_PER_DAY = 288
SLOTS_PER_WEEK = 7 * STEPS_PER_DAY


def out(name):
    return C.OUT_DIR / name


def save_json(obj, name):
    with open(out(name), "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(name):
    with open(out(name)) as f:
        return json.load(f)


def time_of_week(idx):
    idx = pd.DatetimeIndex(idx)
    return (idx.dayofweek * STEPS_PER_DAY + idx.hour * 12 + idx.minute // 5).to_numpy()


def load_speed_raw():
    p = C.RAW_DIR / C.SPEED_FILE
    if not p.exists():
        sys.exit(f"Missing {p}. Download the Seattle Loop data and copy '{C.SPEED_FILE}' into {C.RAW_DIR}.")
    df = pd.read_pickle(p)
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(C.LOCAL_TZ).tz_localize(None)
    elif C.TRAFFIC_INDEX_IS_UTC:
        df.index = df.index.tz_localize("UTC").tz_convert(C.LOCAL_TZ).tz_localize(None)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df.columns = [str(c) for c in df.columns]
    return df.astype("float32")


def load_speed():
    """Speed matrix restricted to sensors that passed validation."""
    df = load_speed_raw()
    keep = load_json("sensors_keep.json")
    return df[keep]


def load_adjacency(columns_all, keep):
    p = C.RAW_DIR / C.ADJ_FILE
    if not p.exists():
        log.warning("No adjacency file; neighbour features/diagnostics disabled.")
        return None
    A = np.load(p)
    pos = {c: i for i, c in enumerate(columns_all)}
    ix = [pos[c] for c in keep]
    return A[np.ix_(ix, ix)].astype("float32")


def row_normalised_neighbours(A):
    if A is None:
        return None
    W = (A > 0).astype("float32")
    np.fill_diagonal(W, 0)
    W = W + W.T
    W = (W > 0).astype("float32")
    deg = W.sum(1, keepdims=True)
    deg[deg == 0] = 1
    return W / deg
