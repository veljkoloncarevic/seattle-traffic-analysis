"""Run the full pipeline: python run_all.py [--from N]"""
import argparse
import runpy
from pathlib import Path

STEPS = ["01_validate_traffic.py", "02_weather.py", "03_targets.py", "04_forecast.py", "05_dependence.py"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=5)
    a = ap.parse_args()
    root = Path(__file__).resolve().parent
    for s in STEPS[a.start - 1:a.end]:
        print(f"\n===== {s} =====")
        runpy.run_path(str(root / s), run_name="__main__")
