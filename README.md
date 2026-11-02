# Seattle loop + NOAA rain: network-degradation warning pilot

Steps 1-5 of the pilot plan: validate the data, add weather, define the
target, train baseline forecasters, and test whether cross-sensor forecast-error
dependence matters for network-level warnings.

## 1. Get the data → `data/raw/`

| File | Source |
|---|---|
| `speed_matrix_2015` | "Data Download Link" in https://github.com/zhiyongc/Seattle-Loop-Data |
| `Loop_Seattle_2015_A.npy` | same link |
| `nodes_loop_mp_list.csv` | same link (optional) |
| `72793024233.csv` | auto-downloaded by step 2 from NOAA Global-Hourly (Sea-Tac, 2015): https://www.ncei.noaa.gov/data/global-hourly/access/2015/72793024233.csv |

The Zenodo copy (record 3258904) only contains `speed_matrix_2015_1mph` with no
adjacency matrix. If you use it, set `SPEED_FILE` in `config.py`.

## 2. Run

```bash
pip install -r requirements.txt
python run_all.py            # all steps
python run_all.py --from 3   # resume from step 3
```

Everything is written to `outputs/`. Edit settings in `config.py` only.

Smoke test without real data:
```bash
python tests/make_synthetic.py
SRP_RAW_DIR=tests/synth SRP_OUT_DIR=tests/synth_out python run_all.py
```
(On Windows PowerShell use `$env:SRP_RAW_DIR="tests/synth"; $env:SRP_OUT_DIR="tests/synth_out"; python run_all.py`.)

## 3. What each step does

| Step | Output | Content |
|---|---|---|
| `01_validate_traffic.py` | `01_validation_report.json`, `sensor_quality.csv`, `sensors_keep.json` | Timestamp regularity and gaps, local-vs-UTC heuristic, NaN/zero/out-of-range values, flatline runs per sensor (proxy for imputed or stuck data), adjacency checks |
| `02_weather.py` | `02_rain_episodes.csv`, `02_weather_5min.pkl`, `02_weather_report.json` | Hourly precipitation from ISD `FM-15` / `AA1`, UTC→local, rain episodes, **as-of** features (an observation is used only after `obs_time + WEATHER_LATENCY_MIN`) |
| `03_targets.py` | `03_targets_report.json`, `03_network_target.pkl`, `03_ref_slot_sensor.npy`, `03_gao_*.csv` | Chronological splits with embargo, pushed out of rain episodes; dry-weather reference speed per sensor × 5-min time-of-week slot (**train + dry only**); shortfall `d_i = max(0, 1 − v_i/ref_i)`; network shortfall `S = mean_i d_i`; threshold = 95th percentile of S on train; Gao et al. (2024) link-reliability resilience per episode |
| `04_forecast.py` | `04_point_metrics.csv`, `04_forecast_*.npz`, `04_feature_importance_*.csv` | Persistence, dry-reference and LightGBM (one model across all sensors) at 15 and 30 min; MAE/RMSE split by wet and dry |
| `05_dependence.py` | `05_residual_dependence.csv`, `05_warning_metrics.csv`, `05_bootstrap_brier_vs_joint.csv`, `05_probs_*.csv` | Residual correlations (adjacent vs. non-adjacent, wet vs. dry), PC1 share, variance inflation of the network sum; warning skill for point / independent (shuffled) / joint / joint-by-weather / low-rank Gaussian / direct classifier; day-block bootstrap CIs |

## 4. How to read step 5

- `variance_inflation_sum` measures how much an independence assumption understates
  the uncertainty of the network total. A value of 1 means no dependence.
- **independent vs joint** is the sanity check, and joint should win.
- **The real question is joint vs lowrank_gauss vs direct.** If the simple joint
  baselines already match each other and the direct classifier, there is little room
  for a new learned dependence model. Look especially at the wet stratum.
- Brier CIs use whole days as bootstrap blocks. Never use per-window counts.

## 5. Known limitations

- **The speed matrix is already imputed** (stated on the Zenodo record), so
  real outages are not marked. `flatline_frac` is only a proxy. For the
  "incomplete observations" angle you need raw WSDOT data or simulated masking.
- Weather comes from one station (Sea-Tac) at hourly resolution. Treat it as
  area-level rain.
- Train covers Jan–Aug and test covers Nov–Dec, so there is a seasonal shift. The test
  event rate will differ from 5%. Change `SPLITS` if needed.
- `GAO_RHO` (0.75) is a placeholder. In the paper ρ is a policy choice, so agree it
  with Prof. Gao. Eqs. 3–5 are implemented as described in the paper text (see the
  docstring in `03_targets.py`). This is an approximation, not a reproduction.
- Holidays are not removed from the dry reference.
