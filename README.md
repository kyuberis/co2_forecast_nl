# CO2 Emission Factor Forecasting (Netherlands)

7-day hourly forecasts of the Dutch grid CO2 emission factor (kg CO2 / kWh) using a Temporal Fusion Transformer with weather forecasts as known-future covariates. NHiTS and seasonal naive (t-24) are included as baselines.

The CO2 emission factor measures the carbon intensity of grid electricity at a given hour. It rises when the grid leans on gas and coal, falls when wind and solar generation are high. A reliable 168h forecast helps shift flexible loads (EV charging, heat pumps, industrial processes) toward cleaner hours.

## Why TFT

- **Known-future inputs**: weather forecast and time features are deterministic for the horizon. TFT uses them in the decoder; an LSTM cannot do this natively.
- **Quantile output**: predicts q10, q50, q90, giving an 80% interval per hour. Useful for downstream load-shifting decisions.
- **Strong baseline included**: NHiTS does not use future covariates, so the gap between TFT and NHiTS is roughly the value added by the weather forecast.

## Results

Test set: 2024-10-01 to 2025-12-31 (~15 months unseen).

| Model                  | MAE (kg/kWh) | RMSE (kg/kWh) | MAPE (%) |
|------------------------|--------------|---------------|----------|
| Seasonal naive (t-24)  | to be updated after full training run   |
| NHiTS                  | to be updated after full training run   |
| TFT                    | to be updated after full training run   |

Per-horizon MAE (TFT):

| Horizon  |                 MAE                   |
|----------|---------------------------------------|
| 1-24h    | to be updated after full training run |
| 25-72h   | to be updated after full training run |
| 73-168h  | to be updated after full training run |

Plots saved to `artifacts/predictions/` after evaluation.

## Project structure

```
co2-forecast-nl/
├── config.yaml         # all paths and hyperparameters
├── requirements.txt
├── src/
│   ├── data.py         # load_and_prepare, make_splits, build_datasets, add_time_features
│   ├── models.py       # build_tft, build_nhints (baseline)
│   ├── train.py        # CLI: train one model
│   ├── evaluate.py     # CLI: metrics, naive baseline, plots
│   └── forecast.py     # CLI: 168h forecast with weather input
├── tests/
│   └── test_basic.py
└── data/               # CSVs go here (gitignored)
```

## Quick start

```bash
git clone https://github.com/<user>/co2-forecast-nl.git
cd co2-forecast-nl

python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place `master_dataset.csv` and `openmeteo_forecast_7days.csv` in `data/`. Then:

```bash
pytest -v                                            # sanity tests
python -m src.train --config config.yaml --model tft
python -m src.train --config config.yaml --model nhits
python -m src.evaluate --config config.yaml
python -m src.forecast --config config.yaml
```

For a CPU smoke-test before a real GPU run, edit `config.yaml`: set `max_epochs: 2`, `batch_size: 16`, `tft_hidden_size: 16`, `accelerator: cpu`. Once the pipeline runs end-to-end, use proper parameters and train on GPU.

## Data

- **Target**: `co2_emissionfactor`, hourly, Dutch grid.
- **Past covariates**: per-source generation and capacity (solar, wind, offwind, biomass, waste, gas, coal, nuclear) plus missingness flags from upstream imputation.
- **Future covariates**: cyclical time features (hour, day of week, day of year, month), `is_daylight`, and Open-Meteo weather forecast variables.

## License

MIT.
