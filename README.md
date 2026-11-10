# Epoch

**An experimental study of temporal data shift and model adaptation.**

How much does a machine learning model's performance change as new data
arrives, and can lightweight updates help? Epoch compares four training
policies on the historical Electricity benchmark.

## Research question

**Can periodic retraining, mini-batch online learning, or error-based drift
adaptation outperform a model that is never updated?**

The methods share one model architecture, the same initial data and the same
held-forward evaluation sequence. This is an exploratory benchmark, not a new
learning algorithm or a production forecasting system.

## Methods

- **Static:** train once and keep the model unchanged.
- **Periodic:** refit on recent labeled data at a fixed interval.
- **Online:** update incrementally after each evaluated batch.
- **ADWIN:** monitor errors and retrain after drift alerts.

The experiment also reports Population Stability Index for the input features.

## Dataset

[Electricity (OpenML ID 151)](https://www.openml.org/d/151): 45,312 ordered
observations with `UP`/`DOWN` labels. The original row sequence is preserved. The normalized `date` field contains known inconsistencies; decreases are reported but never used to reorder observations.
We exclude current electricity price measurements (`nswprice`, `vicprice`)
from the feature set to reduce target-related information leakage. Data is
retrieved directly from OpenML at runtime and is intentionally not committed.

## Setup

Python 3.10 or later. From the repository root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Download, test, and run:

```powershell
python scripts/download_data.py
python -m pytest -q
python scripts/run_experiment.py
python scripts/audit_data.py
```

A working internet connection is required only for the initial OpenML download.

## Results

Running the experiment creates these files under `results/`:

| File | Contents |
|---|---|
| `metrics.json` | Aggregated metrics and update counts |
| `window_metrics.csv` | Scores in successive future-data batches |
| `feature_shift.csv` | PSI for each feature and batch |
| `performance.png` | Accuracy progression by strategy |
| `feature_shift.png` | Feature-shift trends |
| `research_summary.md` | Research question, measured results and limitations |

Generated outputs are local by default. The independent data audit writes
`results/data_audit.json` and reports date reversals and suspected duplicate-day
column matches without changing the input records. See [data quality notes](docs/data_quality.md).
No accuracy claim is made until the experiment is run on the original benchmark.

## Repository layout

```text
Epoch/
├── src/epoch_research/
│   ├── audit.py          # Non-destructive source data quality checks
│   ├── data.py           # Load and validate the ordered dataset
│   ├── drift.py          # Feature-shift measurement
│   ├── experiment.py     # Four adaptation strategies
│   └── report.py         # Results and figures
├── scripts/
│   ├── audit_data.py
│   ├── download_data.py
│   └── run_experiment.py
├── tests/                # Offline regression tests
├── docs/methodology.md   # Experiment design and limitations
├── results/              # Generated locally
├── pyproject.toml
└── README.md
```

For full evaluation assumptions, read [methodology.md](docs/methodology.md).
