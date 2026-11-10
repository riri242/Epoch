# Epoch

**A Study of Temporal Data Shift and Model Adaptation**

Epoch is a small, reproducible machine-learning research study. It compares
four adaptation policies on the [Electricity stream (OpenML 151)](https://www.openml.org/d/151)
using an ordered, test-before-train evaluation. Its aim is to understand
**when updating a classifier helps, and when it does not**.

## Research design

**Primary experiment:** one four-feature dataset configuration, with
`period`, `nswdemand`, `vicdemand`, and `transfer`. The potentially
unreliable `day` and `date` fields and target-related current electricity
prices are excluded. The raw source order is preserved.

| Policy | Training rule |
|---|---|
| Static | Initial fit only; never update |
| Periodic | Retrain on recent labeled observations at scheduled intervals |
| Online | Incrementally update on each completed test batch |
| ADWIN | Retrain when changes in recent prediction errors trigger an alert |

The same model family, earliest 30% training records and later evaluation
records are used throughout. Metrics are accuracy, balanced accuracy, and
F1 for `UP`; PSI describes feature distribution changes.

Two **supplementary checks** (not competing primary experiments):
1. Reintroduce `day` to measure sensitivity to the questionable calendar field.
2. Vary the batch size (128/256/512) while approximately keeping scheduled
   update and cooldown intervals fixed in observed records. Different batch
   sizes also change label-feedback timing.

See the **[research report and measured findings](docs/research_report.md)**,
[methodology](docs/methodology.md), [data-quality audit](docs/data_quality.md)
and [day-feature check](docs/day_sensitivity_results.md).

## Reproduce on Windows PowerShell

Python 3.10+ is required. From the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_all.py
```

The final command downloads the source dataset if missing, validates it,
audits known irregularities, performs the primary study and robustness checks,
and saves all results. A network connection is needed to download the
benchmark initially.

For a previously set-up clone:

```powershell
git pull --ff-only origin main
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_all.py
```

## Outputs

All generated files are in `results/` (excluded from source control).

| File | Purpose |
|---|---|
| `research_report.md` | **Main paper-style report generated from measured data** |
| `metrics.json`, `window_metrics.csv` | Primary no-day metrics and time-block performance |
| `performance.png`, `feature_shift.png` | Primary experiment figures |
| `feature_shift.csv` | Feature PSI over the stream |
| `data_audit.json` | Source date anomalies and suspected partial duplicate |
| `day_sensitivity.md`, `day_sensitivity.json` | With-day versus no-day accuracy and other metric differences |
| `batch_robustness.md`, `batch_robustness.csv` | Performance with batch sizes 128, 256 and 512 |
| `research_summary.md` | Short primary-experiment summary |

GitHub Actions runs `python scripts/run_all.py` after tests and stores the
complete `results/` directory as the **epoch-results** workflow artifact.
[See workflow runs](https://github.com/riri242/Epoch/actions).

## Code organization

```text
Epoch/
├── src/epoch_research/  # Source: model, evaluation, drift, audit, reports
├── scripts/             # Small command-line entrypoints
│   ├── run_all.py       # Recommended: complete study
│   ├── run_experiment.py
│   ├── run_sensitivity.py
│   ├── audit_data.py
│   └── download_data.py
├── tests/               # Automated regression tests
├── docs/                # Methodology, data-quality notes, observed findings
├── results/             # Generated locally and in GitHub Actions
└── pyproject.toml
```

## Interpretation

This is an **exploratory, single-dataset benchmark**, not a novel model,
a production forecasting service, or proof of generalized superiority.
The source calendar is imperfect, source row order is not independently
verified against wall-clock time, and labels are assumed available after
each scored batch. See the generated report for full limitations.
