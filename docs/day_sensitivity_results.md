# Epoch — Day feature sensitivity results

The GitHub Actions experiment evaluated the original Electricity benchmark (OpenML 151) using the original ordered 45,312 records. The initial training partition was 13,593 rows, followed by 31,719 held-forward observations. All four strategies used the same settings and scored the same observations in both feature configurations. The only experimental difference was omitting the potentially unreliable `day` predictor from the second run.

## Measured results (2026-11-10)

Accuracy values below are percentages; deltas are **without day minus with day**, in *percentage points*.

| Policy | Accuracy with day | Accuracy without day | Difference (pp) |
|---|---:|---:|---:|
| Static | 63.287% | 63.246% | -0.041 |
| Periodic | 61.465% | 61.162% | -0.303 |
| Online | 66.544% | 66.547% | +0.003 |
| ADWIN | 61.572% | 61.411% | -0.161 |

Changes in the other evaluation metrics (without day minus with day, in percentage points):

| Policy | Balanced accuracy delta | F1 (UP) delta |
|---|---:|---:|
| Static | +0.063 | +0.229 |
| Periodic | +0.167 | +1.373 |
| Online | -0.166 | -0.545 |
| ADWIN | +0.130 | +0.855 |

**Interpretation:** Online learning remained the highest-accuracy policy under both feature sets. Its accuracy changed by only +0.003 percentage points when `day` was removed. This result supports a narrow robustness claim about the *accuracy ranking in this experiment*, not a general claim that the temporal data are correct or that removing `day` improves every metric. Balanced accuracy and F1 change in different directions across policies.

The source order and timestamps were **not corrected**, and no rows were deleted. Because the source calendar labels contain known anomalies, conclusions are framed as ordered-stream benchmark results rather than verified real-world time forecasts.

## Reproduce

- Script: `python scripts/run_sensitivity.py` (requires the downloaded Electricity dataset)
- Automated runner: [Epoch experiment — GitHub Actions](https://github.com/riri242/Epoch/actions)
- Machine-readable output: `results/day_sensitivity.json`
- Per-block no-day metrics: `results/day_sensitivity_windows.csv`
- No-day PSI: `results/day_sensitivity_feature_shift.csv`
- Generated summary: `results/day_sensitivity.md`

The GitHub Actions run stores these files in the `epoch-results` artifact. Because they are generated outputs, they are not committed to source control.
