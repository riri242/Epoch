# Electricity data integrity

Epoch evaluates the original OpenML Electricity (ELEC2, data ID 151) row sequence. Its normalized `date` field is **not** a dependable timestamp. In the 45,312-row download examined for this project, backward jumps occurred at row indices 25488, 34896, 35232, 36240 and 40704 (zero-based).

A published inspection of ELEC2 also reports a repeated NSW demand day around rows 17376–17423, and warns that the `day` field may be offset following that duplication. This report **does not prove** that the other measurements or target labels are duplicated, so the project must not silently remove rows based on this observation alone.

The default benchmark therefore:
- Preserves the original source row order and does not sort on `date`.
- Excludes both `date` and `day` from primary model predictors.
- Treats all findings as *ordered-stream* results, not independently verified real-time forecasts.
- Keeps the downloaded data unmodified for reproducibility.

## Measured source audit (GitHub Actions, 2026-11-10)

The audit ran against the downloaded OpenML dataset (45,312 records) and found:

| Column | Identical observations in the two candidate 48-row days |
|---|---:|
| `date` | 48 / 48 |
| `day` | 0 / 48 |
| `period` | 48 / 48 |
| `nswprice` | 47 / 48 |
| `nswdemand` | 48 / 48 |
| `vicprice` | 0 / 48 |
| `vicdemand` | 0 / 48 |
| `transfer` | 0 / 48 |
| `class` | 46 / 48 |

**No complete row matches across these two days (0 / 48)**. The evidence is consistent with a partially repeated source segment rather than a full-record duplicate. The raw observations are retained, and the source's noisy calendar labels remain a limitation. The five `date` reversals occur at zero-based rows 25488, 34896, 35232, 36240, and 40704.

Reproduce the findings using `python scripts/audit_data.py` or the GitHub Actions **Epoch research validation** workflow. The workflow stores `results/data_audit.json` alongside model reports as an artifact.

**Sensitivity check:** run `python scripts/run_sensitivity.py`, or use `python scripts/run_all.py` for the complete study (also executed by GitHub Actions). This compares all four strategies with and without `day` under the same stream split and update settings, writing the paired scores to `results/day_sensitivity.md` and `results/day_sensitivity.json`. Investigate source metadata further and avoid claiming the original calendar timestamps have been corrected. Do not claim that the dataset has been fully cleaned or that the stream order is independently calendar-verified. The five backward jumps are recorded as source data-quality caveats, not interpreted as genuine distribution drift.

References:
- [OpenML Electricity dataset](https://www.openml.org/d/151)
- [Ludvig Hult, *Real World Data Sets are Messy: Case Study Elec2* (2023)](https://el-hult.github.io/2023/12/21/elec2.html)
