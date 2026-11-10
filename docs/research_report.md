# Epoch: A Study of Temporal Data Shift and Model Adaptation

**Research status:** completed exploratory benchmark (version 1) · November 2026

## Abstract

This study evaluates whether updating a machine-learning classifier improves performance when observations arrive sequentially. Four adaptation policies—Static, Periodic, Online and ADWIN-triggered retraining—were compared on the historical Electricity benchmark (OpenML ID 151). The primary analysis excludes unreliable calendar fields, while supplementary tests assess sensitivity to the `day` input and to the number of observations in each feedback batch. Online achieved the highest accuracy in the primary run (66.55%), compared with Static (63.25%). Its advantage persisted across the three tested batch sizes. However, Static obtained a higher F1 score for the `UP` class. Findings are exploratory and specific to this benchmark.

## 1. Introduction

**Research question:** Can lightweight model-updating policies improve classification performance on an ordered data stream compared with a model that is never updated?

The study compares established update strategies rather than proposing a new machine-learning algorithm.

## 2. Dataset and methodology

- **Source:** [Electricity (OpenML ID 151)](https://www.openml.org/d/151), 45,312 observations; `UP`/`DOWN` classification.
- **Primary predictors:** `period`, `nswdemand`, `vicdemand`, `transfer`. The fields `day` and `date` are excluded because of known calendar inconsistencies; contemporaneous price fields are excluded due to their close relation to the target.
- **Train/evaluation:** earliest 13,593 records (30%) for warm-up; remaining 31,719 records for ordered evaluation.
- **Model:** SGD logistic classifier with a scaler fitted on the warm-up data only, fixed seed 42.
- **Protocol:** predict each block before using labels from that same block for adaptation. The primary block size is 256 observations. This assumes batch-delayed, rather than instantaneous, label feedback.
- **Policies:** Static never updates; Periodic retrains every eight batches on up to 6,000 recent labeled rows; Online updates incrementally after each scored batch; ADWIN detects changes in past prediction errors and triggers retraining after a cooldown.
- **Metrics:** accuracy, balanced accuracy, F1 for `UP`, plus update counts and feature PSI. PSI is descriptive, not proof of concept drift.

The original row order was preserved. Neither the normalized date values nor any calendar reconstruction were used to reorder observations.

## 3. Results

### 3.1 Primary experiment (without `day`)

| Strategy | Accuracy | Balanced accuracy | F1 (UP) |
|---|---:|---:|---:|
| Static | 63.25% | 63.46% | **59.44%** |
| Periodic | 61.16% | 59.37% | 51.06% |
| Online | **66.55%** | **64.47%** | 56.44% |
| ADWIN | 61.41% | 59.67% | 51.52% |

**Interpretation:** Online improved accuracy over Static by approximately **3.30 percentage points**. Static retained the highest F1 for the positive class, so the preferred update policy depends on the evaluation criterion.

### 3.2 Supplementary check A — sensitivity to `day`

The only change in this check was inclusion of the calendar `day` field. All compared strategies used the same records, model family and evaluation settings.

| Strategy | Accuracy with `day` | Without `day` | Change (pp) |
|---|---:|---:|---:|
| Static | 63.29% | 63.25% | −0.04 |
| Periodic | 61.46% | 61.16% | −0.30 |
| Online | 66.54% | 66.55% | +0.00 |
| ADWIN | 61.57% | 61.41% | −0.16 |

Removing `day` had little influence on the accuracy ranking in this fixed configuration. Balanced accuracy and F1 were not uniformly improved; see the [supplementary feature report](day_sensitivity_results.md).

### 3.3 Supplementary check B — batch size

| Batch size | Static accuracy | Periodic accuracy | Online accuracy | ADWIN accuracy |
|---:|---:|---:|---:|---:|
| 128 | 63.25% | 61.16% | **65.94%** | 62.49% |
| 256 | 63.25% | 61.16% | **66.55%** | 61.41% |
| 512 | 63.25% | 61.16% | **66.01%** | 61.75% |

Online was the highest-accuracy policy at all three tested batch sizes. Scheduled periodic retraining and ADWIN cooldown were adjusted to hold their intervals approximately fixed in *numbers of observations*. **Batch size nevertheless changes feedback delay and adaptation boundaries**, so this should not be interpreted as the isolated causal effect of batch size.

### 3.4 Data-quality observations

The audit identified five backward jumps in the normalized `date` field, at source rows 25488, 34896, 35232, 36240 and 40704 (zero-based). A candidate partially repeated 48-record segment shared all NSW demand values with a neighboring segment, but **no entire rows** were identical. No rows were deleted, synthesized, or reordered; source timing inconsistencies remain documented in [data_quality.md](data_quality.md).

## 4. Discussion

The results suggest that continual adaptation can improve overall accuracy on this stream, but its benefit is not uniform across policies or metrics. Online's advantage across several batch sizes is useful evidence of stability in this particular evaluation. Periodic and ADWIN retraining did not outperform Static in primary accuracy. Feature PSI and the adaptive detector's alarms alone cannot establish that concept drift caused any observed improvement or decline.

## 5. Limitations

- Only one benchmark, model family and seed were evaluated; no confidence intervals, significance tests or independent replication were performed.
- The original row order was treated as the stream sequence, **not independently verified real-world calendar time**.
- Batch-delayed labels and other experimental assumptions may differ from production systems.
- Methods have different update frequencies and costs; fitting-time measurements depend on the execution environment.
- The batch-size robustness check also changes when feedback becomes available.
- The study cannot establish a universal best adaptation policy or prove causality between drift and performance.

## 6. Conclusion

Epoch provides a reproducible first-stage research study comparing four established adaptation policies, supported by a source-data audit and two limited robustness checks. Online achieved the highest primary accuracy and remained the top-accuracy policy at three pre-specified batch sizes, while Static performed better on class-specific F1. The results are suitable as a transparent exploratory case study; the strongest next extension would be evaluation on a second reliably timestamped dataset.

## Reproducibility and references

Run `python scripts/run_all.py` from the repository root to download the benchmark if needed, generate all primary and supplementary findings, and write a fresh detailed report to `results/research_report.md`. The [GitHub Actions workflow](https://github.com/riri242/Epoch/actions/runs/42856230058) confirms the 16 automated tests and complete reproducible pipeline; its `epoch-results` artifact contains the machine-readable measurements and generated figures.

- [OpenML Electricity, data ID 151](https://www.openml.org/d/151)
- [L. Hult — Elec2 data-quality discussion](https://el-hult.github.io/2023/12/21/elec2.html)
- [River ADWIN API](https://riverml.xyz/latest/api/drift/ADWIN/)
