# Research methodology

## Aim and question

Epoch asks whether updating a classifier in response to newly available
labeled observations improves predictive performance compared with a static
classifier on an ordered electricity-market stream.

## Dataset and preprocessing

- Benchmark: OpenML Electricity, data ID 151; 45,312 observations.
- Target: `UP` versus `DOWN`.
- **Primary input fields:** `period`, `nswdemand`, `vicdemand`, `transfer`.
- Excluded from primary predictors: `day`, `date`, `nswprice`, `vicprice`.
  The former two have calendar-quality concerns; the latter two are closely
  related to the target and could enable a misleading evaluation.
- Raw source row order and labels are never modified or sorted.
- The normalized `date` contains five backward jumps. A candidate repeated
  48-record period has repeated NSW demand but differing other columns.
  See [data-quality notes](data_quality.md).

The **earliest 30%** of observations is used for initial model fitting and
scaler estimation; the remaining **70%** is evaluated in original row order.
All strategies share exactly the same training and evaluation records.

## Classifier and prequential evaluation

Every strategy starts from a `StandardScaler` fitted **only** on the warm-up
data and an `SGDClassifier(loss="log_loss", alpha=0.0001, random_state=42)`.
The shared scaler is fixed. The study uses batches of **256 records** by
default: models predict each incoming batch before any labels from that batch
are shown. Adaptive methods then update after the complete batch is scored.
This simulates **batch-delayed feedback**, not immediate per-observation labels.

| Strategy | Update policy |
|---|---|
| Static | Never update after warm-up |
| Periodic | Refit on the most recent 6,000 labeled rows every 8 batches |
| Online | Apply one `partial_fit` after each evaluated batch |
| ADWIN | Monitor already-scored prediction errors and refit on the most recent 6,000 labeled rows after a drift alert, with a 4-batch cooldown |

ADWIN uses its default delta of 0.002; retraining is triggered by changes
in model error, which **does not prove concept drift**.

## Outcome measures

Primary: accuracy, balanced accuracy, and F1 of class `UP`.
Secondary: per-block accuracy, model update counts and fitting time.
The population stability index (PSI) compares each input feature's
distribution with the initial training period. PSI is a *descriptive*
covariate-shift measure; no causal inference is made from it.

## Robustness checks

1. **Day-feature ablation:** reintroduce `day` using the same model, seed,
   original rows, initial split, batch size, update policies and scoring
   protocol. The difference is `without day − with day` in percentage
   points. This does not fix the source calendar.
2. **Batch-size comparison:** evaluate 128, 256 and 512 observations per
   batch, using primary no-day features. The periodic interval is adjusted
   to approximately 2,048 observations and ADWIN cooldown to 1,024
   observations. Batch size also changes label-feedback delays and
   adaptation boundaries, so changes cannot be attributed to batch size alone.

## Validity and interpretation

- Stream ordering follows source rows, **not verified wall-clock timestamps**.
- Calendar anomalies and a partially repeated source segment remain.
- The label-feedback timing is an experimental assumption.
- Only one dataset, classifier family and seed are studied; there is no
  formal statistical test or confidence interval.
- Hyperparameters were fixed in advance rather than optimized on the
  held-forward observations.
- Strategy cost differs by update frequency and training history.
- PSI and changes in model error are not definitive evidence of concept drift.

## Reproducibility

From the project root, install `.[dev]`, then run
`python -m pytest -q` and `python scripts/run_all.py`.
Results are saved in `results/` and uploaded by GitHub Actions
as the `epoch-results` artifact.

## Sources

- [OpenML Electricity data ID 151](https://www.openml.org/d/151)
- [Documented Elec2 data-quality caveats](https://el-hult.github.io/2023/12/21/elec2.html)
- [River ADWIN implementation](https://riverml.xyz/latest/api/drift/ADWIN/)
