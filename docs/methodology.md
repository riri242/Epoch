# Research Methodology

## Objective

Evaluate whether updating a classifier after observing new labeled examples
improves its performance under temporal distribution changes.

## Data

The Electricity benchmark (OpenML data ID 151) is treated as an ordered stream.
The original row sequence is retained. The task is to classify `UP` versus
`DOWN`; the target is based on price movement relative to a moving average.
`nswprice` and `vicprice` are excluded because price measurements can be
closely related to the label itself. Features used: `day`, `period`,
`nswdemand`, `vicdemand`, `transfer`.

## Experimental design

1. Fit a standard scaler on the earliest 30% only.
2. Train one SGD-based logistic classifier per strategy on those rows.
3. Evaluate each subsequent batch in the same order and before learning labels.
4. Allow adaptive methods to update only after the corresponding labels are revealed.
5. Compare identical held-forward test records across four strategies.

| Strategy | Update policy |
|---|---|
| Static | No updates after the first fit |
| Periodic | Refit on the last 6,000 labeled rows every 8 batches |
| Online | `partial_fit` after each scored batch |
| ADWIN | Monitor prediction errors; refit on last 6,000 labeled rows when signaled, with cooldown |

Feature-wise Population Stability Index (PSI) is computed against the training
period. PSI is a descriptive covariate-shift measure, not an established
detection of concept drift. ADWIN detects changes in the observed error stream,
which can indicate degradation but does not prove a cause.

## Outcome measures

Primary: accuracy, balanced accuracy and F1 for the `UP` class, measured on the
same future observations. Secondary: accuracy by block, the number of model
updates, total model fitting time and feature PSI.

## Threats to validity

- The normalized `date` field contains backward jumps, so row order rather than that field is the temporal index.
- Benchmark literature has pointed out inconsistencies in calendar coverage and
  potentially misleading time information. Results are indexed by record order.
- The availability of the contemporaneous features may vary in deployment;
  this is a controlled retrospective classification benchmark.
- Labels are assumed to arrive immediately after each scored batch.
- One dataset and one set of hyperparameters cannot demonstrate broad generalization.
- ADWIN alerts describe changes in model errors, not definitive concept drift.

## Sources

- OpenML Electricity (ID 151): https://www.openml.org/d/151
- Electricity benchmark caveats: https://el-hult.github.io/2023/12/21/elec2.html
- River ADWIN documentation: https://riverml.xyz/latest/api/drift/ADWIN/
