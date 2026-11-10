"""Save reproducible figures and a concise research summary."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_results(summary: dict, blocks: pd.DataFrame, shifts: pd.DataFrame, output: str | Path):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    blocks.to_csv(folder / "window_metrics.csv", index=False)
    shifts.to_csv(folder / "feature_shift.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    for method, group in blocks.groupby("strategy", sort=False):
        ax.plot(group["end_row"], group["accuracy"].rolling(5, min_periods=1).mean(), label=method)
    ax.set(xlabel="Observations processed", ylabel="Accuracy (five-block moving average)",
           title="Performance over the stream", ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder / "performance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for feature, group in shifts.groupby("feature", sort=False):
        ax.plot(group["end_row"], group["psi"], label=feature)
    ax.set(xlabel="Observations processed", ylabel="Population Stability Index",
           title="Feature distribution shift relative to warm-up data")
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(folder / "feature_shift.png", dpi=150)
    plt.close(fig)

    table = "\n".join(
        f"| {name} | {v['accuracy']:.3f} | {v['balanced_accuracy']:.3f} | "
        f"{v['f1_up']:.3f} | {v['model_updates']} |"
        for name, v in summary["strategies"].items()
    )
    note = f"""# Epoch — Experiment Results

## Research question
How do lightweight model adaptation strategies perform on chronologically
ordered electricity market observations?

## Evaluation protocol
- **Dataset:** OpenML Electricity (ID 151), original row order preserved.
- **Train / test:** {summary['initial_training_rows']} initial rows; {summary['evaluated_rows']} future rows.
- **Features:** {', '.join(summary['features'])}. Current price features are excluded to reduce target-related leakage.
- **Prediction:** each block is evaluated before its labels are used for updates.
- **Models:** the same initial scaled SGD logistic classifier, followed by static,
  periodic, mini-batch online or ADWIN-triggered updates.

## Measured results
| Strategy | Accuracy | Balanced accuracy | F1 (UP) | Updates |
|---|---:|---:|---:|---:|
{table}

ADWIN alarmed in {summary['adwin_alarm_blocks']} evaluated blocks.

See `performance.png`, `feature_shift.png`, and the machine-readable CSV/JSON files.

## Interpretation and limitations
This is an *exploratory, single-stream benchmark*, not proof that a particular
strategy generalizes to other domains. Feature PSI describes covariate shift;
it does **not** by itself establish concept drift. The dataset's timestamp is
normalized and the benchmark has known documentation inconsistencies. The
row order is treated as the temporal sequence without reconstructing calendar
times. Labels are assumed to become available immediately after a batch is
scored. Hyperparameters are predefined rather than optimized on test results.
The methods also vary in model update cost; outcomes should be interpreted
alongside training time and number of updates.

## Next research step
Repeat on a second chronologically labeled dataset and test sensitivity to
batch size and update frequency before drawing stronger conclusions.
"""
    (folder / "research_summary.md").write_text(note, encoding="utf-8")
