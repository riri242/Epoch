"""Test robustness to pre-specified feedback batch sizes."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from .experiment import Settings, run

BATCH_SIZES = (128, 256, 512)
STRATEGIES = ("static", "periodic", "online", "adwin")


def compare_batch_sizes(frame, settings=Settings(), sizes=BATCH_SIZES,
                        detector_factory=None, primary_run=None):
    """Preserve the approximate periodic and cooldown intervals in observations."""
    if not sizes or len(set(sizes)) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError("Batch sizes must be distinct positive integers")
    interval = settings.batch_size * settings.retrain_every_batches
    cooldown = settings.batch_size * settings.alarm_cooldown_batches
    records, details = [], []
    warmup, evaluated = None, None

    for size in sizes:
        config = replace(
            settings,
            batch_size=size,
            retrain_every_batches=max(1, round(interval / size)),
            alarm_cooldown_batches=max(1, round(cooldown / size)),
        )
        if primary_run is not None and size == settings.batch_size:
            summary = primary_run[0]
            if summary["settings"] != asdict(config):
                raise ValueError("Provided primary run settings do not match")
        else:
            summary, _, _ = run(frame, config, detector_factory=detector_factory)
        if warmup is None:
            warmup, evaluated = summary["initial_training_rows"], summary["evaluated_rows"]
        if (warmup, evaluated) != (summary["initial_training_rows"], summary["evaluated_rows"]):
            raise ValueError("Robustness runs must evaluate the same observations")
        details.append({
            "batch_size": size, "settings": summary["settings"],
            "strategies": summary["strategies"],
            "adwin_alarm_blocks": summary["adwin_alarm_blocks"],
        })
        for strategy in STRATEGIES:
            value = summary["strategies"][strategy]
            records.append({
                "batch_size": size, "strategy": strategy,
                "accuracy": value["accuracy"],
                "balanced_accuracy": value["balanced_accuracy"],
                "f1_up": value["f1_up"],
                "model_updates": value["model_updates"],
            })
    return {
        "batch_sizes": list(sizes),
        "primary_batch_size": settings.batch_size,
        "initial_training_rows": warmup,
        "evaluated_rows": evaluated,
        "target_periodic_interval_observations": interval,
        "target_adwin_cooldown_observations": cooldown,
        "results": details,
        "caveat": "Batch size also changes label-feedback delay and adaptation boundaries.",
    }, pd.DataFrame(records)


def save_batch_robustness(result, table, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "batch_robustness.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    table.to_csv(output / "batch_robustness.csv", index=False)
    lines = [
        "# Batch-size robustness check", "",
        "All runs use the no-day feature set and the same ordered records.",
        "Periodic retraining and ADWIN cooldown are held approximately constant",
        "in observations; batch size also changes the time at which labels are revealed.",
        "",
        "| Batch | Strategy | Accuracy | Balanced accuracy | F1 (UP) | Updates |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| {row.batch_size} | {row.strategy} | {row.accuracy:.3f} | "
            f"{row.balanced_accuracy:.3f} | {row.f1_up:.3f} | {row.model_updates} |"
        )
    lines.extend([
        "", "## Limitations", "",
        "Different batch sizes imply different label-feedback delays and adaptation",
        "boundaries. Changes cannot be attributed solely to the numerical batch size.",
        "This remains a single-dataset, single-seed exploratory analysis.", "",
    ])
    (output / "batch_robustness.md").write_text("\n".join(lines), encoding="utf-8")
