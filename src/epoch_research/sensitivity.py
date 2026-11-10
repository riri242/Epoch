"""Compare the original inputs with an otherwise identical no-day experiment."""

import json
from pathlib import Path

import pandas as pd

from dataclasses import asdict

from .data import FEATURES, FEATURES_WITH_DAY
from .experiment import Settings, run

STRATEGIES = ("static", "periodic", "online", "adwin")
METRICS = ("accuracy", "balanced_accuracy", "f1_up")


def compare_day_feature(frame: pd.DataFrame, settings: Settings = Settings(), detector_factory=None, primary_run=None):
    baseline, base_windows, _ = run(
        frame, settings, detector_factory=detector_factory, feature_names=FEATURES_WITH_DAY
    )
    if primary_run is None:
        without, windows, shifts = run(
            frame, settings, detector_factory=detector_factory, feature_names=FEATURES
        )
    else:
        without, windows, shifts = primary_run
        if without["features"] != list(FEATURES) or without["settings"] != asdict(settings):
            raise ValueError("Provided primary run has incompatible features or settings")

    keys = ["strategy", "block", "start_row", "end_row", "count"]
    if not base_windows[keys].equals(windows[keys]):
        raise ValueError("Feature ablation must evaluate identical observation blocks")

    comparisons = {}
    for strategy in STRATEGIES:
        first = baseline["strategies"][strategy]
        second = without["strategies"][strategy]
        comparisons[strategy] = {
            "with_day": first,
            "without_day": second,
            "delta_percentage_points": {
                metric: 100 * (second[metric] - first[metric]) for metric in METRICS
            },
            "difference_in_updates": second["model_updates"] - first["model_updates"],
        }

    result = {
        "question": "How sensitive are adaptation strategies to the potentially unreliable day feature?",
        "initial_training_rows": baseline["initial_training_rows"],
        "evaluated_rows": baseline["evaluated_rows"],
        "settings": baseline["settings"],
        "with_day_features": list(FEATURES_WITH_DAY),
        "without_day_features": list(FEATURES),
        "comparison": comparisons,
        "raw_row_order_modified": False,
        "note": "Feature ablation does not repair source timestamps or establish causal drift.",
    }
    return result, windows, shifts


def save_day_sensitivity(result: dict, windows: pd.DataFrame, shifts: pd.DataFrame, output: str | Path):
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "day_sensitivity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    windows.to_csv(folder / "day_sensitivity_windows.csv", index=False)
    shifts.to_csv(folder / "day_sensitivity_feature_shift.csv", index=False)

    lines = [
        "# Epoch — Day feature sensitivity",
        "",
        "The following ablation removes only the `day` predictor. The raw observations,",
        "original order, warm-up rows, labels, training policies, random seed and",
        "future evaluation blocks are held fixed.",
        "",
        f"- Warm-up rows: {result['initial_training_rows']}",
        f"- Evaluated rows: {result['evaluated_rows']}",
        f"- With day: {', '.join(result['with_day_features'])}",
        f"- Without day: {', '.join(result['without_day_features'])}",
        "",
        "## Results",
        "",
        "Deltas are without day minus with day, in percentage points.",
        "",
        "| Strategy | Accuracy with | Accuracy without | Delta (pp) | Balanced with | Balanced without | Delta (pp) | F1 UP with | F1 UP without | Delta (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in STRATEGIES:
        case = result["comparison"][name]
        a, b, diff = case["with_day"], case["without_day"], case["delta_percentage_points"]
        lines.append(
            f"| {name} | {a['accuracy']:.3f} | {b['accuracy']:.3f} | {diff['accuracy']:+.2f} | "
            f"{a['balanced_accuracy']:.3f} | {b['balanced_accuracy']:.3f} | {diff['balanced_accuracy']:+.2f} | "
            f"{a['f1_up']:.3f} | {b['f1_up']:.3f} | {diff['f1_up']:+.2f} |"
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "This is a paired feature ablation, not a correction of calendar errors.",
        "Removing day can help or hurt a strategy for reasons beyond source quality.",
        "One dataset and one fixed configuration do not establish generalization.",
        "Model-update counts and per-block details are in the JSON and CSV outputs.",
        "",
    ])
    (folder / "day_sensitivity.md").write_text("\n".join(lines), encoding="utf-8")
