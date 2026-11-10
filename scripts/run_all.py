"""Reproduce the primary study and two supplementary checks in one command."""

import argparse
import json
from pathlib import Path

from epoch_research.audit import audit_electricity
from epoch_research.data import download_electricity, load_electricity
from epoch_research.experiment import Settings, run
from epoch_research.report import save_results
from epoch_research.robustness import compare_batch_sizes, save_batch_robustness
from epoch_research.sensitivity import compare_day_feature, save_day_sensitivity
from epoch_research.study_report import write_research_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/electricity.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--refresh-data", action="store_true")
    args = parser.parse_args()

    data = Path(args.data)
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=True)
    if args.refresh_data or not data.is_file():
        download_electricity(data)
    frame = load_electricity(data)
    audit = audit_electricity(frame)
    (folder / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    settings = Settings()
    primary = run(frame, settings)
    save_results(*primary, folder)

    day_result, day_windows, day_shift = compare_day_feature(
        frame, settings, primary_run=primary
    )
    save_day_sensitivity(day_result, day_windows, day_shift, folder)

    batch_result, batch_table = compare_batch_sizes(
        frame, settings, primary_run=primary
    )
    save_batch_robustness(batch_result, batch_table, folder)

    report_path = write_research_report(
        primary[0], day_result, batch_result, batch_table, audit, folder
    )
    print(json.dumps({
        "primary_features": primary[0]["features"],
        "evaluated_rows": primary[0]["evaluated_rows"],
        "accuracy": {
            name: values["accuracy"]
            for name, values in primary[0]["strategies"].items()
        },
        "batch_size_accuracy": batch_table.pivot(
            index="batch_size", columns="strategy", values="accuracy"
        ).round(6).to_dict(orient="index"),
        "research_report": str(report_path),
    }, indent=2))


if __name__ == "__main__":
    main()
