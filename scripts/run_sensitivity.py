"""Run the paired day-feature ablation and save the results."""

import argparse
import json

from epoch_research.data import load_electricity
from epoch_research.experiment import Settings
from epoch_research.sensitivity import compare_day_feature, save_day_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/electricity.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--retrain-every", type=int, default=8)
    args = parser.parse_args()

    frame = load_electricity(args.data)
    settings = Settings(
        batch_size=args.batch_size, retrain_every_batches=args.retrain_every
    )
    result, windows, shifts = compare_day_feature(frame, settings)
    save_day_sensitivity(result, windows, shifts, args.output)
    print(json.dumps({
        "evaluated_rows": result["evaluated_rows"],
        "comparison": {
            name: {
                "accuracy_with_day": value["with_day"]["accuracy"],
                "accuracy_without_day": value["without_day"]["accuracy"],
                "delta_accuracy_pp": value["delta_percentage_points"]["accuracy"],
                "delta_balanced_accuracy_pp": value["delta_percentage_points"]["balanced_accuracy"],
                "delta_f1_up_pp": value["delta_percentage_points"]["f1_up"],
            }
            for name, value in result["comparison"].items()
        },
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
