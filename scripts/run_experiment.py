"""Run the complete Epoch experiment and save figures and metrics."""

import argparse
import json

from epoch_research.data import load_electricity
from epoch_research.experiment import Settings, run
from epoch_research.report import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/electricity.csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--retrain-every", type=int, default=8)
    args = parser.parse_args()

    frame = load_electricity(args.data)
    settings = Settings(batch_size=args.batch_size, retrain_every_batches=args.retrain_every)
    summary, blocks, shifts = run(frame, settings)
    save_results(summary, blocks, shifts, args.output)
    print(json.dumps({
        "rows": summary["dataset_rows"],
        "evaluated_rows": summary["evaluated_rows"],
        "strategies": summary["strategies"],
        "output": args.output,
    }, indent=2))


if __name__ == "__main__":
    main()
