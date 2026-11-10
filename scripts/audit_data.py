"""Write an audit of the Electricity benchmark without modifying the data."""

import argparse
import json
from pathlib import Path

from epoch_research.audit import audit_electricity
from epoch_research.data import load_electricity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/electricity.csv")
    parser.add_argument("--output", default="results/data_audit.json")
    args = parser.parse_args()

    report = audit_electricity(load_electricity(args.data))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Audit saved: {path}")


if __name__ == "__main__":
    main()
