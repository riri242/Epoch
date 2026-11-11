"""Load published and generated Epoch results for a read-only dashboard."""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
STRATEGIES = ("static", "periodic", "online", "adwin")


def _markdown_table(markdown: str, section: str) -> pd.DataFrame:
    """Read the first simple Markdown table after a specific report heading."""
    if section not in markdown:
        raise ValueError(f"Missing report section: {section}")
    region = markdown.split(section, 1)[1]
    lines = region.splitlines()
    start = next((i for i, line in enumerate(lines) if line.startswith("|")), None)
    if start is None:
        raise ValueError(f"No table found under: {section}")
    table_lines = []
    for line in lines[start:]:
        if not line.startswith("|"):
            break
        table_lines.append(line)
    if len(table_lines) < 3:
        raise ValueError(f"Incomplete Markdown table under: {section}")

    def cells(line):
        return [cell.strip().replace("**", "").replace("`", "")
                for cell in line.strip().strip("|").split("|")]

    headers = cells(table_lines[0])
    rows = [cells(line) for line in table_lines[2:]]
    if any(len(row) != len(headers) for row in rows):
        raise ValueError(f"Malformed Markdown table under: {section}")
    return pd.DataFrame(rows, columns=headers)


def _number(value) -> float:
    return float(str(value).replace("%", "").replace("−", "-").strip())


def _results_json(root: Path, name: str):
    path = root / "results" / name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _results_csv(root: Path, name: str):
    path = root / "results" / name
    return pd.read_csv(path) if path.is_file() else None


def published_report(root: Path = ROOT) -> str:
    report = root / "results" / "research_report.md"
    if not report.is_file():
        report = root / "docs" / "research_report.md"
    return report.read_text(encoding="utf-8")


def published_primary(root: Path = ROOT):
    """Return percent-scale metrics, and whether local experiment files were used."""
    summary = _results_json(root, "metrics.json")
    if summary is not None:
        rows = [
            {
                "Strategy": strategy.title(),
                "Accuracy": 100 * summary["strategies"][strategy]["accuracy"],
                "Balanced Accuracy": 100 * summary["strategies"][strategy]["balanced_accuracy"],
                "F1 (UP)": 100 * summary["strategies"][strategy]["f1_up"],
                "Updates": summary["strategies"][strategy]["model_updates"],
            }
            for strategy in STRATEGIES
        ]
        return pd.DataFrame(rows), "local", summary

    report = (root / "docs" / "research_report.md").read_text(encoding="utf-8")
    values = _markdown_table(report, "### 3.1 Primary experiment")
    return pd.DataFrame({
        "Strategy": values["Strategy"],
        "Accuracy": values["Accuracy"].map(_number),
        "Balanced Accuracy": values["Balanced accuracy"].map(_number),
        "F1 (UP)": values["F1 (UP)"].map(_number),
    }), "published", None


def published_day_check(root: Path = ROOT) -> pd.DataFrame:
    data = _results_json(root, "day_sensitivity.json")
    if data:
        return pd.DataFrame([
            {
                "Strategy": name.title(),
                "With day": 100 * info["with_day"]["accuracy"],
                "Without day": 100 * info["without_day"]["accuracy"],
                "Difference (pp)": info["delta_percentage_points"]["accuracy"],
            }
            for name, info in data["comparison"].items()
        ])
    report = (root / "docs" / "research_report.md").read_text(encoding="utf-8")
    frame = _markdown_table(report, "### 3.2 Supplementary check A")
    return pd.DataFrame({
        "Strategy": frame["Strategy"],
        "With day": frame["Accuracy with day"].map(_number),
        "Without day": frame["Without day"].map(_number),
        "Difference (pp)": frame["Change (pp)"].map(_number),
    })


def published_batch_check(root: Path = ROOT) -> pd.DataFrame:
    data = _results_csv(root, "batch_robustness.csv")
    if data is not None:
        return pd.DataFrame({
            "Batch size": data["batch_size"].astype(int),
            "Strategy": data["strategy"].str.title(),
            "Accuracy": data["accuracy"] * 100,
            "Balanced Accuracy": data["balanced_accuracy"] * 100,
            "F1 (UP)": data["f1_up"] * 100,
        })
    report = (root / "docs" / "research_report.md").read_text(encoding="utf-8")
    frame = _markdown_table(report, "### 3.3 Supplementary check B")
    long = frame.melt(id_vars="Batch size", var_name="Strategy", value_name="Accuracy")
    long["Batch size"] = long["Batch size"].astype(int)
    long["Strategy"] = long["Strategy"].str.replace(" accuracy", "", regex=False)
    long["Accuracy"] = long["Accuracy"].map(_number)
    return long


def window_metrics(root: Path = ROOT):
    return _results_csv(root, "window_metrics.csv")


def feature_shift(root: Path = ROOT):
    return _results_csv(root, "feature_shift.csv")
