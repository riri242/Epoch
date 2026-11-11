"""Verify dashboard figures are sourced from the paper or generated data."""

import json

import pandas as pd
import pytest

from dashboard_utils import (
    feature_shift,
    published_batch_check,
    published_day_check,
    published_primary,
    published_report,
    window_metrics,
)


@pytest.fixture
def dashboard_root(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "research_report.md").write_text(
        """# Study

### 3.1 Primary experiment (without `day`)

| Strategy | Accuracy | Balanced accuracy | F1 (UP) |
|---|---:|---:|---:|
| Static | 63.25% | 63.46% | **59.44%** |
| Periodic | 61.16% | 59.37% | 51.06% |
| Online | **66.55%** | **64.47%** | 56.44% |
| ADWIN | 61.41% | 59.67% | 51.52% |

### 3.2 Supplementary check A — sensitivity to day

| Strategy | Accuracy with day | Without day | Change (pp) |
|---|---:|---:|---:|
| Static | 63.29% | 63.25% | −0.04 |
| Periodic | 61.46% | 61.16% | −0.30 |
| Online | 66.54% | 66.55% | +0.00 |
| ADWIN | 61.57% | 61.41% | −0.16 |

### 3.3 Supplementary check B — batch size

| Batch size | Static accuracy | Periodic accuracy | Online accuracy | ADWIN accuracy |
|---:|---:|---:|---:|---:|
| 128 | 63.25% | 61.16% | **65.94%** | 62.49% |
| 256 | 63.25% | 61.16% | **66.55%** | 61.41% |
| 512 | 63.25% | 61.16% | **66.01%** | 61.75% |
""",
        encoding="utf-8",
    )
    return tmp_path


def test_dashboard_works_with_published_report_only(dashboard_root):
    primary, source, summary = published_primary(dashboard_root)
    assert source == "published" and summary is None
    assert primary.loc[primary["Strategy"].eq("Online"), "Accuracy"].iloc[0] == 66.55
    assert published_day_check(dashboard_root).loc[
        lambda table: table["Strategy"].eq("Online"), "Difference (pp)"
    ].iloc[0] == 0
    sizes = published_batch_check(dashboard_root)
    assert len(sizes) == 12
    assert sizes.loc[
        sizes["Strategy"].eq("Online") & sizes["Batch size"].eq(128),
        "Accuracy",
    ].iloc[0] == 65.94
    assert "Primary experiment" in published_report(dashboard_root)
    assert feature_shift(dashboard_root) is None
    assert window_metrics(dashboard_root) is None


def test_dashboard_prioritizes_generated_results(dashboard_root):
    result = dashboard_root / "results"
    result.mkdir()
    metrics = {
        "dataset_rows": 1200, "evaluated_rows": 840,
        "strategies": {
            name: {
                "accuracy": accuracy,
                "balanced_accuracy": accuracy,
                "f1_up": accuracy,
                "model_updates": updates,
            }
            for name, accuracy, updates in [
                ("static", 0.61, 0), ("periodic", 0.62, 2),
                ("online", 0.69, 8), ("adwin", 0.63, 1)
            ]
        },
    }
    (result / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (result / "day_sensitivity.json").write_text(json.dumps({
        "comparison": {
            "online": {
                "with_day": {"accuracy": 0.68},
                "without_day": {"accuracy": 0.69},
                "delta_percentage_points": {"accuracy": 1.0},
            }
        }
    }), encoding="utf-8")
    pd.DataFrame([{
        "batch_size": 256, "strategy": "online", "accuracy": 0.69,
        "balanced_accuracy": 0.68, "f1_up": 0.67,
    }]).to_csv(result / "batch_robustness.csv", index=False)

    primary, source, summary = published_primary(dashboard_root)
    assert source == "local"
    assert summary["evaluated_rows"] == 840
    assert primary.loc[primary["Strategy"].eq("Online"), "Accuracy"].iloc[0] == pytest.approx(69)
    assert published_day_check(dashboard_root).iloc[0]["Difference (pp)"] == 1.0
    assert published_batch_check(dashboard_root).iloc[0]["Accuracy"] == pytest.approx(69)
