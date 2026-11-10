import json

import pytest

from epoch_research.data import FEATURES, FEATURES_WITH_DAY, features_target
from epoch_research.experiment import Settings, run
from epoch_research.sensitivity import compare_day_feature, save_day_sensitivity
from test_data import example


class NoDrift:
    drift_detected = False

    def update(self, value):
        pass


def test_no_day_feature_selection():
    frame = example(120)
    reduced, labels = features_target(frame)
    full, original_labels = features_target(frame, FEATURES_WITH_DAY)
    assert reduced.shape == (120, 4)
    assert full.shape == (120, 5)
    assert (reduced == full[:, 1:]).all()
    assert (labels == original_labels).all()
    with pytest.raises(ValueError, match="unique supported"):
        features_target(frame, ("period", "period"))
    with pytest.raises(ValueError, match="unique supported"):
        features_target(frame, ("nswprice",))


def test_paired_sensitivity_uses_identical_evaluation_records(tmp_path):
    frame = example(1200)
    settings = Settings(
        batch_size=100, retrain_every_batches=2,
        recent_window=300, minimum_retrain_samples=100
    )
    comparison, windows, shifts = compare_day_feature(
        frame, settings, detector_factory=NoDrift
    )
    baseline, base_windows, _ = run(
        frame, settings, detector_factory=NoDrift
    )

    assert comparison["evaluated_rows"] == baseline["evaluated_rows"] == 840
    assert comparison["with_day_features"] == list(FEATURES_WITH_DAY)
    assert comparison["without_day_features"] == list(FEATURES)
    assert "day" not in shifts["feature"].unique()
    keys = ["strategy", "block", "start_row", "end_row", "count"]
    assert windows[keys].equals(base_windows[keys])
    for name, record in comparison["comparison"].items():
        assert record["without_day"]["accuracy"] == baseline["strategies"][name]["accuracy"]
        assert record["delta_percentage_points"]["accuracy"] == pytest.approx(
            100 * (record["without_day"]["accuracy"] - record["with_day"]["accuracy"])
        )

    save_day_sensitivity(comparison, windows, shifts, tmp_path)
    result = json.loads((tmp_path / "day_sensitivity.json").read_text(encoding="utf-8"))
    assert result["evaluated_rows"] == 840
    for name in (
        "day_sensitivity.md", "day_sensitivity_windows.csv",
        "day_sensitivity_feature_shift.csv"
    ):
        assert (tmp_path / name).stat().st_size > 0
