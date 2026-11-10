import json

import pytest

from epoch_research.experiment import Settings, run
from epoch_research.robustness import compare_batch_sizes, save_batch_robustness
from test_data import example


class NoDrift:
    drift_detected = False

    def update(self, value):
        pass


def test_batch_robustness_preserves_evaluation_rows(tmp_path):
    frame = example(1100)
    config = Settings(
        batch_size=100, retrain_every_batches=2,
        alarm_cooldown_batches=2,
        recent_window=300, minimum_retrain_samples=100
    )
    primary = run(frame, config, detector_factory=NoDrift)
    report, table = compare_batch_sizes(
        frame, config, sizes=(50, 100, 200),
        detector_factory=NoDrift, primary_run=primary
    )
    assert report["evaluated_rows"] == 770
    assert report["initial_training_rows"] == 330
    assert table.shape[0] == 12
    for detail in report["results"]:
        assert detail["settings"]["batch_size"] * detail["settings"]["retrain_every_batches"] == 200
        assert detail["settings"]["batch_size"] * detail["settings"]["alarm_cooldown_batches"] == 200
    baseline = table[(table["batch_size"] == 100) & (table["strategy"] == "static")].iloc[0]
    assert baseline["accuracy"] == primary[0]["strategies"]["static"]["accuracy"]
    save_batch_robustness(report, table, tmp_path)
    for name in ("batch_robustness.md", "batch_robustness.json", "batch_robustness.csv"):
        assert (tmp_path / name).stat().st_size > 0
    assert json.loads((tmp_path / "batch_robustness.json").read_text())["evaluated_rows"] == 770


def test_invalid_batch_sizes_rejected():
    with pytest.raises(ValueError, match="distinct positive"):
        compare_batch_sizes(example(1100), sizes=(128, 128))
