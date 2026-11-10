import numpy as np
import pytest

from epoch_research.experiment import Settings, run
from test_data import example


class NoDrift:
    drift_detected = False

    def update(self, value):
        pass


class AlarmEvery20:
    def __init__(self):
        self.count = 0
        self.drift_detected = False

    def update(self, value):
        self.count += 1
        self.drift_detected = self.count >= 20


def test_all_methods_evaluate_same_future_rows():
    frame = example(1200)
    config = Settings(batch_size=100, retrain_every_batches=2,
                      recent_window=300, minimum_retrain_samples=100)
    summary, windows, shift = run(frame, config, detector_factory=NoDrift)
    assert summary["initial_training_rows"] == 360
    assert summary["evaluated_rows"] == 840
    assert windows.groupby("strategy")["count"].sum().to_dict() == {
        "static": 840, "periodic": 840, "online": 840, "adwin": 840,
    }
    assert summary["strategies"]["static"]["model_updates"] == 0
    assert summary["strategies"]["periodic"]["model_updates"] == 4
    assert summary["strategies"]["online"]["model_updates"] == 9
    assert summary["strategies"]["adwin"]["model_updates"] == 0
    assert len(shift) == 9 * 5
    # Identical initial models must give identical predictions on the first batch.
    first = windows.loc[windows["block"].eq(0)]
    assert first["accuracy"].nunique() == 1


def test_adwin_updates_only_after_predictions():
    cfg = Settings(batch_size=80, retrain_every_batches=2,
                   minimum_retrain_samples=100, alarm_cooldown_batches=2)
    summary, windows, _ = run(example(1000), cfg, detector_factory=AlarmEvery20)
    assert summary["strategies"]["adwin"]["model_updates"] > 0
    assert windows.loc[windows["block"].eq(0), "accuracy"].nunique() == 1


def test_configuration_rejects_invalid_sizes():
    with pytest.raises(ValueError, match="positive"):
        Settings(batch_size=0).check()
