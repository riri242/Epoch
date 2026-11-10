import json

from epoch_research.audit import audit_electricity
from epoch_research.experiment import Settings, run
from epoch_research.robustness import compare_batch_sizes
from epoch_research.sensitivity import compare_day_feature
from epoch_research.study_report import write_research_report
from test_data import example


class NoDrift:
    drift_detected = False

    def update(self, value):
        pass


def test_structured_report_uses_measured_values(tmp_path):
    frame = example(900)
    settings = Settings(batch_size=100, retrain_every_batches=2,
                        recent_window=240, minimum_retrain_samples=100,
                        alarm_cooldown_batches=2)
    primary = run(frame, settings, detector_factory=NoDrift)
    day, _, _ = compare_day_feature(
        frame, settings, detector_factory=NoDrift, primary_run=primary
    )
    batch, table = compare_batch_sizes(
        frame, settings, sizes=(50, 100, 200),
        detector_factory=NoDrift, primary_run=primary
    )
    path = write_research_report(
        primary[0], day, batch, table, audit_electricity(frame), tmp_path
    )
    report = path.read_text(encoding="utf-8")
    assert "# Epoch: A Study" in report
    assert "## 2. Data and methods" in report
    assert "## 3. Results" in report
    assert "## 5. Limitations" in report
    assert str(primary[0]["evaluated_rows"]) in report
    assert f"{primary[0]['strategies']['online']['accuracy']:.3f}" in report
    assert "without day" in report
    assert "128" not in report.split("### 3.3 Batch-size sensitivity")[1].split("## 4. Discussion")[0]
