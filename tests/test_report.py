from epoch_research.experiment import Settings, run
from epoch_research.report import save_results
from test_data import example


class QuietDetector:
    drift_detected = False

    def update(self, value):
        return None


def test_report_writes_all_artifacts(tmp_path):
    metrics, windows, drift = run(
        example(900),
        Settings(batch_size=120, retrain_every_batches=3,
                 recent_window=240, minimum_retrain_samples=100),
        detector_factory=QuietDetector,
    )
    save_results(metrics, windows, drift, tmp_path)
    for name in (
        "metrics.json", "window_metrics.csv", "feature_shift.csv",
        "performance.png", "feature_shift.png", "research_summary.md",
    ):
        assert (tmp_path / name).is_file()
        assert (tmp_path / name).stat().st_size > 0
