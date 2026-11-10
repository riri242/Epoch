import numpy as np
import pandas as pd

from epoch_research.audit import audit_electricity


def sample(n=160):
    return pd.DataFrame({
        "date": np.repeat(np.arange(n // 4), 4).astype(float),
        "day": (np.arange(n) // 4) % 7 + 1,
        "period": np.arange(n) % 4,
        "nswdemand": np.arange(n, dtype=float),
        "vicdemand": np.arange(n, dtype=float) + 100,
        "transfer": np.arange(n, dtype=float) + 200,
        "class": np.where(np.arange(n) % 2, "UP", "DOWN"),
    })


def test_audit_detects_only_true_column_repeats():
    frame = sample()
    start, block = 16, 4
    frame.loc[start + block:start + 2 * block - 1, "nswdemand"] = frame.loc[start:start + block - 1, "nswdemand"].to_numpy()
    frame.loc[start + block:start + 2 * block - 1, "date"] = frame.loc[start:start + block - 1, "date"].to_numpy()
    frame.loc[start + block, "date"] = -1

    before = frame.copy(deep=True)
    report = audit_electricity(frame, duplicate_start=start, day_rows=block)
    assert report["matching_values_per_column"]["nswdemand"] == block
    assert report["matching_values_per_column"]["vicdemand"] == 0
    assert report["matching_entire_rows"] == 0
    assert report["date_decrease_rows"] == [start + block]
    assert report["removal_performed"] is False
    pd.testing.assert_frame_equal(frame, before)


def test_audit_on_short_dataset_reports_not_examined():
    report = audit_electricity(sample())
    assert report["suspected_duplicate_examined"] is False
