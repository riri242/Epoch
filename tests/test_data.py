import numpy as np
import pandas as pd
import pytest

from epoch_research.data import FEATURES, count_date_decreases, download_electricity, features_target, validate_electricity
from epoch_research.drift import population_stability_index


def example(n=200):
    return pd.DataFrame({
        "date": np.arange(n, dtype=float) / n,
        "day": np.arange(n) % 7,
        "period": np.arange(n) % 48,
        "nswdemand": np.arange(n) / 100,
        "vicdemand": np.arange(n) / 200,
        "transfer": np.sin(np.arange(n)),
        "nswprice": 1e6 + np.arange(n),
        "vicprice": 1e6 + np.arange(n),
        "class": np.where(np.arange(n) % 2, "UP", "DOWN"),
    })


def test_target_related_price_fields_excluded():
    assert "nswprice" not in FEATURES and "vicprice" not in FEATURES
    assert "day" not in FEATURES
    x, y = features_target(example())
    assert x.shape == (200, 4)
    assert set(y) == {0, 1}


def test_missing_and_invalid_date_rejected():
    frame = example()
    frame.loc[1, "transfer"] = np.nan
    with pytest.raises(ValueError, match="missing values"):
        validate_electricity(frame)
    frame = example()
    frame.loc[50, "date"] = np.inf
    with pytest.raises(ValueError, match="Date feature must be finite"):
        validate_electricity(frame)


def test_irregular_date_preserves_stream_order():
    frame = example()
    frame.loc[50, "date"] = -1
    validate_electricity(frame)
    assert count_date_decreases(frame) == 1
    assert frame.iloc[50]["date"] == -1


def test_download_preserves_source_rows(monkeypatch, tmp_path):
    frame = example()
    frame.loc[50, "date"] = -1
    class Dataset:
        pass
    dataset = Dataset()
    dataset.frame = frame
    monkeypatch.setattr("epoch_research.data.fetch_openml", lambda **kwargs: dataset)
    output = download_electricity(tmp_path / "electricity.csv")
    saved = pd.read_csv(output)
    assert saved["date"].tolist() == frame["date"].tolist()
    assert saved["class"].tolist() == frame["class"].tolist()


def test_population_stability_index_is_zero_for_same_sample():
    sample = np.linspace(-2, 2, 500)
    assert population_stability_index(sample, sample) == pytest.approx(0, abs=1e-12)
    assert population_stability_index(sample, sample + 3) > 0.5
    # A constant reference feature may start varying later in the stream.
    assert population_stability_index(np.zeros(100), np.ones(100)) > 0.5
