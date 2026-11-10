"""Download and validate the chronological Electricity benchmark."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

DATA_ID = 151
TARGET = "class"
# Current electricity prices are excluded to avoid target-related leakage.
FEATURES = ("day", "period", "nswdemand", "vicdemand", "transfer")
EXCLUDED = ("nswprice", "vicprice")


def download_electricity(destination: str | Path = "data/electricity.csv") -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataset = fetch_openml(data_id=DATA_ID, as_frame=True, parser="auto")
    frame = dataset.frame.copy()
    validate_electricity(frame)
    jumps = count_date_decreases(frame)
    if jumps:
        print(f"Note: {jumps} decreases in the normalized date field; source row order preserved.")
    # Keep the original sequence for stream evaluation.
    frame.to_csv(destination, index=False)
    return destination


def validate_electricity(frame: pd.DataFrame) -> None:
    required = set(FEATURES) | {TARGET}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing dataset columns: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Empty dataset or missing values in required columns")
    if not set(frame[TARGET].astype(str).str.upper().unique()) <= {"UP", "DOWN"}:
        raise ValueError("Expected the Electricity UP/DOWN target labels")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Both target classes must be present")
    numeric = frame.loc[:, FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Features must be finite")
    if "date" in frame.columns:
        date = pd.to_numeric(frame["date"], errors="raise")
        if not np.isfinite(date.to_numpy(dtype=float)).all():
            raise ValueError("Date feature must be finite")


def count_date_decreases(frame: pd.DataFrame) -> int:
    """Report backward jumps without sorting a known noisy date field."""
    if "date" not in frame.columns:
        return 0
    date = pd.to_numeric(frame["date"], errors="raise")
    return int(date.diff().lt(-1e-8).sum())


def load_electricity(path: str | Path = "data/electricity.csv") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python scripts/download_data.py")
    frame = pd.read_csv(path)
    validate_electricity(frame)
    return frame


def features_target(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    validate_electricity(frame)
    x = frame.loc[:, FEATURES].to_numpy(dtype=float)
    y = frame[TARGET].astype(str).str.upper().eq("UP").to_numpy(dtype=int)
    return x, y
