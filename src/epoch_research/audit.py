"""Summarize known calendar anomalies without changing benchmark observations."""

import numpy as np
import pandas as pd

from .data import count_date_decreases, validate_electricity


def audit_electricity(frame: pd.DataFrame, duplicate_start: int = 17376, day_rows: int = 48) -> dict:
    """Compare a suspected repeated day across every available column."""
    validate_electricity(frame)
    dates = pd.to_numeric(frame["date"], errors="raise") if "date" in frame else None
    decreases = (
        np.flatnonzero(dates.diff().lt(-1e-8).to_numpy()).tolist()
        if dates is not None else []
    )
    result = {
        "rows": int(len(frame)),
        "date_decreases": int(count_date_decreases(frame)),
        "date_decrease_rows": [int(i) for i in decreases],
        "original_row_order_preserved": True,
        "suspected_duplicate_start": duplicate_start,
        "suspected_duplicate_length": day_rows,
    }
    end = duplicate_start + day_rows * 2
    if duplicate_start < 0 or day_rows < 1 or end > len(frame):
        result["suspected_duplicate_examined"] = False
        return result

    first = frame.iloc[duplicate_start:duplicate_start + day_rows]
    second = frame.iloc[duplicate_start + day_rows:end]
    equal_by_column = {}
    for column in frame.columns:
        a = first[column].to_numpy()
        b = second[column].to_numpy()
        if pd.api.types.is_numeric_dtype(frame[column]):
            equal = np.isclose(a.astype(float), b.astype(float), rtol=0, atol=1e-9, equal_nan=True)
        else:
            equal = a.astype(str) == b.astype(str)
        equal_by_column[column] = int(np.count_nonzero(equal))

    result.update({
        "suspected_duplicate_examined": True,
        "matching_values_per_column": equal_by_column,
        "matching_entire_rows": int(sum(
            all(
                np.isclose(float(first[column].iloc[i]), float(second[column].iloc[i]), rtol=0, atol=1e-9)
                if pd.api.types.is_numeric_dtype(frame[column])
                else str(first[column].iloc[i]) == str(second[column].iloc[i])
                for column in frame.columns
            ) for i in range(day_rows)
        )),
        "removal_performed": False,
    })
    return result
