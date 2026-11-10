"""Measure changes in feature distributions against the initial training period."""

import numpy as np


def quantile_bins(reference: np.ndarray, count: int = 10) -> np.ndarray:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, count + 1)))
    if len(edges) < 2:
        center = float(edges[0])
        tolerance = max(abs(center) * 1e-7, 1e-9)
        return np.array([-np.inf, center - tolerance, center + tolerance, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def population_stability_index(reference: np.ndarray, observed: np.ndarray) -> float:
    """PSI is descriptive evidence of feature shift, not proof of concept drift."""
    edges = quantile_bins(np.asarray(reference, dtype=float))
    a = np.histogram(reference, bins=edges)[0].astype(float)
    b = np.histogram(observed, bins=edges)[0].astype(float)
    # Add a small pseudocount to prevent undefined log(0).
    a = (a + 0.5) / (a.sum() + 0.5 * len(a))
    b = (b + 0.5) / (b.sum() + 0.5 * len(b))
    return float(np.sum((b - a) * np.log(b / a)))
