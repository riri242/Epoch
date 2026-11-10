"""Test-then-train comparison of four adaptation strategies on ordered data."""

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .data import FEATURES, features_target
from .drift import population_stability_index


@dataclass(frozen=True)
class Settings:
    warmup_fraction: float = 0.30
    batch_size: int = 256
    retrain_every_batches: int = 8
    recent_window: int = 6000
    minimum_retrain_samples: int = 1000
    alarm_cooldown_batches: int = 4
    adwin_delta: float = 0.002
    seed: int = 42

    def check(self) -> None:
        if not 0.1 <= self.warmup_fraction <= 0.8:
            raise ValueError("warmup_fraction must be between 0.1 and 0.8")
        if min(self.batch_size, self.retrain_every_batches,
               self.recent_window, self.minimum_retrain_samples,
               self.alarm_cooldown_batches) < 1:
            raise ValueError("Settings require positive integer sizes")
        if not 0 < self.adwin_delta < 1:
            raise ValueError("adwin_delta must be between 0 and 1")


def classifier(seed: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss", alpha=0.0001, max_iter=800,
        tol=1e-3, random_state=seed,
    )


def new_detector(delta):
    from river.drift import ADWIN
    return ADWIN(delta=delta)


def metrics(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_up": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def run(frame: pd.DataFrame, settings: Settings = Settings(), detector_factory=None):
    """All four methods predict each block before observing that block's labels.

    The scaler is fitted on the warm-up period only and never refitted.
    Adaptation decisions and model updates occur *after* block scoring.
    """
    settings.check()
    x, y = features_target(frame)
    warmup = int(len(x) * settings.warmup_fraction)
    if warmup < 20 or len(x) - warmup < settings.batch_size:
        raise ValueError("Not enough observations for chronological evaluation")
    scaler = StandardScaler().fit(x[:warmup])
    x = scaler.transform(x)
    names = ("static", "periodic", "online", "adwin")
    models = {}
    fitting_seconds = {}
    update_counts = dict.fromkeys(names, 0)
    for name in names:
        model = classifier(settings.seed)
        start = perf_counter()
        model.fit(x[:warmup], y[:warmup])
        fitting_seconds[name] = perf_counter() - start
        models[name] = model
    if detector_factory is None:
        detector_factory = lambda: new_detector(settings.adwin_delta)
    detector = detector_factory()
    last_adaptation = -settings.alarm_cooldown_batches
    alarm_count = 0
    truth = []
    predictions = {name: [] for name in names}
    blocks = []
    shifts = []

    for block_idx, left in enumerate(range(warmup, len(x), settings.batch_size)):
        right = min(left + settings.batch_size, len(x))
        batch_x, batch_y = x[left:right], y[left:right]
        predicted = {name: model.predict(batch_x) for name, model in models.items()}
        truth.extend(batch_y.tolist())
        for name in names:
            predictions[name].extend(predicted[name].tolist())
            values = metrics(batch_y, predicted[name])
            blocks.append({
                "strategy": name, "block": block_idx, "start_row": left,
                "end_row": right, "count": right - left, **values,
            })
        for col, feature in enumerate(FEATURES):
            shifts.append({
                "block": block_idx, "end_row": right, "feature": feature,
                "psi": population_stability_index(x[:warmup, col], batch_x[:, col]),
            })

        # The detector observes only errors on already-scored data.
        alarm = False
        for error in (predicted["adwin"] != batch_y).astype(int):
            detector.update(int(error))
            alarm = alarm or bool(detector.drift_detected)

        # Labels become available to all adaptive methods after prediction.
        start = perf_counter()
        models["online"].partial_fit(batch_x, batch_y)
        fitting_seconds["online"] += perf_counter() - start
        update_counts["online"] += 1

        if (block_idx + 1) % settings.retrain_every_batches == 0:
            start = perf_counter()
            recent_start = max(0, right - settings.recent_window)
            models["periodic"] = classifier(settings.seed)
            models["periodic"].fit(x[recent_start:right], y[recent_start:right])
            fitting_seconds["periodic"] += perf_counter() - start
            update_counts["periodic"] += 1

        if alarm:
            alarm_count += 1
        if alarm and block_idx - last_adaptation >= settings.alarm_cooldown_batches:
            recent_start = max(0, right - settings.recent_window)
            if right - recent_start >= settings.minimum_retrain_samples:
                start = perf_counter()
                models["adwin"] = classifier(settings.seed)
                models["adwin"].fit(x[recent_start:right], y[recent_start:right])
                fitting_seconds["adwin"] += perf_counter() - start
                update_counts["adwin"] += 1
                last_adaptation = block_idx
                detector = detector_factory()

    summary = {
        "dataset_rows": len(x),
        "initial_training_rows": warmup,
        "evaluated_rows": len(truth),
        "features": list(FEATURES),
        "excluded_price_features": ["nswprice", "vicprice"],
        "settings": asdict(settings),
        "strategies": {
            name: {
                **metrics(truth, predictions[name]),
                "model_updates": update_counts[name],
                "total_fitting_seconds": round(fitting_seconds[name], 4),
            }
            for name in names
        },
        "adwin_alarm_blocks": alarm_count,
        "protocol": "chronological 30% warm-up; predict block then reveal labels and adapt",
        "production_ready": False,
    }
    return summary, pd.DataFrame(blocks), pd.DataFrame(shifts)
