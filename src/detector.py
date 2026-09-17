"""Решение о выходе за диапазон и подтверждение отклонения."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


PERSISTENCE_SECONDS = 5


@dataclass(frozen=True)
class DetectorSeries:
    outlier_second: np.ndarray
    suspicious: np.ndarray
    target_nan_guardrail: np.ndarray
    input_nan_guardrail: np.ndarray
    system_alarm: np.ndarray


# подтверждает отклонение только при согласии двух веток
def detect_consensus(
    target_values: np.ndarray,
    strict_predictions: np.ndarray,
    full_predictions: np.ndarray,
    strict_valid: np.ndarray,
    full_valid: np.ndarray,
) -> DetectorSeries:
    target = np.asarray(target_values, dtype=np.float64)
    strict = np.asarray(strict_predictions, dtype=np.float64)
    full = np.asarray(full_predictions, dtype=np.float64)
    strict_valid = np.asarray(strict_valid, dtype=bool)
    full_valid = np.asarray(full_valid, dtype=bool)

    target_nan = ~np.isfinite(target)
    input_invalid = (~strict_valid | ~full_valid) & ~target_nan
    bounds_valid = np.isfinite(strict).all(axis=1) & np.isfinite(full).all(axis=1)
    usable = ~target_nan & ~input_invalid & bounds_valid
    strict_low = usable & (target < strict[:, 0])
    strict_high = usable & (target > strict[:, 2])
    full_low = usable & (target < full[:, 0])
    full_high = usable & (target > full[:, 2])
    outlier = (strict_low & full_low) | (strict_high & full_high)

    suspicious = np.zeros(len(target), dtype=bool)
    system_alarm = np.zeros(len(target), dtype=bool)
    streak = 0
    for index in range(len(target)):
        if target_nan[index]:
            system_alarm[index] = True
            streak = 0
        elif input_invalid[index] or not bounds_valid[index]:
            streak = 0
        elif outlier[index]:
            streak += 1
        else:
            streak = 0
        if streak >= PERSISTENCE_SECONDS:
            suspicious[index] = True
            system_alarm[index] = True

    return DetectorSeries(
        outlier_second=outlier,
        suspicious=suspicious,
        target_nan_guardrail=target_nan,
        input_nan_guardrail=input_invalid,
        system_alarm=system_alarm,
    )


# собирает общий диапазон из двух веток
def final_interval(prediction: Any) -> np.ndarray:
    lower = np.minimum(prediction.strict.lower, prediction.full.lower)
    center = (prediction.strict.center + prediction.full.center) / 2.0
    upper = np.maximum(prediction.strict.upper, prediction.full.upper)
    return np.column_stack((lower, center, upper))


def run_inference(
    bundle: Any,
    frame: pd.DataFrame,
    targets: list[str] | None = None,
) -> dict[str, Any]:
    target_names = list(targets or bundle.targets)
    return {target: bundle.detect(frame, target) for target in target_names}
