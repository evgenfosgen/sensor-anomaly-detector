"""Загрузка сохранённых моделей и расчёт их отклонений."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


@dataclass
class LinearModel:
    numeric_features: list[str]
    means: np.ndarray
    stds: np.ndarray
    mode_categories: list[int]
    coefficients: np.ndarray

    # собирает признаки и считает три границы
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        parts = [np.ones((len(frame), 1), dtype=np.float64)]
        if self.numeric_features:
            numeric = frame[self.numeric_features].to_numpy(dtype=np.float64)
            parts.append((numeric - self.means) / self.stds)
        modes = frame["X01"].to_numpy(dtype=np.int64)
        parts.extend(
            (modes == category).astype(np.float64)[:, None]
            for category in self.mode_categories[1:]
        )
        return np.sort(np.column_stack(parts) @ self.coefficients, axis=1)


# читает линейную или CatBoost-модель из JSON
def _load_model(path: Path, model_type: str) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if model_type == "linear_quantile":
        encoder = payload["encoder"]
        return LinearModel(
            numeric_features=list(encoder["numeric_features"]),
            means=np.array(
                [encoder["numeric_means"][name] for name in encoder["numeric_features"]],
                dtype=np.float64,
            ),
            stds=np.array(
                [encoder["numeric_stds"][name] for name in encoder["numeric_features"]],
                dtype=np.float64,
            ),
            mode_categories=[int(value) for value in encoder["mode_categories"]],
            coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        )
    if model_type == "catboost_multi_quantile":
        model = CatBoostRegressor()
        model.load_model(str(path), format="json")
        return model
    raise KeyError(model_type)


# расширяет границы после калибровки
def apply_calibration(
    predictions: np.ndarray,
    modes: np.ndarray,
    calibration: dict[str, Any],
) -> np.ndarray:
    """Расширяет сохранённые границы симметричной conformal-калибровкой."""

    result = np.asarray(predictions, dtype=np.float64).copy()
    global_expansion = float(calibration["global_expansion"])
    if calibration["method"] == "global_symmetric_conformal":
        expansions = np.full(len(result), global_expansion)
    elif calibration["method"] == "X01_mondrian":
        group_expansions = {
            str(key): float(value)
            for key, value in calibration.get("group_expansions", {}).items()
        }
        expansions = np.array(
            [group_expansions.get(str(int(mode)), global_expansion) for mode in modes],
            dtype=np.float64,
        )
    else:
        raise KeyError(calibration["method"])
    result[:, 0] -= expansions
    result[:, 2] += expansions
    return result


@dataclass
class BranchPrediction:
    predictions: np.ndarray
    feature_valid: np.ndarray

    @property
    def lower(self) -> np.ndarray:
        return self.predictions[:, 0]

    @property
    def center(self) -> np.ndarray:
        return self.predictions[:, 1]

    @property
    def upper(self) -> np.ndarray:
        return self.predictions[:, 2]


@dataclass
class LoadedBranch:
    features: list[str]
    quantile_levels: tuple[float, float, float]
    calibration: dict[str, Any]
    model: Any

    # предсказывает только по строкам с валидными признаками
    def predict(self, frame: pd.DataFrame) -> BranchPrediction:
        valid = np.isfinite(
            frame[self.features].to_numpy(dtype=np.float64)
        ).all(axis=1)
        predictions = np.full((len(frame), 3), np.nan, dtype=np.float64)
        if valid.any():
            raw = np.asarray(
                self.model.predict(frame.loc[valid, self.features]),
                dtype=np.float64,
            )
            if raw.ndim == 1:
                raw = raw.reshape(-1, 3)
            predictions[valid] = np.sort(raw, axis=1)
        modes = frame["X01"].to_numpy(dtype=np.int64)
        return BranchPrediction(
            apply_calibration(predictions, modes, self.calibration),
            valid,
        )


@dataclass
class ConsensusPrediction:
    strict: BranchPrediction
    full: BranchPrediction
    detector: Any


@dataclass
class RuntimeBundle:
    branches: dict[str, dict[str, LoadedBranch]]

    @property
    def targets(self) -> list[str]:
        return list(self.branches)

    @property
    def model_count(self) -> int:
        return sum(len(branches) for branches in self.branches.values())

    # оставляет отклонение только при согласии двух веток
    def detect(self, frame: pd.DataFrame, target: str) -> ConsensusPrediction:
        strict = self.branches[target]["strict_robust"].predict(frame)
        full = self.branches[target]["full_telemetry"].predict(frame)
        from .detector import detect_consensus

        detector = detect_consensus(
            frame[target].to_numpy(dtype=np.float64),
            strict.predictions,
            full.predictions,
            strict.feature_valid,
            full.feature_valid,
        )
        return ConsensusPrediction(strict, full, detector)


# загружает пару веток для каждого канала
def load_model_bundle(model_dir: Path | str) -> RuntimeBundle:
    """Загружает модели по manifest."""

    model_dir = Path(model_dir)
    manifest = json.loads(
        (model_dir / "model_info.json").read_text(encoding="utf-8")
    )
    branches: dict[str, dict[str, LoadedBranch]] = {}
    for target in manifest["targets"]:
        branches[target] = {}
        for branch in manifest["branches"]:
            record = manifest["models"][target][branch]
            path = model_dir / record["model_path"]
            branches[target][branch] = LoadedBranch(
                features=list(record["features"]),
                quantile_levels=tuple(record["quantile_levels"]),
                calibration=record["calibration"],
                model=_load_model(path, record["model_type"]),
            )
    return RuntimeBundle(branches)
