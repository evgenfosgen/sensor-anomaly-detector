"""Компактный генератор обезличенной телеметрии."""

from __future__ import annotations

import copy
from typing import Mapping

import numpy as np
import pandas as pd


SECONDS_PER_DAY = 86_400
OBSERVED_COLUMNS = ["timestamp", *[f"X{index:02d}" for index in range(1, 13)]]

SIMULATOR_CONFIG = {
    "dataset": {
        "start_timestamp": "2025-01-01T00:00:00Z",
        "sampling_period_seconds": 1,
    },
    "analog_channels": [f"X{index:02d}" for index in range(4, 13)],
    "modes": [
        {"id": 0, "name": "M01", "X02": 42.0, "X03": 42.0},
        {"id": 1, "name": "M02", "X02": 68.0, "X03": 55.0},
        {"id": 2, "name": "M03", "X02": 88.0, "X03": 72.0},
    ],
    "mode_schedule": {
        "cycle": [1, 0, 1, 2, 1],
        "dwell_seconds_min": 1800,
        "dwell_seconds_max": 10800,
        "transition_seconds": 120,
        "transition_response_fraction": 0.95,
    },
    "controls": {
        "ar_rho": 0.995,
        "sigma_X02": 2.2,
        "sigma_X03": 3.2,
    },
    "load": {
        "base": 0.02,
        "gain_X02": 0.78,
        "reference_X02": 35.0,
        "scale_X02": 55.0,
        "gain_X03": 0.18,
        "reference_X03": 40.0,
        "scale_X03": 40.0,
        "ar_rho": 0.96,
        "disturbance_sigma": 0.012,
        "min": 0.08,
        "max": 1.0,
    },
    "context": {
        "A": {
            "mean": 17.0,
            "seasonal_amplitude": 2.5,
            "seasonal_period_hours": 24.0,
            "mean_reversion_hours": 8.0,
            "stationary_sigma": 0.65,
        },
        "B": {
            "mean": 47.0,
            "seasonal_amplitude": 2.0,
            "seasonal_period_hours": 36.0,
            "mean_reversion_hours": 12.0,
            "stationary_sigma": 0.6,
        },
    },
    "states": {
        "H01": {
            "initial": 0.08,
            "drift": 0.012,
            "mean_reversion_hours": 12.0,
            "stationary_sigma": 0.008,
            "min": 0.0,
            "max": 0.65,
        },
        "H02": {
            "initial": 0.985,
            "drift": 0.001,
            "mean_reversion_hours": 72.0,
            "stationary_sigma": 0.0015,
            "min": 0.85,
            "max": 1.0,
        },
        "H03": {
            "initial": 0.98,
            "drift": -0.0005,
            "mean_reversion_hours": 48.0,
            "stationary_sigma": 0.018,
            "min": 0.88,
            "max": 1.03,
        },
        "H04": {
            "initial": 0.35,
            "drift": 0.0,
            "mean_reversion_hours": 8.0,
            "stationary_sigma": 0.1,
            "min": 0.05,
            "max": 0.9,
        },
    },
    "equations": {
        "X11": {
            "base": 7.0,
            "load_gain": 21.0,
            "H01_gain": -3.0,
            "A_gain": 0.08,
            "A_reference": 17.0,
            "B_gain": -0.03,
            "B_reference": 47.0,
            "X03_gain": 0.025,
            "X03_reference": 55.0,
            "H04_gain": -2.8,
            "H04_reference": 0.35,
        },
        "X06": {
            "base": 2.1,
            "load_gain": 2.5,
            "X11_gain": 0.025,
            "X03_gain": 0.012,
            "X03_reference": 50.0,
            "A_gain": 0.006,
            "A_reference": 17.0,
            "H04_gain": 0.45,
            "H04_reference": 0.35,
        },
        "X07": {
            "margin": 0.25,
            "X11_gain": 0.018,
            "load_gain": 0.05,
        },
        "X09": {
            "base": 0.12,
            "X11_gain": 0.028,
            "X11_squared_gain": 0.0004,
            "H01_gain": 0.65,
            "load_gain": 0.04,
        },
        "X10": {
            "base": 3.2,
            "load_gain": 3.7,
            "X11_gain": 0.025,
            "H01_gain": 0.5,
            "B_gain": 0.012,
            "B_reference": 47.0,
            "H04_gain": 0.25,
            "H04_reference": 0.35,
        },
        "X12": {
            "base": 22.0,
            "B_gain": 3.4,
            "H02_gain": 300.0,
            "H01_gain": 55.0,
            "A_gain": 1.0,
            "A_reference": 17.0,
            "X10_gain": -0.4,
            "X10_reference": 5.5,
        },
    },
    "noise": {
        "ar_rho": 0.96,
        "mode_multipliers": {"M01": 0.8, "M02": 1.0, "M03": 1.25},
        "transition_multiplier": 1.35,
        "channel_sigma": {
            "X06": 0.035,
            "X07": 0.035,
            "X08": 0.035,
            "X09": 0.018,
            "X10": 0.045,
            "X11": 0.16,
            "X12": 1.8,
        },
        "measurement": {
            "X04": 0.04,
            "X05": 0.04,
            "X06": 0.018,
            "X07": 0.018,
            "X08": 0.018,
            "X09": 0.012,
            "X10": 0.025,
            "X11": 0.08,
            "X12": 1.2,
        },
    },
    "ranges": {
        "X02": {"min": 0.0, "max": 100.0},
        "X03": {"min": 0.0, "max": 100.0},
    },
}


# создаёт коррелированный шум для ряда
def _ar_noise(
    length: int,
    rho: float,
    sigma: float,
    rng: np.random.Generator,
    scale: np.ndarray | None = None,
) -> np.ndarray:
    if scale is None:
        scale = np.ones(length, dtype=np.float64)
    innovations = rng.standard_normal(length)
    result = np.empty(length, dtype=np.float64)
    innovation_sigma = sigma * np.sqrt(1.0 - rho * rho)
    result[0] = sigma * scale[0] * innovations[0]
    for index in range(1, length):
        result[index] = (
            rho * result[index - 1]
            + innovation_sigma * scale[index] * innovations[index]
        )
    return result


# тянет ряд к меняющейся цели с шумом
def _mean_reverting_series(
    target: np.ndarray,
    initial: float,
    mean_reversion_seconds: float,
    stationary_sigma: float,
    step_seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(target.size, dtype=np.float64)
    rho = np.exp(-step_seconds / mean_reversion_seconds)
    innovation_sigma = stationary_sigma * np.sqrt(1.0 - rho * rho)
    result[0] = initial
    for index in range(1, target.size):
        result[index] = (
            target[index]
            + rho * (result[index - 1] - target[index])
            + innovation_sigma * rng.standard_normal()
        )
    return result


# тянет скрытый ряд к цели и удерживает его в границах
def _bounded_mean_reverting_series(
    target: np.ndarray,
    initial: float,
    mean_reversion_seconds: float,
    stationary_sigma: float,
    minimum: float,
    maximum: float,
    step_seconds: float,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.empty(target.size, dtype=np.float64)
    rho = np.exp(-step_seconds / mean_reversion_seconds)
    innovation_sigma = stationary_sigma * np.sqrt(1.0 - rho * rho)
    result[0] = np.clip(initial, minimum, maximum)
    for index in range(1, target.size):
        result[index] = np.clip(
            target[index]
            + rho * (result[index - 1] - target[index])
            + innovation_sigma * rng.standard_normal(),
            minimum,
            maximum,
        )
    return result


# собирает последовательность режимов с заданной длительностью
def _build_mode_targets(
    length: int,
    mode_schedule: Mapping[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    cycle = [int(value) for value in mode_schedule["cycle"]]
    minimum = int(mode_schedule["dwell_seconds_min"])
    maximum = int(mode_schedule["dwell_seconds_max"])
    targets = np.empty(length, dtype=np.int8)
    cursor = 0
    cycle_index = 0
    while cursor < length:
        mode_id = cycle[cycle_index % len(cycle)]
        duration = int(rng.integers(minimum, maximum + 1))
        end = min(length, cursor + duration)
        targets[cursor:end] = mode_id
        cursor = end
        cycle_index += 1
    return targets


def _build_transition_mask(mode_targets: np.ndarray, transition_seconds: int) -> np.ndarray:
    mask = np.zeros(mode_targets.size, dtype=bool)
    changes = np.flatnonzero(mode_targets[1:] != mode_targets[:-1]) + 1
    for start in changes:
        mask[start : min(len(mode_targets), start + transition_seconds)] = True
    return mask


# плавно переводит ряд к новому уровню
def _smooth_follow(
    target: np.ndarray,
    initial: float,
    step_seconds: float,
    transition_seconds: float,
    response_fraction: float,
) -> np.ndarray:
    time_constant = transition_seconds / (-np.log1p(-response_fraction))
    alpha = 1.0 - np.exp(-step_seconds / time_constant)
    result = np.empty(target.size, dtype=np.float64)
    result[0] = initial
    for index in range(1, target.size):
        result[index] = result[index - 1] + alpha * (target[index] - result[index - 1])
    return result


# меняет шум канала для режима и перехода
def _channel_noise(
    channel: str,
    length: int,
    noise_config: Mapping[str, object],
    mode_names: Mapping[int, str],
    mode_targets: np.ndarray,
    transition_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    multipliers = noise_config["mode_multipliers"]
    scales = np.array(
        [float(multipliers[mode_names[int(mode)]]) for mode in mode_targets],
        dtype=np.float64,
    )
    scales[transition_mask] *= float(noise_config["transition_multiplier"])
    return _ar_noise(
        length,
        rho=float(noise_config["ar_rho"]),
        sigma=float(noise_config["channel_sigma"][channel]),
        rng=rng,
        scale=scales,
    )


def _observed_value(
    channel: str,
    true_value: np.ndarray,
    measurement_config: Mapping[str, object],
    rng: np.random.Generator,
) -> np.ndarray:
    return (
        true_value
        + rng.normal(0.0, float(measurement_config[channel]), true_value.size)
    ).astype(np.float32)


# строит два медленных контекста ряда
def _build_context(
    length: int,
    step_seconds: float,
    config: Mapping[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    elapsed = np.arange(length, dtype=np.float64) * step_seconds

    context_a = config["A"]
    period_a = float(context_a["seasonal_period_hours"]) * 3600.0
    target_a = float(context_a["mean"]) + float(
        context_a["seasonal_amplitude"]
    ) * np.sin(2.0 * np.pi * elapsed / period_a)
    x04 = _mean_reverting_series(
        target_a,
        float(context_a["mean"]),
        float(context_a["mean_reversion_hours"]) * 3600.0,
        float(context_a["stationary_sigma"]),
        step_seconds,
        rng,
    )

    context_b = config["B"]
    period_b = float(context_b["seasonal_period_hours"]) * 3600.0
    target_b = float(context_b["mean"]) + float(
        context_b["seasonal_amplitude"]
    ) * np.cos(2.0 * np.pi * elapsed / period_b)
    x05 = _mean_reverting_series(
        target_b,
        float(context_b["mean"]),
        float(context_b["mean_reversion_hours"]) * 3600.0,
        float(context_b["stationary_sigma"]),
        step_seconds,
        rng,
    )
    return x04, x05


# строит медленные скрытые состояния
def _build_latent(
    length: int,
    step_seconds: float,
    config: Mapping[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    elapsed_days = np.arange(length, dtype=np.float64) * step_seconds / SECONDS_PER_DAY

    h01_config = config["H01"]
    h01 = _bounded_mean_reverting_series(
        float(h01_config["initial"]) + float(h01_config["drift"]) * elapsed_days,
        float(h01_config["initial"]),
        float(h01_config["mean_reversion_hours"]) * 3600.0,
        float(h01_config["stationary_sigma"]),
        float(h01_config["min"]),
        float(h01_config["max"]),
        step_seconds,
        rng,
    )

    h02_config = config["H02"]
    h02 = _bounded_mean_reverting_series(
        float(h02_config["initial"]) - float(h02_config["drift"]) * elapsed_days,
        float(h02_config["initial"]),
        float(h02_config["mean_reversion_hours"]) * 3600.0,
        float(h02_config["stationary_sigma"]),
        float(h02_config["min"]),
        float(h02_config["max"]),
        step_seconds,
        rng,
    )

    h03_config = config["H03"]
    h03 = _bounded_mean_reverting_series(
        float(h03_config["initial"]) + float(h03_config["drift"]) * elapsed_days,
        float(h03_config["initial"]),
        float(h03_config["mean_reversion_hours"]) * 3600.0,
        float(h03_config["stationary_sigma"]),
        float(h03_config["min"]),
        float(h03_config["max"]),
        step_seconds,
        rng,
    )

    h04_config = config["H04"]
    h04 = _bounded_mean_reverting_series(
        float(h04_config["initial"]) + float(h04_config["drift"]) * elapsed_days,
        float(h04_config["initial"]),
        float(h04_config["mean_reversion_hours"]) * 3600.0,
        float(h04_config["stationary_sigma"]),
        float(h04_config["min"]),
        float(h04_config["max"]),
        step_seconds,
        rng,
    )
    return h01, h02, h03, h04


# собирает полный ряд из режимов, состояний и каналов
def _generate(
    config: Mapping[str, object],
    length: int,
    seed: int,
) -> pd.DataFrame:
    dataset = config["dataset"]
    step_seconds = float(dataset["sampling_period_seconds"])
    rng = np.random.default_rng(int(seed))
    timestamps = pd.date_range(
        start=pd.Timestamp(dataset["start_timestamp"]).tz_convert("UTC"),
        periods=length,
        freq=pd.to_timedelta(int(step_seconds), unit="s"),
    )

    mode_schedule = config["mode_schedule"]
    mode_targets = _build_mode_targets(length, mode_schedule, rng)
    transition_mask = _build_transition_mask(
        mode_targets, int(mode_schedule["transition_seconds"])
    )
    mode_definitions = {int(mode["id"]): mode for mode in config["modes"]}
    mode_names = {
        mode_id: str(mode["name"]) for mode_id, mode in mode_definitions.items()
    }

    controls = config["controls"]
    x02_target = np.array(
        [float(mode_definitions[int(mode)]["X02"]) for mode in mode_targets],
        dtype=np.float64,
    )
    x02_target += _ar_noise(
        length,
        float(controls["ar_rho"]),
        float(controls["sigma_X02"]),
        rng,
    )
    x03_target = np.array(
        [float(mode_definitions[int(mode)]["X03"]) for mode in mode_targets],
        dtype=np.float64,
    )
    x03_target += _ar_noise(
        length,
        float(controls["ar_rho"]),
        float(controls["sigma_X03"]),
        rng,
    )
    x02 = np.clip(
        _smooth_follow(
            x02_target,
            x02_target[0],
            step_seconds,
            float(mode_schedule["transition_seconds"]),
            float(mode_schedule["transition_response_fraction"]),
        ),
        float(config["ranges"]["X02"]["min"]),
        float(config["ranges"]["X02"]["max"]),
    )
    x03 = np.clip(
        _smooth_follow(
            x03_target,
            x03_target[0],
            step_seconds,
            float(mode_schedule["transition_seconds"]),
            float(mode_schedule["transition_response_fraction"]),
        ),
        float(config["ranges"]["X03"]["min"]),
        float(config["ranges"]["X03"]["max"]),
    )

    load_config = config["load"]
    load_signal = np.clip(
        float(load_config["base"])
        + float(load_config["gain_X02"])
        * (x02 - float(load_config["reference_X02"]))
        / float(load_config["scale_X02"])
        + float(load_config["gain_X03"])
        * (x03 - float(load_config["reference_X03"]))
        / float(load_config["scale_X03"])
        + _ar_noise(
            length,
            float(load_config["ar_rho"]),
            float(load_config["disturbance_sigma"]),
            rng,
        ),
        float(load_config["min"]),
        float(load_config["max"]),
    )

    x04, x05 = _build_context(length, step_seconds, config["context"], rng)
    h01, h02, h03, h04 = _build_latent(length, step_seconds, config["states"], rng)
    scaled_load = load_signal * h03
    noise_config = config["noise"]
    equations = config["equations"]

    x11_config = equations["X11"]
    x11 = (
        float(x11_config["base"])
        + float(x11_config["load_gain"]) * scaled_load
        + float(x11_config["H01_gain"]) * h01
        + float(x11_config["A_gain"])
        * (x04 - float(x11_config["A_reference"]))
        + float(x11_config["B_gain"])
        * (x05 - float(x11_config["B_reference"]))
        + float(x11_config["X03_gain"])
        * (x03 - float(x11_config["X03_reference"]))
        + float(x11_config["H04_gain"])
        * (h04 - float(x11_config["H04_reference"]))
        + _channel_noise(
            "X11", length, noise_config, mode_names, mode_targets, transition_mask, rng
        )
    )

    x06_config = equations["X06"]
    x06 = (
        float(x06_config["base"])
        + float(x06_config["load_gain"]) * scaled_load
        + float(x06_config["X11_gain"]) * x11
        + float(x06_config["X03_gain"])
        * (x03 - float(x06_config["X03_reference"]))
        + float(x06_config["A_gain"])
        * (x04 - float(x06_config["A_reference"]))
        + float(x06_config["H04_gain"])
        * (h04 - float(x06_config["H04_reference"]))
        + _channel_noise(
            "X06", length, noise_config, mode_names, mode_targets, transition_mask, rng
        )
    )

    x07_config = equations["X07"]
    x07 = (
        x06
        + float(x07_config["margin"])
        + float(x07_config["X11_gain"]) * x11
        + float(x07_config["load_gain"]) * load_signal
        + _channel_noise(
            "X07", length, noise_config, mode_names, mode_targets, transition_mask, rng
        )
    )

    x09_config = equations["X09"]
    x09 = (
        float(x09_config["base"])
        + float(x09_config["X11_gain"]) * x11
        + float(x09_config["X11_squared_gain"]) * np.square(x11)
        + float(x09_config["H01_gain"]) * h01
        + float(x09_config["load_gain"]) * load_signal
        + _channel_noise(
            "X09", length, noise_config, mode_names, mode_targets, transition_mask, rng
        )
    )
    x08 = x07 - x09 + _channel_noise(
        "X08", length, noise_config, mode_names, mode_targets, transition_mask, rng
    )

    x10_config = equations["X10"]
    x10 = (
        float(x10_config["base"])
        + float(x10_config["load_gain"]) * scaled_load
        + float(x10_config["X11_gain"]) * x11
        + float(x10_config["H01_gain"]) * h01
        + float(x10_config["B_gain"])
        * (x05 - float(x10_config["B_reference"]))
        + float(x10_config["H04_gain"])
        * (h04 - float(x10_config["H04_reference"]))
        + _channel_noise(
            "X10", length, noise_config, mode_names, mode_targets, transition_mask, rng
        )
    )

    x12_config = equations["X12"]
    x12 = (
        float(x12_config["base"])
        + float(x12_config["B_gain"]) * x05
        + float(x12_config["H02_gain"]) * (1.0 - h02)
        + float(x12_config["H01_gain"]) * h01
        + float(x12_config["A_gain"])
        * (x04 - float(x12_config["A_reference"]))
        + float(x12_config["X10_gain"])
        * (x10 - float(x12_config["X10_reference"]))
        + _channel_noise(
            "X12", length, noise_config, mode_names, mode_targets, transition_mask, rng
        )
    )

    values = {
        "X04": x04,
        "X05": x05,
        "X06": x06,
        "X07": x07,
        "X08": x08,
        "X09": x09,
        "X10": x10,
        "X11": x11,
        "X12": x12,
    }
    measurement = noise_config["measurement"]
    observed = {
        "timestamp": timestamps,
        "X01": mode_targets.astype(np.int8),
        "X02": x02.astype(np.float32),
        "X03": x03.astype(np.float32),
    }
    for channel in config["analog_channels"]:
        observed[channel] = _observed_value(channel, values[channel], measurement, rng)
    return pd.DataFrame(observed)


def generate_trace(
    duration_seconds: int = 900,
    seed: int = 20260907,
    mode_sequence: tuple[int, ...] = (1, 0, 1, 2, 1),
    dwell_seconds: int = 180,
    start_timestamp: str | None = None,
) -> pd.DataFrame:
    config = copy.deepcopy(SIMULATOR_CONFIG)
    if start_timestamp is not None:
        config["dataset"]["start_timestamp"] = start_timestamp
    config["mode_schedule"] = {
        "cycle": [int(mode) for mode in mode_sequence],
        "dwell_seconds_min": int(dwell_seconds),
        "dwell_seconds_max": int(dwell_seconds),
        "transition_seconds": int(config["mode_schedule"]["transition_seconds"]),
        "transition_response_fraction": float(
            config["mode_schedule"]["transition_response_fraction"]
        ),
    }
    return _generate(config, int(duration_seconds), int(seed))


# продолжает ряд после смены режима
def generate_transition(
    start_row: pd.Series,
    from_mode: int,
    to_mode: int,
    duration_seconds: int,
    seed: int,
) -> pd.DataFrame:
    config = copy.deepcopy(SIMULATOR_CONFIG)
    timestamp = pd.Timestamp(start_row["timestamp"])
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    config["dataset"]["start_timestamp"] = timestamp.isoformat()
    config["mode_schedule"] = {
        "cycle": [int(from_mode)] + [int(to_mode)] * max(0, int(duration_seconds) - 1),
        "dwell_seconds_min": 1,
        "dwell_seconds_max": 1,
        "transition_seconds": int(config["mode_schedule"]["transition_seconds"]),
        "transition_response_fraction": float(
            config["mode_schedule"]["transition_response_fraction"]
        ),
    }
    for mode in config["modes"]:
        if int(mode["id"]) == int(from_mode):
            mode["X02"] = float(start_row["X02"])
            mode["X03"] = float(start_row["X03"])
    config["context"]["A"]["mean"] = float(start_row["X04"])
    config["context"]["B"]["mean"] = float(start_row["X05"])

    length = int(duration_seconds)
    initial = _generate(config, length, int(seed))
    source_mode = next(
        mode for mode in config["modes"] if int(mode["id"]) == int(from_mode)
    )
    source_mode["X02"] += float(start_row["X02"]) - float(initial["X02"].iloc[0])
    source_mode["X03"] += float(start_row["X03"]) - float(initial["X03"].iloc[0])
    result = _generate(config, length, int(seed))
    result["timestamp"] = pd.date_range(
        start=timestamp,
        periods=length,
        freq="1s",
    )
    for column in OBSERVED_COLUMNS:
        result.at[0, column] = start_row[column]
    return result


__all__ = ["OBSERVED_COLUMNS", "generate_trace", "generate_transition"]
