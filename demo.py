"""Локальная демонстрация детектора на синтетическом временном ряде."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse

from src.detector import PERSISTENCE_SECONDS
from src.models import load_model_bundle
from src.simulator import generate_transition


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "assets" / "index.html"
TARGETS = ["X06", "X07", "X08", "X09", "X10", "X11"]
MODE_CODES = {1: 0, 2: 1, 3: 2}
DEMO_SEED = 20260913


def _load_config(root: Path) -> dict[str, Any]:
    return json.loads((root / "config.json").read_text(encoding="utf-8"))


def _load_trace(root: Path) -> pd.DataFrame:
    dtypes = {f"X{index:02d}": "float32" for index in range(1, 13)}
    dtypes["X01"] = "int8"
    return pd.read_csv(
        root / "data" / "demo.csv",
        parse_dates=["timestamp"],
        dtype=dtypes,
    )


def _json_values(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


# считает длину текущей серии отклонений
def _trailing_streak(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.int16)
    streak = 0
    for index, value in enumerate(np.asarray(values, dtype=bool)):
        streak = min(PERSISTENCE_SECONDS, streak + 1) if value else 0
        result[index] = streak
    return result


# переводит флаги detector в состояния интерфейса
def _status_values(detector: Any) -> np.ndarray:
    statuses = np.full(len(detector.outlier_second), "NORMAL", dtype=object)
    guardrail = detector.target_nan_guardrail | detector.input_nan_guardrail
    statuses[detector.outlier_second] = "OUTSIDE"
    statuses[detector.suspicious] = "SUSPICIOUS"
    statuses[guardrail] = "GUARDRAIL"
    return statuses


class DemoEngine:
    """Одна независимая сессия демонстрации."""

    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        self.demo_config = _load_config(self.root)
        self.bundle = load_model_bundle(self.root / "models")
        self.base_frame = _load_trace(self.root)
        self.clean_frame = self.base_frame.copy(deep=True)
        self.position = 0.0
        self.playing = False
        self.speed = int(self.demo_config["default_speed"])
        self.requested_mode = int(self.clean_frame["X01"].iloc[0]) + 1
        self.manual_fault: dict[str, Any] | None = None
        self.transition_from: int | None = None
        self.transition_to: int | None = None
        self.transition_start: int | None = None
        self.transition_end: int | None = None
        self.transition_count = 0
        self.last_wall_time: float | None = None
        self.action_message = f"Режим {self.requested_mode}"
        self.logged_detection_targets: set[str] = set()
        self.generation = 0
        self.events: list[str] = []
        self.series: dict[str, dict[str, Any]] = {}
        self._reset_events()
        self._recompute()

    @property
    def duration_seconds(self) -> int:
        return len(self.clean_frame)

    @property
    def current_index(self) -> int:
        return min(self.duration_seconds - 1, max(0, int(self.position)))

    @property
    def current_mode_code(self) -> int:
        return int(self.clean_frame["X01"].iloc[self.current_index])

    @property
    def current_mode(self) -> int:
        return self.current_mode_code + 1

    @property
    def transition_active(self) -> bool:
        if self.transition_start is None or self.transition_end is None:
            return False
        return self.transition_start - 1 <= self.current_index <= self.transition_end

    @property
    def finished(self) -> bool:
        return self.current_index >= self.duration_seconds - 1

    def _reset_events(self) -> None:
        self.events = [f"Режим {self.requested_mode}"]
        self.action_message = f"Режим {self.requested_mode}"

    def _add_event(self, message: str, index: int | None = None) -> None:
        row_index = self.current_index if index is None else int(index)
        timestamp = pd.Timestamp(self.clean_frame["timestamp"].iloc[row_index])
        self.events.append(f"{timestamp.strftime('%H:%M:%S')} - {message}")
        self.events = self.events[-20:]

    # запоминает первую тревогу по каждому каналу
    def _record_current_detections(self) -> None:
        index = self.current_index
        for target in TARGETS:
            status = self.series[target]["status"][index]
            if status not in {"SUSPICIOUS", "GUARDRAIL"}:
                continue
            if target in self.logged_detection_targets:
                continue
            self.logged_detection_targets.add(target)
            self._add_event(f"Обнаружено отклонение: {target}", index=index)

    # закрывает переход после последней секунды нового ряда
    def _finish_transition_if_needed(self) -> None:
        if self.transition_start is None or self.transition_active:
            return
        if self.action_message.startswith("Переход:"):
            self.action_message = f"Режим {self.requested_mode}"
        self.transition_start = None
        self.transition_end = None
        self.transition_from = None
        self.transition_to = None

    # добавляет текущую неисправность к копии чистого ряда
    def _materialize_frame(self) -> pd.DataFrame:
        frame = self.clean_frame.copy(deep=True)
        if self.manual_fault is not None:
            start = self.manual_fault["start_index"]
            target = self.manual_fault["target"]
            values = frame.loc[start:, target].to_numpy(dtype=np.float64)
            frame.loc[start:, target] = (
                values + float(self.manual_fault["offset"])
            ).astype(np.float32)
        return frame

    # пересчитывает интервалы и состояния detector для всего ряда
    def _recompute(self) -> None:
        frame = self._materialize_frame()
        computed: dict[str, dict[str, Any]] = {}
        for target in TARGETS:
            prediction = self.bundle.detect(frame, target)
            detector = prediction.detector
            lower = np.minimum(prediction.strict.lower, prediction.full.lower)
            upper = np.maximum(prediction.strict.upper, prediction.full.upper)
            guardrail = detector.target_nan_guardrail | detector.input_nan_guardrail
            statuses = _status_values(detector)
            computed[target] = {
                "values": _json_values(frame[target].to_numpy(dtype=np.float64)),
                "lower": _json_values(lower),
                "upper": _json_values(upper),
                "outlier": detector.outlier_second.tolist(),
                "suspicious": detector.suspicious.tolist(),
                "guardrail": guardrail.tolist(),
                "persistence": _trailing_streak(detector.outlier_second).tolist(),
                "status": [str(value) for value in statuses],
            }
        self.series = computed
        self.generation += 1

    def update_clock(self, now: float | None = None) -> None:
        if not self.playing:
            self._finish_transition_if_needed()
            return
        current_time = time.monotonic() if now is None else float(now)
        if self.last_wall_time is None:
            self.last_wall_time = current_time
            return
        elapsed = max(0.0, current_time - self.last_wall_time)
        self.last_wall_time = current_time
        self.position += elapsed * self.speed
        if self.position >= self.duration_seconds - 1:
            self.position = float(self.duration_seconds - 1)
            self.playing = False
            self.last_wall_time = None
            self._add_event("Воспроизведение завершено")
        self._finish_transition_if_needed()

    def play(self) -> None:
        if self.finished:
            self.position = 0.0
            self.manual_fault = None
            self.logged_detection_targets.clear()
            self._recompute()
            self._add_event("Повторный проход начинается с чистой телеметрии")
        self.playing = True
        self.last_wall_time = time.monotonic()

    def pause(self) -> None:
        self.update_clock()
        self.playing = False
        self.last_wall_time = None

    def set_speed(self, speed: int) -> None:
        self.update_clock()
        self.speed = int(speed)
        if self.playing:
            self.last_wall_time = time.monotonic()

    def reset(self) -> None:
        was_playing = self.playing
        self.clean_frame = self.base_frame.copy(deep=True)
        self.position = 0.0
        self.playing = False
        self.speed = int(self.demo_config["default_speed"])
        self.requested_mode = int(self.clean_frame["X01"].iloc[0]) + 1
        self.manual_fault = None
        self.transition_from = None
        self.transition_to = None
        self.transition_start = None
        self.transition_end = None
        self.transition_count = 0
        self.last_wall_time = None
        self.logged_detection_targets.clear()
        self._reset_events()
        self._recompute()
        if was_playing:
            self.play()

    # перестраивает продолжение ряда после смены режима
    def set_mode(self, mode: int) -> None:
        mode = int(mode)
        self.update_clock()
        if mode == self.requested_mode and not self.transition_active:
            return
        was_playing = self.playing
        current_index = self.current_index
        if current_index >= self.duration_seconds - 1:
            self.clean_frame = self.base_frame.copy(deep=True)
            self.position = 0.0
            current_index = 0
        from_mode = self.current_mode_code
        remaining = self.duration_seconds - current_index
        if remaining > 1:
            transition_seed = (
                DEMO_SEED
                + 1_000_003 * (self.transition_count + 1)
                + 10_007 * (from_mode + 1)
                + 100_003 * (MODE_CODES[mode] + 1)
                + current_index
            )
            continuation = generate_transition(
                self.clean_frame.iloc[current_index],
                from_mode,
                MODE_CODES[mode],
                remaining,
                transition_seed,
            )
            prefix = self.clean_frame.iloc[: current_index + 1]
            self.clean_frame = pd.concat(
                [prefix, continuation.iloc[1:]],
                ignore_index=True,
            )
            transition_seconds = int(
                self.demo_config["scenario"]["transition_seconds"]
            )
            self.transition_start = current_index + 1
            self.transition_end = min(
                self.duration_seconds - 1,
                current_index + transition_seconds,
            )
            self.transition_from = from_mode + 1
            self.transition_to = mode
            self.transition_count += 1
        self.requested_mode = mode
        self.manual_fault = None
        self.logged_detection_targets.clear()
        self._recompute()
        self.action_message = (
            f"Переход: режим {from_mode + 1} -> режим {mode}"
            if self.transition_start is not None
            else f"Режим {mode}"
        )
        self._add_event(f"Переход: режим {from_mode + 1} -> режим {mode}")
        self.playing = was_playing
        self.last_wall_time = time.monotonic() if was_playing else None

    def inject_fault(self, target: str) -> None:
        was_playing = self.playing
        self.update_clock()
        self.manual_fault = {
            "target": target,
            "type": "bias",
            "start_index": self.current_index,
            "offset": float(self.demo_config["bias_offsets"][target]),
        }
        self.logged_detection_targets.clear()
        self.action_message = f"Инъекция смещения: {target}"
        self._recompute()
        self._add_event(f"Инъекция смещения: {target}")
        if was_playing and self.playing:
            self.last_wall_time = time.monotonic()

    def clear_fault(self) -> None:
        if self.manual_fault is None:
            return
        was_playing = self.playing
        self.update_clock()
        target = self.manual_fault["target"]
        self.manual_fault = None
        self.logged_detection_targets.discard(target)
        self.action_message = f"Смещение {target} сброшено"
        self._recompute()
        self._add_event(f"Смещение {target} сброшено")
        if was_playing and self.playing:
            self.last_wall_time = time.monotonic()

    def _detector_state(self) -> str:
        index = self.current_index
        if self.manual_fault is not None:
            target = self.manual_fault["target"]
            if self.series[target]["status"][index] in {"SUSPICIOUS", "GUARDRAIL"}:
                return f"Обнаружено отклонение: {target}"
            return f"Ожидание подтверждения отклонения: {target}"
        guardrails = [
            target for target in TARGETS if self.series[target]["guardrail"][index]
        ]
        if guardrails:
            return f"Обнаружено отклонение: {guardrails[0]}"
        suspicious = [
            target for target in TARGETS if self.series[target]["suspicious"][index]
        ]
        if suspicious:
            return f"Обнаружено отклонение: {suspicious[0]}"
        outside = [
            target for target in TARGETS if self.series[target]["outlier"][index]
        ]
        if outside:
            return f"Отклонение: {outside[0]} - ожидание подтверждения"
        return "Норма"

    def _current_states(self) -> dict[str, dict[str, Any]]:
        index = self.current_index
        return {
            target: {
                "value": self.series[target]["values"][index],
                "lower": self.series[target]["lower"][index],
                "upper": self.series[target]["upper"][index],
                "status": self.series[target]["status"][index],
                "persistence": self.series[target]["persistence"][index],
            }
            for target in TARGETS
        }

    # собирает текущее состояние для интерфейса
    def clock(self) -> dict[str, Any]:
        self.update_clock()
        self._record_current_detections()
        index = self.current_index
        detector_state = self._detector_state()
        timestamp = pd.Timestamp(self.clean_frame["timestamp"].iloc[index])
        fault = None
        if self.manual_fault is not None:
            fault = {
                "target": self.manual_fault["target"],
                "type": self.manual_fault["type"],
                "label": "Смещение",
            }
        return {
            "type": "clock",
            "generation": self.generation,
            "position": float(self.position),
            "index": index,
            "playing": bool(self.playing),
            "finished": self.finished,
            "speed": self.speed,
            "timestamp": timestamp.isoformat(),
            "action": self.action_message,
            "detector_state": detector_state,
            "headline": detector_state,
            "mode": self.current_mode,
            "requested_mode": self.requested_mode,
            "transition_active": self.transition_active,
            "transition_from": self.transition_from,
            "transition_to": self.transition_to,
            "fault": fault,
            "states": self._current_states(),
            "events": list(self.events[-12:]),
        }

    def snapshot(self) -> dict[str, Any]:
        timestamps = pd.to_datetime(self.clean_frame["timestamp"], utc=True)
        timestamp_values = [
            int(value // 1_000_000)
            for value in timestamps.astype("int64").to_numpy(dtype=np.int64)
        ]
        scenario = self.demo_config["scenario"]
        return {
            "type": "snapshot",
            "generation": self.generation,
            "duration_seconds": self.duration_seconds,
            "timestamps": timestamp_values,
            "targets": list(TARGETS),
            "series": self.series,
            "models": {
                "count": self.bundle.model_count,
                "branches": ["strict_robust", "full_telemetry"],
                "persistence_seconds": PERSISTENCE_SECONDS,
                "interval": "итоговый диапазон по двум веткам с одинаковым направлением отклонения",
            },
            "trace": {
                "path": "data/demo.csv",
                "rows": self.duration_seconds,
                "frequency_hz": 1,
            },
            "clock": self.clock(),
            "demo": {
                "playback_speeds": list(self.demo_config["playback_speeds"]),
                "scenario_modes": list(scenario["modes"]),
                "scenario_targets": list(scenario["targets"]),
                "mode_dwell_seconds": float(scenario["mode_dwell_seconds"]),
                "fault_hold_after_alarm_seconds": float(
                    scenario["fault_hold_after_alarm_seconds"]
                ),
                "reset_dwell_seconds": float(scenario["reset_dwell_seconds"]),
                "detection_timeout_seconds": float(
                    scenario["detection_timeout_seconds"]
                ),
            },
        }

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        command = str(payload.get("command", ""))
        if command == "play":
            self.play()
            return self.clock()
        if command == "pause":
            self.pause()
            return self.clock()
        if command == "set_speed":
            self.set_speed(int(payload.get("speed", self.demo_config["default_speed"])))
            return self.clock()
        if command == "set_mode":
            self.set_mode(int(payload.get("mode", self.requested_mode)))
            return self.snapshot()
        if command == "inject_fault":
            self.inject_fault(str(payload.get("target", "")))
            return self.snapshot()
        if command == "clear_fault":
            self.clear_fault()
            return self.snapshot()
        if command == "reset":
            self.reset()
            return self.snapshot()
        raise KeyError(command)


engine = DemoEngine()
app = FastAPI(title="Detector Anomaly — локальная демонстрация", docs_url=None, redoc_url=None)


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(HTML_PATH, media_type="text/html; charset=utf-8")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "detector_anomaly_public_demo",
        "model_count": engine.bundle.model_count,
        "targets": len(TARGETS),
        "trace_rows": engine.duration_seconds,
        "persistence_seconds": PERSISTENCE_SECONDS,
        "trace": "data/demo.csv",
    }


@app.get("/api/snapshot")
async def snapshot() -> dict[str, Any]:
    return engine.snapshot()


@app.get("/api/clock")
async def clock() -> dict[str, Any]:
    return engine.clock()


@app.post("/api/command")
async def command(payload: dict[str, Any]) -> dict[str, Any]:
    return engine.handle(payload)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
