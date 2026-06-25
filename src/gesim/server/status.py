"""Thread-safe live status for the world-model server dashboard.

The HTTP handler threads write status as requests arrive; the dashboard reads a
snapshot over ``GET /api/status``. All access is guarded by one lock.
"""

from __future__ import annotations

import threading
import time


class ServerStatus:
    """A small mutable snapshot of what the server is doing right now."""

    def __init__(self, model_name: str, chunk_size: int):
        self._lock = threading.Lock()
        self._model_name = model_name
        self._chunk_size = chunk_size
        self._started_at = time.time()
        self._phase = "idle"  # idle -> ready -> running -> ready
        self._task = ""
        self._step_count = 0
        self._frames_generated = 0
        self._last_state: list[float] | None = None
        self._last_action: list[float] | None = None
        self._preview: bytes | None = None

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def on_init(self) -> None:
        with self._lock:
            self._phase = "ready"
            self._task = ""
            self._step_count = 0
            self._frames_generated = 0
            self._last_state = None
            self._last_action = None
            self._preview = None

    def on_task(self, task: str) -> None:
        with self._lock:
            self._task = task

    def on_close(self) -> None:
        with self._lock:
            self._phase = "idle"

    def on_step(self, *, frames: int, state_row, action_row, preview: bytes | None) -> None:
        with self._lock:
            self._phase = "ready"
            self._step_count += 1
            self._frames_generated += frames
            self._last_state = self._as_list(state_row)
            self._last_action = self._as_list(action_row)
            if preview is not None:
                self._preview = preview

    def preview(self) -> bytes | None:
        with self._lock:
            return self._preview

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "model": self._model_name,
                "chunk_size": self._chunk_size,
                "phase": self._phase,
                "uptime_s": round(time.time() - self._started_at, 1),
                "task": self._task,
                "step_count": self._step_count,
                "frames_generated": self._frames_generated,
                "state": self._last_state,
                "action": self._last_action,
                "has_preview": self._preview is not None,
            }

    @staticmethod
    def _as_list(row) -> list[float] | None:
        return [round(float(x), 4) for x in row] if row is not None else None
