"""Optional server-side camera worker (local USB device or RTSP/HTTP stream).

Browsers are the primary capture path (frames arrive over ``/ws/camera``). This
worker covers deployments where the camera is attached to, or reachable from,
the server itself: set ``FAIRFACE_CAMERA_SOURCE`` to a device index (``0``) or
a stream URL (``rtsp://...``) and start it from the dashboard.

The source is taken from server configuration only. It is deliberately not an
API parameter: ``cv2.VideoCapture`` will open arbitrary files and URLs, so
letting a client choose the source would be a file-read / request-forgery hole.

Two threads keep latency flat. The reader drains the device as fast as it
delivers and keeps only the newest frame; the analyser picks up whatever is
newest when it becomes free. Frames that arrive while a previous one is being
analysed are skipped, never queued. Frames live in memory only.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import cv2
import numpy as np

logger = logging.getLogger("fairface.camera")

PREVIEW_MAX_WIDTH = 960
PREVIEW_JPEG_QUALITY = 75
RECONNECT_DELAY_S = 2.0


def parse_source(raw: str | None) -> int | str | None:
    if raw is None or not raw.strip():
        return None
    raw = raw.strip()
    return int(raw) if raw.isdigit() else raw


def describe_source(source: int | str | None) -> str | None:
    """Human-readable source with any URL credentials removed."""
    if source is None:
        return None
    if isinstance(source, int):
        return f"device {source}"
    if "@" in source and "://" in source:
        scheme, rest = source.split("://", 1)
        # The password may itself contain "@": the host starts after the last one.
        return f"{scheme}://***@{rest.rsplit('@', 1)[1]}"
    return source


class ServerCamera:
    def __init__(
        self,
        source: int | str,
        analyse: Callable[[np.ndarray], dict],
        open_capture: Callable[[int | str], "cv2.VideoCapture"] = cv2.VideoCapture,
    ) -> None:
        self.source = source
        self._analyse = analyse
        self._open_capture = open_capture
        self._lock = threading.Lock()
        self._frame_ready = threading.Condition(self._lock)
        self._preview_ready = threading.Condition(self._lock)
        self._latest: np.ndarray | None = None
        self._latest_id = 0
        self._preview: bytes | None = None
        self._preview_id = 0
        self._result: dict | None = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.state = "stopped"
        self.error: str | None = None
        self.frames_read = 0
        self.frames_analysed = 0

    # -- lifecycle --------------------------------------------------------- #

    @property
    def running(self) -> bool:
        return any(t.is_alive() for t in self._threads)

    def start(self) -> None:
        if self.running:
            return
        # A fresh stop flag per run: a reader still blocked in a slow
        # capture.read() from the previous run keeps seeing *its* flag set and
        # exits when the read returns, instead of being revived by clear().
        self._stop = stop = threading.Event()
        self.state, self.error = "connecting", None
        self.frames_read = self.frames_analysed = 0
        self._threads = [
            threading.Thread(target=self._read_loop, args=(stop,), name="camera-read", daemon=True),
            threading.Thread(target=self._analyse_loop, args=(stop,), name="camera-analyse", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._frame_ready.notify_all()
            self._preview_ready.notify_all()
        for thread in self._threads:
            thread.join(timeout=5.0)
        self._threads = []
        with self._lock:
            self._latest = self._preview = self._result = None
        self.state = "stopped"

    def status(self) -> dict:
        skipped = max(0, self.frames_read - self.frames_analysed)
        return {
            "running": self.running,
            "state": self.state,
            "error": self.error,
            "source": describe_source(self.source),
            "frames_read": self.frames_read,
            "frames_analysed": self.frames_analysed,
            "frames_skipped": skipped,
        }

    # -- threads ----------------------------------------------------------- #

    def _read_loop(self, stop: threading.Event) -> None:
        capture = None
        while not stop.is_set():
            if capture is None:
                try:
                    capture = self._open_capture(self.source)
                    opened = bool(capture.isOpened())
                except Exception as exc:  # backend-specific failures
                    opened, self.error = False, f"{type(exc).__name__}: {exc}"
                if not opened:
                    self.state = "unavailable"
                    self.error = self.error or "could not open the camera source"
                    if capture is not None:
                        capture.release()
                    capture = None
                    stop.wait(RECONNECT_DELAY_S)
                    continue
                self.state, self.error = "running", None
            ok, frame = capture.read()
            if not ok or frame is None:
                # Device unplugged or stream dropped: reconnect instead of spinning.
                self.state, self.error = "reconnecting", "frame read failed"
                capture.release()
                capture = None
                stop.wait(RECONNECT_DELAY_S)
                continue
            if stop.is_set():
                break
            self.frames_read += 1
            with self._lock:
                self._latest = frame
                self._latest_id += 1
                self._frame_ready.notify_all()
        if capture is not None:
            capture.release()

    def _analyse_loop(self, stop: threading.Event) -> None:
        seen = self._latest_id
        while not stop.is_set():
            with self._lock:
                while self._latest_id == seen and not stop.is_set():
                    self._frame_ready.wait(timeout=0.5)
                if stop.is_set():
                    return
                frame, seen = self._latest, self._latest_id
            if frame is None:
                continue
            try:
                result = self._analyse(frame)
            except Exception as exc:
                logger.exception("server camera analysis failed")
                result = {"type": "error", "code": "processing-failed", "message": f"{type(exc).__name__}: {exc}"}
            preview = frame
            if preview.shape[1] > PREVIEW_MAX_WIDTH:
                scale = PREVIEW_MAX_WIDTH / preview.shape[1]
                preview = cv2.resize(preview, (PREVIEW_MAX_WIDTH, max(1, round(preview.shape[0] * scale))))
                result = {**result, "preview_scale": scale}
            ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
            self.frames_analysed += 1
            with self._lock:
                self._result = {**result, "seq": self.frames_analysed}
                if ok:
                    self._preview = encoded.tobytes()
                    self._preview_id += 1
                self._preview_ready.notify_all()

    # -- consumers --------------------------------------------------------- #

    def latest_result(self) -> dict | None:
        with self._lock:
            return self._result

    def next_preview(self, after_id: int, timeout: float = 2.0) -> tuple[int, bytes | None]:
        """Block until a preview newer than ``after_id`` exists (or timeout)."""
        deadline = time.monotonic() + timeout
        with self._lock:
            while self._preview_id == after_id and not self._stop.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return after_id, None
                self._preview_ready.wait(timeout=remaining)
            return self._preview_id, self._preview
