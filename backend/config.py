"""Runtime configuration: paths, decision thresholds, ONNX execution providers.

Everything here is overridable through environment variables so the same code
runs unmodified on a laptop, in Docker and on a cloud host. Nothing in this
module imports onnxruntime at import time; provider detection is lazy so the
storage, decision and fairness layers stay importable without the ML stack.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
FRONTEND_DIR = BASE_DIR / "frontend"


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw).expanduser().resolve() if raw else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def data_dir() -> Path:
    return _env_path("FAIRFACE_DATA_DIR", BASE_DIR / "data")


def models_dir() -> Path:
    return _env_path("FAIRFACE_MODELS_DIR", BASE_DIR / "models")


# --------------------------------------------------------------------------- #
# Decision thresholds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Thresholds:
    """Operating point of the three-zone decision policy.

    ``tau_high``/``tau_low``/``margin`` live on the calibrated score scale
    ``s in [0, 1]`` produced by :mod:`backend.engine.calibrator`; ``q_min`` and
    ``l_min`` are on the quality and liveness scales, also ``[0, 1]``.
    """

    tau_high: float = 0.78
    tau_low: float = 0.48
    margin: float = 0.08
    q_min: float = 0.40
    l_min: float = 0.50

    def validate(self) -> "Thresholds":
        for name, value in asdict(self).items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be a number")
            if not (value == value) or value in (float("inf"), float("-inf")):
                raise ValueError(f"{name} must be finite")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1], got {value}")
        if self.tau_low >= self.tau_high:
            raise ValueError(
                f"tau_low ({self.tau_low}) must be strictly below tau_high ({self.tau_high})"
            )
        return self

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def thresholds_from_env() -> Thresholds:
    defaults = Thresholds()
    return Thresholds(
        tau_high=_env_float("FAIRFACE_TAU_HIGH", defaults.tau_high),
        tau_low=_env_float("FAIRFACE_TAU_LOW", defaults.tau_low),
        margin=_env_float("FAIRFACE_MARGIN", defaults.margin),
        q_min=_env_float("FAIRFACE_Q_MIN", defaults.q_min),
        l_min=_env_float("FAIRFACE_L_MIN", defaults.l_min),
    ).validate()


class ThresholdStore:
    """Thread-safe holder for the live operating point (editable from the UI)."""

    def __init__(self, initial: Thresholds | None = None) -> None:
        self._lock = threading.Lock()
        self._value = (initial or thresholds_from_env()).validate()

    def get(self) -> Thresholds:
        with self._lock:
            return self._value

    def update(self, **changes: float) -> Thresholds:
        unknown = set(changes) - set(Thresholds.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown threshold(s): {sorted(unknown)}")
        with self._lock:
            candidate = replace(self._value, **{k: float(v) for k, v in changes.items()})
            candidate.validate()
            self._value = candidate
            return candidate


# --------------------------------------------------------------------------- #
# Engine settings
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    # Detection (InsightFace SCRFD defaults).
    det_thresh: float
    det_nms: float
    det_size: int
    max_faces: int
    # Liveness can be switched off for cameras/streams where it is unreliable;
    # the decision engine then skips the SPOOF check instead of faking a score.
    liveness_enabled: bool
    # Naive comparison engine: raw ArcFace cosine against one fixed threshold.
    baseline_cosine_threshold: float
    # Enrollment template policy.
    enroll_min_samples: int
    enroll_max_samples: int
    enroll_min_quality: float
    # Stream handling.
    max_upload_bytes: int
    max_frame_pixels: int
    stream_audit_interval_s: float
    # Optional shared secret for the API and WebSockets.
    api_key: str | None
    # WebRTC ICE servers handed to browsers (JSON list, RTCIceServer objects).
    ice_servers: tuple[dict, ...]
    public_base_url: str | None
    onnx_threads: int
    # Server-side camera (device index or stream URL); None disables the worker.
    camera_source: str | None
    # Photos kept per person (rolling: newest kept). Larger than one capture so
    # check-ins on several days build up a template covering hair, glasses and light.
    enroll_retain_samples: int = 15
    # Employee onboarding: number of separate check-in days that complete it.
    onboarding_days: int = 5
    # Face crops behind open review cases: kept at most this many hours (and
    # deleted as soon as a person decides the case). 0 turns snapshots off.
    review_snapshot_hours: float = 24.0


DEFAULT_ICE_SERVERS = ({"urls": ["stun:stun.l.google.com:19302"]},)


def _ice_servers_from_env() -> tuple[dict, ...]:
    raw = os.environ.get("FAIRFACE_ICE_SERVERS")
    if not raw:
        return DEFAULT_ICE_SERVERS
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("FAIRFACE_ICE_SERVERS must be a JSON list of RTCIceServer objects") from exc
    if not isinstance(parsed, list) or not all(isinstance(s, dict) and "urls" in s for s in parsed):
        raise ValueError("FAIRFACE_ICE_SERVERS must be a JSON list of objects with a 'urls' field")
    return tuple(parsed)


def load_settings() -> Settings:
    settings = Settings(
        host=os.environ.get("FAIRFACE_HOST", "127.0.0.1"),
        port=_env_int("FAIRFACE_PORT", 8000),
        det_thresh=_env_float("FAIRFACE_DET_THRESH", 0.5),
        det_nms=_env_float("FAIRFACE_DET_NMS", 0.4),
        det_size=_env_int("FAIRFACE_DET_SIZE", 640),
        max_faces=_env_int("FAIRFACE_MAX_FACES", 6),
        liveness_enabled=_env_bool("FAIRFACE_LIVENESS", True),
        baseline_cosine_threshold=_env_float("FAIRFACE_BASELINE_COSINE", 0.32),
        enroll_min_samples=_env_int("FAIRFACE_ENROLL_MIN_SAMPLES", 3),
        enroll_max_samples=_env_int("FAIRFACE_ENROLL_MAX_SAMPLES", 5),
        enroll_min_quality=_env_float("FAIRFACE_ENROLL_MIN_QUALITY", 0.40),
        max_upload_bytes=_env_int("FAIRFACE_MAX_UPLOAD_BYTES", 8 * 1024 * 1024),
        max_frame_pixels=_env_int("FAIRFACE_MAX_FRAME_PIXELS", 4096 * 4096),
        stream_audit_interval_s=_env_float("FAIRFACE_STREAM_AUDIT_INTERVAL", 2.0),
        api_key=os.environ.get("FAIRFACE_API_KEY") or None,
        ice_servers=_ice_servers_from_env(),
        public_base_url=(os.environ.get("FAIRFACE_PUBLIC_URL") or "").rstrip("/") or None,
        onnx_threads=_env_int("FAIRFACE_ONNX_THREADS", 0),
        camera_source=(os.environ.get("FAIRFACE_CAMERA_SOURCE") or "").strip() or None,
        enroll_retain_samples=_env_int("FAIRFACE_ENROLL_RETAIN_SAMPLES", 15),
        onboarding_days=_env_int("FAIRFACE_ONBOARDING_DAYS", 5),
        review_snapshot_hours=_env_float("FAIRFACE_REVIEW_SNAPSHOT_HOURS", 24.0),
    )
    if settings.det_size % 32 != 0 or settings.det_size < 160:
        raise ValueError("FAIRFACE_DET_SIZE must be a multiple of 32 and at least 160")
    if not 1 <= settings.enroll_min_samples <= settings.enroll_max_samples:
        raise ValueError("enrollment sample bounds are inconsistent")
    if settings.enroll_retain_samples < settings.enroll_max_samples:
        raise ValueError("FAIRFACE_ENROLL_RETAIN_SAMPLES must be at least FAIRFACE_ENROLL_MAX_SAMPLES")
    if settings.onboarding_days < 1:
        raise ValueError("FAIRFACE_ONBOARDING_DAYS must be at least 1")
    return settings


# --------------------------------------------------------------------------- #
# ONNX Runtime execution providers
# --------------------------------------------------------------------------- #

# Accelerators first, CPU always last so a broken/missing GPU stack degrades
# instead of failing. Order mirrors InsightFace's provider policy, extended
# with DirectML for Windows laptops.
PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",
    "CoreMLExecutionProvider",
    "DmlExecutionProvider",
    "CPUExecutionProvider",
)


def select_providers(available: list[str] | None = None) -> list[str]:
    """Return the provider list to hand to ``onnxruntime.InferenceSession``.

    ``FAIRFACE_ONNX_PROVIDERS`` (comma separated) overrides auto-detection;
    providers this build of onnxruntime does not ship are dropped, and
    ``CPUExecutionProvider`` is always appended as the final fallback.
    """
    if available is None:
        import onnxruntime as ort

        available = list(ort.get_available_providers())
    override = os.environ.get("FAIRFACE_ONNX_PROVIDERS")
    wanted = (
        [p.strip() for p in override.split(",") if p.strip()] if override else list(PROVIDER_PRIORITY)
    )
    chosen = [p for p in wanted if p in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen
