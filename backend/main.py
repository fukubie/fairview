"""FastAPI application: REST API, live camera WebSocket and WebRTC signaling.

REST surface (resource layout follows CompreFace's subject/example model):

    POST   /api/enroll                 enroll 1..N frames for a subject
    POST   /api/identify               1:N open-set identification of a frame
    POST   /api/verify                 1:1 verification against a claimed subject
    POST   /api/verify/pair            1:1 comparison of two frames (nothing stored)
    GET    /api/subjects               list subjects
    PUT    /api/subjects/{subject}     rename
    DELETE /api/subjects/{subject}     erase a subject and their templates
    DELETE /api/subjects?confirm=true  erase everything
    POST/DELETE /api/subjects/{subject}/ban   banned-from-premises list
    GET    /api/review/cases           review queue (one case per camera moment)
    POST   /api/review/cases/{id}      a reviewer's decision
    GET    /api/review/frames/{seq}    face crop behind an open case
    GET    /api/audit                  recent decisions
    GET    /api/audit/verify           hash-chain integrity check
    GET    /api/audit/export           full log as JSON or CSV
    GET    /api/fairness/audit         synthetic subgroup benchmark
    GET    /api/fairness/rfw           real-score RFW report (when data is present)
    POST   /api/stress/lookalike       lookalike / abstention simulation
    GET/PUT /api/config[/thresholds]   engine status and operating point
    GET/POST /api/server-camera/...    optional server-side USB / RTSP camera worker
    POST   /api/rooms                  create a WebRTC room (join URL + QR)
    DELETE /api/rooms/{room_id}        end a call and revoke its join link

WebSockets:

    /ws/camera            browser frames in, per-face decisions out
    /ws/room/{room_id}    SDP / ICE relay between the two peers of a room

Frames are decoded, analysed and discarded in memory, and multipart uploads are
kept in memory up to the per-image size limit instead of being spooled to a
temporary file (see ``create_app``). Two things are stored as images, both
encrypted in the vault: the small face crop behind an open review case (deleted
when a person decides the case, or after ``FAIRFACE_REVIEW_SNAPSHOT_HOURS``),
and the record photo an administrator adds for a banned person.
"""

from __future__ import annotations

import asyncio
import io
import ipaddress
import json
import logging
import os
import re
import secrets
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import config
from backend.camera import ServerCamera, parse_source
from backend.config import Settings, Thresholds, ThresholdStore
from backend.engine.decision import Mode
from backend.engine.pipeline import (
    FrameResult,
    ImageError,
    IncompleteTemplate,
    RecognitionPipeline,
    UnknownSubject,
    decode_image,
)
from backend.security import LOOPBACK_HOSTS, RequestGuard, is_loopback, keys_equal
from backend.storage.audit import AuditLog
from backend.storage import review as review_queue
from backend.storage.vault import (
    BiometricVault,
    SubjectExists,
    SubjectNotFound,
    VaultError,
    normalize_subject_name,
)

logger = logging.getLogger("fairface.api")

STREAM_NAMES = {0: "local", 1: "remote"}
MAX_ROOMS = 200
ROOM_IDLE_SECONDS = 6 * 3600
MAX_SIGNAL_BYTES = 64 * 1024
SIGNAL_TYPES = {"offer", "answer", "candidate", "bye"}

PipelineFactory = Callable[[BiometricVault, Settings], tuple[RecognitionPipeline, dict]]


# --------------------------------------------------------------------------- #
# Application state
# --------------------------------------------------------------------------- #


@dataclass
class Room:
    room_id: str
    created: float
    last_active: float
    peers: dict[str, WebSocket] = field(default_factory=dict)  # role -> socket
    host_token: str = ""  # only the creator of the room may join as host


class AppState:
    def __init__(self, settings: Settings, pipeline_factory: PipelineFactory | None) -> None:
        self.settings = settings
        self.thresholds = ThresholdStore()
        self.data_dir = config.data_dir()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vault = BiometricVault(self.data_dir / "biometrics.db")
        self.audit = AuditLog(self.data_dir / "biometrics.db", self.data_dir / "audit.jsonl")
        self.pipeline: RecognitionPipeline | None = None
        self.engine_info: dict = {"ready": False, "error": "models are loading"}
        self._pipeline_factory = pipeline_factory
        # One inference at a time: ONNX Runtime already parallelises inside a
        # call, and serialising calls keeps latency predictable on laptops.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
        self.aux_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="aux")
        self.rooms: dict[str, Room] = {}
        self.server_camera: ServerCamera | None = None
        self.server_camera_session = {"mode": Mode.IDENTIFY, "claimed": None, "compare": False}
        self.benchmark = None
        self._benchmark_lock = asyncio.Lock()
        self.started = time.time()

    def load_pipeline(self) -> None:
        if self._pipeline_factory is None:
            self.engine_info = {"ready": False, "error": "no recognition engine configured"}
            return
        try:
            self.pipeline, info = self._pipeline_factory(self.vault, self.settings)
            self.engine_info = {"ready": True, **info}
        except Exception as exc:  # surfaced through /api/health and 503s
            logger.exception("recognition engine failed to start")
            self.pipeline = None
            self.engine_info = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}

    def require_pipeline(self) -> RecognitionPipeline:
        if self.pipeline is None:
            raise HTTPException(503, detail=f"recognition engine unavailable: {self.engine_info.get('error')}")
        return self.pipeline

    async def infer(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def aux(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.aux_executor, fn, *args)

    async def get_benchmark(self):
        async with self._benchmark_lock:
            if self.benchmark is None:
                from backend.fairness.audit import FairnessBenchmark

                self.benchmark = await self.aux(FairnessBenchmark)
            return self.benchmark

    def close(self) -> None:
        if self.server_camera is not None:
            self.server_camera.stop()
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.aux_executor.shutdown(wait=False, cancel_futures=True)
        self.audit.close()
        self.vault.close()


# --------------------------------------------------------------------------- #
# Audit helpers
# --------------------------------------------------------------------------- #


def audit_frame(state: AppState, result: FrameResult, source: str, face_count: int | None = None) -> list[dict]:
    """Write one audit entry per analysed face. Returns the stored entries."""
    entries = []
    face_count = len(result.faces) if face_count is None else face_count
    for face in result.faces:
        decision = face.decision
        entries.append(
            state.audit.record(
                mode=decision.mode.value,
                decision_zone=decision.zone.value,
                abstention_reason=decision.reason.value,
                quality_score=decision.quality,
                liveness_score=decision.liveness,
                top1_subject=decision.top1.subject if decision.top1 else None,
                top1_score=decision.top1.score if decision.top1 else None,
                top2_subject=decision.top2.subject if decision.top2 else None,
                top2_score=decision.top2.score if decision.top2 else None,
                margin=decision.margin,
                thresholds=decision.thresholds.to_dict(),
                engine="fair",
                source=source,
                asserted_subject=decision.subject,
                claimed_subject=result.claimed_subject,
                face_count=face_count,
            )
        )
    return entries


class StreamAuditThrottle:
    """Logs each distinct outcome on a video stream at most once per interval.

    A live stream produces many frames per second showing the same person; the
    audit log records the decision when it first appears, whenever it changes,
    and periodically while it persists, instead of once per frame.
    """

    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._last: dict[tuple, float] = {}

    def due(self, key: tuple, now: float) -> bool:
        last = self._last.get(key)
        if last is not None and now - last < self.interval_s:
            return False
        if len(self._last) > 512:
            self._last = {k: v for k, v in self._last.items() if now - v < self.interval_s}
        self._last[key] = now
        return True


SNAPSHOT_MAX_SIDE = 256
SNAPSHOT_MAX_COUNT = 2000


def face_snapshot(image: np.ndarray, bbox) -> bytes | None:
    """A small JPEG of one face with some context around it, for a reviewer."""
    import cv2

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    pad_x, pad_y = 0.35 * (x2 - x1), 0.35 * (y2 - y1)
    left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
    right, bottom = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
    if right - left < 8 or bottom - top < 8:
        return None
    crop = image[top:bottom, left:right]
    scale = SNAPSHOT_MAX_SIDE / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return encoded.tobytes() if ok else None


def keep_review_snapshots(state: AppState, faces, entries: list[dict], image: np.ndarray | None) -> None:
    """Store a face crop for each logged face a person will have to review."""
    if image is None or state.settings.review_snapshot_hours <= 0:
        return
    banned = None
    for face, entry in zip(faces, entries):
        if entry.get("decision_zone") not in review_queue.REVIEWABLE_ZONES:
            if banned is None:
                banned = set(state.vault.banned())
            if not review_queue.is_reviewable(entry, banned):
                continue
        elif not review_queue.is_reviewable(entry):
            continue
        try:
            jpeg = face_snapshot(image, face.observation.detection.bbox)
            if jpeg:
                state.vault.add_snapshot(entry["seq"], jpeg)
        except Exception:  # a missing snapshot must never break recognition
            logger.exception("could not keep a review snapshot")


def audit_stream_frame(state: AppState, result: FrameResult, stream: str, throttle: StreamAuditThrottle,
                       image: np.ndarray | None = None) -> None:
    now = time.monotonic()
    due_faces = []
    for face in result.faces:
        d = face.decision
        # Coarse position keeps two people with the same outcome in one frame
        # (e.g. two unknown faces) from being collapsed into a single entry.
        box = face.observation.detection.bbox
        cell = (int(4 * (box[0] + box[2]) / 2 / max(result.width, 1)), int(4 * (box[1] + box[3]) / 2 / max(result.height, 1)))
        key = (stream, d.mode.value, d.zone.value, d.reason.value, d.top1.subject if d.top1 else None, cell)
        if throttle.due(key, now):
            due_faces.append(face)
    if due_faces:
        entries = audit_frame(state, FrameResult(result.width, result.height, result.mode, due_faces,
                                                 claimed_subject=result.claimed_subject), f"stream:{stream}",
                              face_count=len(result.faces))
        keep_review_snapshots(state, due_faces, entries, image)


# --------------------------------------------------------------------------- #
# Request models
# --------------------------------------------------------------------------- #


class ThresholdUpdate(BaseModel):
    tau_high: float | None = Field(None, ge=0, le=1)
    tau_low: float | None = Field(None, ge=0, le=1)
    margin: float | None = Field(None, ge=0, le=1)
    q_min: float | None = Field(None, ge=0, le=1)
    l_min: float | None = Field(None, ge=0, le=1)


class RenameRequest(BaseModel):
    subject: str


class ReviewDecision(BaseModel):
    outcome: str = Field(..., pattern="^(same_person|different_person|spoof)$")
    # For "same_person": who the reviewer confirmed. Defaults to the case's best match.
    subject: str | None = Field(None, max_length=80)


class RoomRequest(BaseModel):
    base_url: str | None = Field(None, max_length=300)


class CameraStartRequest(BaseModel):
    mode: Mode = Mode.IDENTIFY
    claimed_subject: str | None = None
    compare_baseline: bool = False


class LookalikeRequest(BaseModel):
    similarity: float = Field(0.85, ge=0, lt=1)
    quality: float = Field(0.8, ge=0, le=1)
    both_enrolled: bool = True
    group: str = "IV"
    trials: int = Field(200, ge=1, le=1000)
    tau_high: float | None = Field(None, ge=0, le=1)
    tau_low: float | None = Field(None, ge=0, le=1)
    margin: float | None = Field(None, ge=0, le=1)
    q_min: float | None = Field(None, ge=0, le=1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def lan_addresses() -> list[str]:
    """Best-effort list of this host's non-loopback IPv4 addresses."""
    found: list[str] = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("192.0.2.1", 9))  # TEST-NET-1: no packet is sent for UDP connect
            found.append(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.append(info[4][0])
    except OSError:
        pass
    unique = []
    for address in found:
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if not parsed.is_loopback and address not in unique:
            unique.append(address)
    return unique


def _is_loopback_host(host: str) -> bool:
    if host in ("localhost", ""):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def qr_svg(text: str) -> str:
    import qrcode
    import qrcode.image.svg

    code = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=2)
    code.add_data(text)
    code.make(fit=True)
    buffer = io.BytesIO()
    code.make_image(image_factory=qrcode.image.svg.SvgPathImage).save(buffer)
    return buffer.getvalue().decode("utf-8")


def thresholds_from_query(state: AppState, **overrides: float | None) -> Thresholds:
    base = state.thresholds.get().to_dict()
    base.update({k: v for k, v in overrides.items() if v is not None})
    try:
        return Thresholds(**base).validate()
    except ValueError as exc:
        raise HTTPException(422, detail=str(exc)) from exc


def _same_origin(websocket: WebSocket) -> bool:
    """Reject cross-site WebSocket connections (browsers always send Origin)."""
    # RequestGuard has already enforced the origin policy (including the
    # configured public origin) before the handshake reached this handler.
    return True


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #


def default_pipeline_factory(vault: BiometricVault, settings: Settings):
    from backend.engine.model_manager import build_pipeline

    return build_pipeline(vault, settings)


def create_app(pipeline_factory: PipelineFactory | None = default_pipeline_factory) -> FastAPI:
    settings = config.load_settings()
    # Starlette spools upload parts larger than 1 MB to a temporary file on
    # disk. Raise the in-memory threshold above the largest image we accept so
    # an uploaded face never touches the filesystem, even transiently.
    from starlette.formparsers import MultiPartParser

    max_body_bytes = settings.enroll_max_samples * settings.max_upload_bytes + 1024 * 1024
    # The guard below refuses any request whose declared body exceeds
    # max_body_bytes, so with the spool threshold at the same value no part can
    # ever roll over to a temporary file.
    MultiPartParser.spool_max_size = max(MultiPartParser.spool_max_size, max_body_bytes + 1024)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = AppState(settings, pipeline_factory)
        app.state.ffg = state
        # Model loading can take seconds (or a download); keep the server
        # responsive and report progress through /api/health meanwhile.
        loader = asyncio.get_running_loop().run_in_executor(state.executor, state.load_pipeline)
        try:
            yield
        finally:
            loader.cancel()
            state.close()

    app = FastAPI(title="FairFace Guard", version="1.0.0", lifespan=lifespan, docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")

    def get_state(request: Request) -> AppState:
        return request.app.state.ffg

    def require_key(request: Request) -> None:
        # Second line of defence; RequestGuard already rejected bad keys
        # before the body was read.
        if settings.api_key and not keys_equal(request.headers.get("x-api-key"), settings.api_key):
            raise HTTPException(401, detail="missing or invalid X-API-Key")

    protected = [Depends(require_key)]

    allowed_hosts: set[str] | None = set(LOOPBACK_HOSTS)
    configured_hosts = [h.strip().lower() for h in os.environ.get("FAIRFACE_ALLOWED_HOSTS", "").split(",") if h.strip()]
    if "*" in configured_hosts:
        allowed_hosts = None
    else:
        allowed_hosts.update(configured_hosts)
        if settings.public_base_url:
            allowed_hosts.add((urlsplit(settings.public_base_url).hostname or "").lower())
        if not is_loopback(settings.host):
            allowed_hosts.update(lan_addresses())
            allowed_hosts.add(socket.gethostname().lower())
        # GitHub Codespaces forwards the port under a predictable public name.
        codespace, domain = os.environ.get("CODESPACE_NAME"), os.environ.get("GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN")
        if codespace and domain:
            allowed_hosts.add(f"{codespace}-{settings.port}.{domain}".lower())
    app.add_middleware(
        RequestGuard,
        api_key=settings.api_key,
        allowed_hosts=allowed_hosts,
        max_body_bytes=max_body_bytes,
        open_paths=("/api/health",),
        # Reachable without the key: what a room guest needs (each handler
        # checks the room id), and the token-protected camera preview.
        open_prefixes=("/api/rooms/", "/api/rtc-config", "/api/server-camera/stream.mjpg"),
        trusted_origins=(settings.public_base_url or "",),
    )

    @app.exception_handler(VaultError)
    async def vault_unreadable(_request: Request, exc: VaultError):
        logger.error("biometric store error: %s", exc)
        return JSONResponse({"detail": f"biometric store unavailable: {exc}"}, status_code=503)

    def admin_event(state: AppState, event: str, subject: str | None = None, detail: str | None = None,
                    other: str | None = None) -> None:
        """Chain an administrative action into the audit log.

        Enrollment, erasure, renames and threshold changes alter what every
        later decision means, so they are recorded alongside the decisions.
        """
        state.audit.record(
            mode="admin", decision_zone=event, abstention_reason=detail, quality_score=None,
            liveness_score=None, top1_subject=subject, top1_score=None, top2_score=None, margin=None,
            thresholds=state.thresholds.get().to_dict(), engine="n/a", source="api:admin", claimed_subject=other,
        )

    # Pages that may show the dashboard in a frame, e.g. the Hugging Face Space
    # page at https://huggingface.co. None by default: the dashboard is never
    # framed unless the operator names the exact HTTPS origins allowed to.
    frame_ancestors = [
        origin for origin in os.environ.get("FAIRFACE_FRAME_ANCESTORS", "").split()
        if re.fullmatch(r"https://[A-Za-z0-9.-]+(:\d+)?", origin)
    ]
    frame_ancestors_policy = " ".join(frame_ancestors) or "'none'"

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not frame_ancestors:
            response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Permissions-Policy", "camera=(self), microphone=(), geolocation=()")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "connect-src 'self' ws: wss:; style-src 'self'; script-src 'self'; "
            f"frame-ancestors {frame_ancestors_policy}; "
            "base-uri 'none'; form-action 'self'",
        )
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    def request_state(_upload: UploadFile) -> AppState:
        return app.state.ffg

    async def read_image(upload: UploadFile) -> np.ndarray:
        data = await upload.read(settings.max_upload_bytes + 1)
        try:
            # Decoding is CPU work; keep it off the event loop.
            return await request_state(upload).aux(
                decode_image, data, settings.max_upload_bytes, settings.max_frame_pixels
            )
        except ImageError as exc:
            raise HTTPException(422, detail=f"{upload.filename or 'image'}: {exc}") from exc

    # -- status & configuration ------------------------------------------- #

    @app.get("/api/health")
    async def health(state: AppState = Depends(get_state)):
        calibration = state.engine_info.get("calibration") or {}
        calibrated = bool(calibration.get("fitted_on_data")) and bool(calibration.get("equalises_subgroups"))
        return {
            "status": "ok" if state.engine_info.get("ready") and calibrated else "degraded",
            # False means the built-in fallback thresholds are in use: no
            # false-match rate was measured for them and subgroup equalisation
            # is off. Rebuild backend/assets with tools/build_calibration.py.
            "calibrated": calibrated if state.engine_info.get("ready") else None,
            "engine_ready": bool(state.engine_info.get("ready")),
            "engine_error": state.engine_info.get("error") if not state.engine_info.get("ready") else None,
            "uptime_seconds": round(time.time() - state.started, 1),
        }

    @app.get("/api/config", dependencies=protected)
    async def get_config(state: AppState = Depends(get_state)):
        return {
            "thresholds": state.thresholds.get().to_dict(),
            "engine": state.engine_info,
            "liveness_enabled": settings.liveness_enabled,
            "baseline_cosine_threshold": settings.baseline_cosine_threshold,
            "enrollment": {
                "min_samples": settings.enroll_min_samples,
                "max_samples": settings.enroll_max_samples,
                "min_quality": settings.enroll_min_quality,
                "retain_samples": settings.enroll_retain_samples,
                "onboarding_days": settings.onboarding_days,
            },
            "storage": {"cipher": state.vault.cipher_name, "key_source": state.vault.key_source,
                        "raw_images_stored": False},
            "ice_servers": list(settings.ice_servers),
            "auth_required": bool(settings.api_key),
        }

    @app.put("/api/config/thresholds", dependencies=protected)
    async def put_thresholds(update: ThresholdUpdate, state: AppState = Depends(get_state)):
        changes = {k: v for k, v in update.model_dump().items() if v is not None}
        try:
            updated = state.thresholds.update(**changes).to_dict()
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        if changes:
            await state.aux(admin_event, state, "THRESHOLDS_CHANGED", None,
                            json.dumps(changes, sort_keys=True))
        return {"thresholds": updated}

    @app.get("/api/rtc-config")
    async def rtc_config(request: Request, room: str = "", state: AppState = Depends(get_state)):
        # ICE configuration may carry TURN credentials. Guests have no API
        # key, so they must name an open room; everyone else needs the key.
        if room not in state.rooms:
            if settings.api_key and not keys_equal(request.headers.get("x-api-key"), settings.api_key):
                raise HTTPException(401, detail="a valid room id or API key is required")
        return {"ice_servers": list(settings.ice_servers)}

    # -- enrollment & subjects -------------------------------------------- #

    @app.post("/api/enroll", dependencies=protected)
    async def enroll(
        subject: str = Form(...),
        files: list[UploadFile] = File(...),
        replace: bool = Form(False),
        state: AppState = Depends(get_state),
    ):
        pipeline = state.require_pipeline()
        try:
            name = normalize_subject_name(subject)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        if not 1 <= len(files) <= settings.enroll_max_samples:
            raise HTTPException(422, detail=f"send between 1 and {settings.enroll_max_samples} images")
        images = [await read_image(f) for f in files]
        result = await state.infer(pipeline.enroll, name, images, state.thresholds.get(), replace)
        del images
        if not result["enrolled"]:
            return JSONResponse(result, status_code=422)
        await state.aux(admin_event, state, "SUBJECT_REPLACED" if replace else "SUBJECT_ENROLLED", result["subject"],
                        f"accepted={result['accepted']} samples={result['samples']}")
        return result

    @app.get("/api/subjects", dependencies=protected)
    async def list_subjects(state: AppState = Depends(get_state)):
        subjects = [s.to_dict() for s in await state.aux(state.vault.list_subjects)]
        banned = await state.aux(state.vault.banned)
        for item in subjects:
            item["template_complete"] = item["samples"] >= settings.enroll_min_samples
            item["active"] = item["template_complete"]
            item["onboarding_complete"] = item["check_in_days"] >= settings.onboarding_days
            entry = banned.get(item["subject"])
            item["banned"] = entry is not None
            item["banned_note"] = entry["note"] if entry else None
            item["banned_photo"] = bool(entry and entry["has_photo"])
        return {"subjects": subjects, "count": len(subjects)}

    # -- banned from premises ---------------------------------------------- #

    def record_photo_jpeg(image: np.ndarray) -> bytes:
        import cv2

        scale = 512 / max(image.shape[:2])
        if scale < 1:
            image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)),
                               interpolation=cv2.INTER_AREA)
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(422, detail="the record photo could not be encoded")
        return encoded.tobytes()

    @app.post("/api/subjects/{subject}/ban", dependencies=protected)
    async def ban_subject(
        subject: str,
        note: str = Form("", max_length=300),
        photo: UploadFile | None = File(None),
        state: AppState = Depends(get_state),
    ):
        jpeg = None
        if photo is not None and photo.filename:
            # Decoding and re-encoding also drops any metadata in the upload.
            jpeg = await state.aux(record_photo_jpeg, await read_image(photo))
        try:
            await state.aux(state.vault.set_banned, subject, " ".join(note.split()) or None, jpeg)
        except SubjectNotFound:
            raise HTTPException(404, detail="subject not found") from None
        await state.aux(admin_event, state, "SUBJECT_BANNED", subject, "record photo" if jpeg else None)
        return {"subject": subject, "banned": True}

    @app.delete("/api/subjects/{subject}/ban", dependencies=protected)
    async def unban_subject(subject: str, state: AppState = Depends(get_state)):
        try:
            removed = await state.aux(state.vault.clear_banned, subject)
        except SubjectNotFound:
            raise HTTPException(404, detail="subject not found") from None
        if removed:
            await state.aux(admin_event, state, "SUBJECT_UNBANNED", subject)
        return {"subject": subject, "banned": False}

    @app.get("/api/subjects/{subject}/record-photo", dependencies=protected)
    async def banned_record_photo(subject: str, state: AppState = Depends(get_state)):
        try:
            jpeg = await state.aux(state.vault.banned_photo, subject)
        except SubjectNotFound:
            raise HTTPException(404, detail="subject not found") from None
        if jpeg is None:
            raise HTTPException(404, detail="no record photo")
        return Response(jpeg, media_type="image/jpeg")

    @app.put("/api/subjects/{subject}", dependencies=protected)
    async def rename_subject(subject: str, body: RenameRequest, state: AppState = Depends(get_state)):
        try:
            renamed = (await state.aux(state.vault.rename_subject, subject, body.subject)).to_dict()
            await state.aux(admin_event, state, "SUBJECT_RENAMED", renamed["subject"], None, subject)
            return renamed
        except SubjectNotFound:
            raise HTTPException(404, detail="subject not found") from None
        except SubjectExists:
            raise HTTPException(409, detail="a subject with that name already exists") from None
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.delete("/api/subjects/{subject}", dependencies=protected)
    async def delete_subject(subject: str, state: AppState = Depends(get_state)):
        try:
            removed = await state.aux(state.vault.delete_subject, subject)
            await state.aux(admin_event, state, "SUBJECT_DELETED", subject, f"samples={removed}")
        except SubjectNotFound:
            raise HTTPException(404, detail="subject not found") from None
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return {"subject": subject, "samples_deleted": removed}

    @app.delete("/api/subjects", dependencies=protected)
    async def delete_all_subjects(confirm: bool = Query(False), state: AppState = Depends(get_state)):
        if not confirm:
            raise HTTPException(400, detail="pass confirm=true to erase every enrolled subject")
        deleted = await state.aux(state.vault.delete_all)
        await state.aux(admin_event, state, "GALLERY_ERASED", None, f"subjects={deleted}")
        return {"deleted": deleted}

    # -- recognition ------------------------------------------------------- #

    async def recognise(state: AppState, image: np.ndarray, mode: Mode, claimed: str | None, compare: bool, source: str):
        pipeline = state.require_pipeline()
        try:
            result: FrameResult = await state.infer(
                pipeline.process, image, mode, state.thresholds.get(), claimed, compare
            )
        except UnknownSubject:
            raise HTTPException(404, detail="claimed subject is not enrolled") from None
        except IncompleteTemplate as exc:
            raise HTTPException(409, detail=str(exc)) from None
        entries = await state.aux(audit_frame, state, result, source)
        await state.aux(keep_review_snapshots, state, result.faces, entries, image)
        if not result.faces:
            # An attempt that found nobody is still a recognition event.
            entries = [await state.aux(lambda: state.audit.record(
                mode=mode.value, decision_zone="NO_FACE_DETECTED", abstention_reason=None, quality_score=None,
                liveness_score=None, top1_subject=None, top1_score=None, top2_score=None, margin=None,
                thresholds=state.thresholds.get().to_dict(), source=source, claimed_subject=claimed, face_count=0,
            ))]
        payload = result.to_dict()
        payload["audit"] = [{"seq": e["seq"], "hash": e["hash"]} for e in entries]
        return payload

    @app.post("/api/identify", dependencies=protected)
    async def identify(
        file: UploadFile = File(...),
        compare_baseline: bool = Form(False),
        state: AppState = Depends(get_state),
    ):
        return await recognise(state, await read_image(file), Mode.IDENTIFY, None, compare_baseline, "api:identify")

    @app.post("/api/verify", dependencies=protected)
    async def verify(
        subject: str = Form(...),
        file: UploadFile = File(...),
        compare_baseline: bool = Form(False),
        state: AppState = Depends(get_state),
    ):
        try:
            claimed = normalize_subject_name(subject)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc
        return await recognise(state, await read_image(file), Mode.VERIFY, claimed, compare_baseline, "api:verify")

    @app.post("/api/verify/pair", dependencies=protected)
    async def verify_pair(
        file_a: UploadFile = File(...),
        file_b: UploadFile = File(...),
        state: AppState = Depends(get_state),
    ):
        pipeline = state.require_pipeline()
        image_a, image_b = await read_image(file_a), await read_image(file_b)
        result = await state.infer(pipeline.compare, image_a, image_b, state.thresholds.get())
        decision = result.get("decision")
        if decision is not None:
            top1 = decision["top1"] or {}
            entry = await state.aux(lambda: state.audit.record(
                mode="verify", decision_zone=decision["zone"], abstention_reason=decision["reason"],
                quality_score=decision["quality"], liveness_score=decision["liveness"],
                top1_subject=None, top1_score=top1.get("score"), top2_score=None, margin=None,
                thresholds=decision["thresholds"], engine="fair", source="api:verify-pair", face_count=2,
            ))
            result["audit"] = [{"seq": entry["seq"], "hash": entry["hash"]}]
        return result

    # -- audit log --------------------------------------------------------- #

    @app.get("/api/audit", dependencies=protected)
    async def audit_recent(limit: int = Query(50, ge=1, le=1000), state: AppState = Depends(get_state)):
        entries = await state.aux(state.audit.entries, limit, True)
        head = await state.aux(state.audit.head)
        return {"entries": entries, "total": head["entries"], "head_hash": head["head_hash"]}

    # -- human review queue ------------------------------------------------ #

    REVIEW_WINDOW = 5000  # most recent audit entries scanned for cases

    def current_cases(state: AppState) -> list[dict]:
        """Build the cases and tidy snapshots: decided cases lose theirs, old ones expire."""
        entries = state.audit.entries(REVIEW_WINDOW, True)
        banned = state.vault.banned()
        state.vault.purge_snapshots(max(0.0, state.settings.review_snapshot_hours) * 3600, SNAPSHOT_MAX_COUNT)
        stored = state.vault.snapshot_seqs()
        cases = review_queue.build_cases(entries, banned, stored)
        # A decided case keeps no pictures; an open one keeps only its clearest
        # frames, however long the person stays in front of the camera.
        drop = []
        for case in cases:
            shown = {f["seq"] for f in case["frames"]} if case["status"] == "open" else set()
            drop.extend(s for s in case["seqs"] if s in stored and s not in shown)
        if drop:
            state.vault.delete_snapshots(drop)
            stored -= set(drop)
        for case in cases:
            case["frames"] = [dict(f, image=f["seq"] in stored) for f in case["frames"]]
            case["banned"] = [{"subject": c["subject"], "note": banned[c["subject"]]["note"],
                               "photo": banned[c["subject"]]["has_photo"]}
                              for c in case["candidates"] if c["subject"] in banned]
        # Cases dismissed by "clear the queue" are gone from the queue; the
        # audit log still holds every entry behind them.
        return [c for c in cases if c["status"] != "dismissed"]

    def public_case(case: dict) -> dict:
        return {k: v for k, v in case.items() if k != "seqs"}

    @app.get("/api/review/cases", dependencies=protected)
    async def review_cases(
        status: str = Query("all", pattern="^(all|open|resolved)$"),
        limit: int = Query(100, ge=1, le=500),
        state: AppState = Depends(get_state),
    ):
        cases = await state.aux(current_cases, state)
        open_count = sum(1 for c in cases if c["status"] == "open")
        if status != "all":
            cases = [c for c in cases if c["status"] == status]
        return {"cases": [public_case(c) for c in cases[:limit]], "open": open_count, "total": len(cases)}

    @app.post("/api/review/clear", dependencies=protected)
    async def review_clear(state: AppState = Depends(get_state)):
        """Dismiss every open case. Nothing is deleted from the audit log: the
        clearing is itself recorded there, and the cases' face crops are dropped."""
        cleared = sum(1 for c in await state.aux(current_cases, state) if c["status"] == "open")
        if cleared:
            await state.aux(admin_event, state, review_queue.QUEUE_CLEARED, None, f"cases={cleared}")
            await state.aux(current_cases, state)  # drops the dismissed cases' snapshots
        return {"cleared": cleared}

    @app.get("/api/review/frames/{seq}", dependencies=protected)
    async def review_frame(seq: int, state: AppState = Depends(get_state)):
        jpeg = await state.aux(state.vault.snapshot, seq)
        if jpeg is None:
            raise HTTPException(404, detail="no snapshot for this frame (it may have been reviewed already)")
        return Response(jpeg, media_type="image/jpeg")

    @app.post("/api/review/cases/{case_id}", dependencies=protected)
    async def review_decide(case_id: int, body: ReviewDecision, state: AppState = Depends(get_state)):
        entries = await state.aux(state.audit.entries, REVIEW_WINDOW, True)
        banned = await state.aux(state.vault.banned)
        case = next((c for c in review_queue.build_cases(entries, banned) if c["case_id"] == case_id), None)
        if case is None:
            raise HTTPException(404, detail="review case not found")
        if case["status"] == "resolved":
            raise HTTPException(409, detail="this case has already been reviewed")
        if case["status"] != "open":
            raise HTTPException(409, detail="this case was cleared from the queue")
        subject = None
        if body.outcome == "same_person":
            try:
                subject = normalize_subject_name(body.subject or case["candidate"] or "")
            except ValueError as exc:
                raise HTTPException(422, detail="name the person the reviewer confirmed") from exc
        entry = await state.aux(lambda: state.audit.record(
            mode="review", decision_zone=review_queue.OUTCOMES[body.outcome],
            abstention_reason=review_queue.case_ref(case_id), quality_score=None, liveness_score=None,
            top1_subject=subject, top1_score=None, top2_score=None, margin=None,
            thresholds=state.thresholds.get().to_dict(), engine="human", source="api:review",
            claimed_subject=case["candidate"],
        ))
        case.update(status="resolved", review={
            "outcome": body.outcome, "subject": subject, "timestamp": entry["timestamp"],
            "seq": entry["seq"], "hash": entry["hash"],
        })
        # Decided: the face crops behind the case are no longer needed.
        await state.aux(state.vault.delete_snapshots, case["seqs"])
        return public_case(case)

    @app.get("/api/audit/verify", dependencies=protected)
    async def audit_verify(state: AppState = Depends(get_state)):
        return await state.aux(state.audit.verify)

    @app.get("/api/audit/export", dependencies=protected)
    async def audit_export(format: str = Query("json", pattern="^(json|csv)$"), state: AppState = Depends(get_state)):
        if format == "csv":
            body = await state.aux(state.audit.export_csv)
            return PlainTextResponse(body, media_type="text/csv", headers={
                "Content-Disposition": 'attachment; filename="audit-log.csv"'})
        body = await state.aux(state.audit.export_json)
        return Response(body, media_type="application/json", headers={
            "Content-Disposition": 'attachment; filename="audit-log.json"'})

    # -- fairness & stress test ------------------------------------------- #

    @app.get("/api/fairness/audit", dependencies=protected)
    async def fairness_audit(
        tau_high: float | None = Query(None, ge=0, le=1),
        tau_low: float | None = Query(None, ge=0, le=1),
        margin: float | None = Query(None, ge=0, le=1),
        q_min: float | None = Query(None, ge=0, le=1),
        baseline_threshold: float | None = Query(None, ge=-1, le=1),
        state: AppState = Depends(get_state),
    ):
        thresholds = thresholds_from_query(state, tau_high=tau_high, tau_low=tau_low, margin=margin, q_min=q_min)
        benchmark = await state.get_benchmark()
        return await state.aux(benchmark.evaluate, thresholds, baseline_threshold)

    @app.get("/api/fairness/rfw", dependencies=protected)
    async def fairness_rfw(
        target_fmr: float = Query(1e-2, ge=1e-3, le=0.2),
        cosine_threshold: float | None = Query(None, ge=-1, le=1),
        state: AppState = Depends(get_state),
    ):
        from backend.fairness.audit import load_rfw_scores, rfw_report

        data = await state.aux(load_rfw_scores)
        if data is None:
            return {"available": False, "detail": "RFW score files were not found next to this project"}
        return {"available": True, **await state.aux(rfw_report, data, target_fmr, cosine_threshold)}

    @app.post("/api/stress/lookalike", dependencies=protected)
    async def stress_lookalike(body: LookalikeRequest, state: AppState = Depends(get_state)):
        thresholds = thresholds_from_query(
            state, tau_high=body.tau_high, tau_low=body.tau_low, margin=body.margin, q_min=body.q_min
        )
        benchmark = await state.get_benchmark()
        try:
            return await state.aux(
                benchmark.simulate_lookalike, body.similarity, body.quality, body.both_enrolled,
                thresholds, body.group, body.trials,
            )
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    # -- WebRTC rooms ------------------------------------------------------ #

    def prune_rooms(state: AppState) -> None:
        now = time.time()
        for room_id in [r.room_id for r in state.rooms.values()
                        if not r.peers and now - r.last_active > ROOM_IDLE_SECONDS]:
            state.rooms.pop(room_id, None)

    @app.post("/api/rooms", dependencies=protected)
    async def create_room(body: RoomRequest, request: Request, state: AppState = Depends(get_state)):
        prune_rooms(state)
        if len(state.rooms) >= MAX_ROOMS:
            raise HTTPException(429, detail="too many open rooms")
        base = settings.public_base_url or body.base_url or str(request.base_url).rstrip("/")
        parts = urlsplit(base)
        if parts.scheme not in ("http", "https") or not parts.netloc:
            raise HTTPException(422, detail="base_url must be an http(s) origin")
        origin = f"{parts.scheme}://{parts.netloc}"
        warnings = []
        host = parts.hostname or ""
        alternatives = []
        if _is_loopback_host(host):
            port = f":{parts.port}" if parts.port else ""
            if is_loopback(settings.host):
                # Bound to loopback: no other device can reach this server at all.
                warnings.append(
                    "The server only listens on this machine, so this link cannot be opened from "
                    "another device. Restart with `python run.py --https` to accept participants "
                    "from your network."
                )
            else:
                alternatives = [f"{parts.scheme}://{ip}{port}" for ip in await state.aux(lan_addresses)]
                warnings.append(
                    "This link points at localhost and only works on this machine. Share one of the "
                    "network addresses below instead."
                )
        if parts.scheme == "http" and not _is_loopback_host(host):
            warnings.append(
                "Browsers only grant camera access on HTTPS (or localhost). Participants opening this "
                "http:// link on another device will not be able to share their camera. Start the "
                "server with --https or put it behind a TLS proxy."
            )
        room_id = secrets.token_urlsafe(9)
        now = time.time()
        host_token = secrets.token_urlsafe(18)
        state.rooms[room_id] = Room(room_id, now, now, host_token=host_token)
        join_url = f"{origin}/?room={room_id}"
        return {
            "room_id": room_id,
            "host_token": host_token,
            "join_url": join_url,
            "qr_svg": await state.aux(qr_svg, join_url),
            "warnings": warnings,
            "lan_origins": alternatives,
            "ice_servers": list(settings.ice_servers),
        }

    @app.get("/api/rooms/{room_id}")
    async def room_status(room_id: str, state: AppState = Depends(get_state)):
        room = state.rooms.get(room_id)
        if room is None:
            raise HTTPException(404, detail="room not found or expired")
        return {"room_id": room_id, "peers": sorted(room.peers)}

    @app.delete("/api/rooms/{room_id}", dependencies=protected)
    async def close_room(room_id: str, state: AppState = Depends(get_state)):
        """End a call: revoke the join link and disconnect anyone still in the room."""
        room = state.rooms.pop(room_id, None)
        if room is None:
            raise HTTPException(404, detail="room not found or expired")
        for peer in list(room.peers.values()):
            try:
                await peer.send_json({"type": "error", "code": "room-closed"})
                await peer.close(code=4410)
            except (WebSocketDisconnect, RuntimeError):
                pass
        return {"room_id": room_id, "closed": True}

    @app.websocket("/ws/room/{room_id}")
    async def room_socket(websocket: WebSocket, room_id: str, role: str = "guest", token: str = ""):
        state: AppState = websocket.app.state.ffg
        if not _same_origin(websocket):
            await websocket.close(code=4403)
            return
        room = state.rooms.get(room_id)
        if room is None or role not in ("host", "guest"):
            await websocket.accept()
            await websocket.send_json({"type": "error", "code": "room-not-found"})
            await websocket.close(code=4404)
            return
        await websocket.accept()
        if role == "host" and not keys_equal(token, room.host_token):
            # Knowing the room id (it is in the join link) must not be enough
            # to take the host seat and receive the guest's video.
            await websocket.send_json({"type": "error", "code": "not-host"})
            await websocket.close(code=4403)
            return
        if role in room.peers:
            await websocket.send_json({"type": "error", "code": "room-full"})
            await websocket.close(code=4409)
            return
        other_role = "guest" if role == "host" else "host"
        room.peers[role] = websocket
        room.last_active = time.time()
        try:
            await websocket.send_json({"type": "joined", "role": role, "peer_present": other_role in room.peers})
            other = room.peers.get(other_role)
            if other is not None:
                await other.send_json({"type": "peer-joined", "role": role})
            while True:
                raw = await websocket.receive_text()
                if len(raw) > MAX_SIGNAL_BYTES:
                    await websocket.send_json({"type": "error", "code": "message-too-large"})
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "code": "invalid-json"})
                    continue
                if not isinstance(message, dict) or message.get("type") not in SIGNAL_TYPES:
                    await websocket.send_json({"type": "error", "code": "unsupported-message"})
                    continue
                room.last_active = time.time()
                other = room.peers.get(other_role)
                if other is None:
                    await websocket.send_json({"type": "error", "code": "peer-absent"})
                    continue
                message["from"] = role
                try:
                    await other.send_json(message)
                except (WebSocketDisconnect, RuntimeError):
                    await websocket.send_json({"type": "error", "code": "peer-absent"})
        except WebSocketDisconnect:
            pass
        finally:
            if room.peers.get(role) is websocket:
                room.peers.pop(role, None)
            room.last_active = time.time()
            other = room.peers.get(other_role)
            if other is not None:
                try:
                    await other.send_json({"type": "peer-left", "role": role})
                except (WebSocketDisconnect, RuntimeError):
                    pass

    # -- live camera ------------------------------------------------------- #

    @app.websocket("/ws/camera")
    async def camera_socket(websocket: WebSocket):
        state: AppState = websocket.app.state.ffg
        if not _same_origin(websocket):
            await websocket.close(code=4403)
            return
        await websocket.accept()
        if settings.api_key:
            # The key travels in the first message, never in the URL, so it
            # cannot end up in access logs or browser history.
            try:
                first = await asyncio.wait_for(websocket.receive(), timeout=10.0)
                supplied = json.loads(first.get("text") or "{}")
                authorised = (
                    isinstance(supplied, dict) and supplied.get("type") == "auth"
                    and isinstance(supplied.get("api_key"), str)
                    and keys_equal(supplied["api_key"], settings.api_key)
                )
            except (asyncio.TimeoutError, json.JSONDecodeError, WebSocketDisconnect, RuntimeError):
                authorised = False
            if not authorised:
                try:
                    await websocket.send_json({"type": "error", "code": "unauthorized"})
                    await websocket.close(code=4401)
                except (WebSocketDisconnect, RuntimeError):
                    pass
                return
        session = {"mode": Mode.IDENTIFY, "claimed": None, "compare": False}
        # Latest-frame-wins slots: a frame that arrives while the previous one
        # is still being analysed replaces it, so latency never accumulates.
        pending: dict[int, bytes] = {}
        dropped = {0: 0, 1: 0}
        sequence = {0: 0, 1: 0}
        wake = asyncio.Event()
        throttle = StreamAuditThrottle(settings.stream_audit_interval_s)
        closed = False

        async def worker():
            turn = 0
            while not closed:
                await wake.wait()
                wake.clear()
                while pending and not closed:
                    stream = turn if turn in pending else next(iter(pending))
                    turn = 1 - stream
                    data = pending.pop(stream)
                    sequence[stream] += 1
                    name = STREAM_NAMES[stream]
                    message: dict
                    try:
                        if state.pipeline is None:
                            raise RuntimeError(f"recognition engine unavailable: {state.engine_info.get('error')}")
                        image = await state.aux(
                            decode_image, data, settings.max_upload_bytes, settings.max_frame_pixels)
                        result = await state.infer(
                            state.pipeline.process, image, session["mode"], state.thresholds.get(),
                            session["claimed"], session["compare"],
                        )
                        await state.aux(audit_stream_frame, state, result, name, throttle, image)
                        del image
                        message = {"type": "result", "stream": name, "seq": sequence[stream],
                                   "dropped": dropped[stream], **result.to_dict()}
                    except UnknownSubject:
                        message = {"type": "error", "stream": name, "code": "unknown-subject",
                                   "message": "claimed subject is not enrolled"}
                    except IncompleteTemplate as exc:
                        message = {"type": "error", "stream": name, "code": "incomplete-template", "message": str(exc)}
                    except ImageError as exc:
                        message = {"type": "error", "stream": name, "code": "bad-frame", "message": str(exc)}
                    except Exception as exc:
                        logger.exception("frame processing failed")
                        message = {"type": "error", "stream": name, "code": "processing-failed",
                                   "message": f"{type(exc).__name__}: {exc}"}
                    await websocket.send_json(message)

        task = asyncio.create_task(worker())
        try:
            await websocket.send_json({
                "type": "ready",
                "engine_ready": state.pipeline is not None,
                "thresholds": state.thresholds.get().to_dict(),
            })
            while True:
                incoming = await websocket.receive()
                if incoming["type"] == "websocket.disconnect":
                    break
                if incoming.get("bytes") is not None:
                    frame = incoming["bytes"]
                    if len(frame) < 2 or frame[0] not in STREAM_NAMES:
                        await websocket.send_json({"type": "error", "code": "bad-frame",
                                                   "message": "frames are one stream byte (0 or 1) followed by an encoded image"})
                        continue
                    stream = frame[0]
                    if stream in pending:
                        dropped[stream] += 1
                    pending[stream] = frame[1:]
                    wake.set()
                elif incoming.get("text") is not None:
                    try:
                        message = json.loads(incoming["text"])
                        if isinstance(message, dict) and message.get("type") == "auth":
                            continue  # harmless when no key is configured
                        if message.get("type") != "config":
                            raise ValueError("unsupported message type")
                        mode = Mode(message.get("mode", session["mode"].value))
                        claimed = message.get("claimed_subject")
                        claimed = normalize_subject_name(claimed) if claimed else None
                        if mode is Mode.VERIFY and not claimed:
                            raise ValueError("verify mode requires claimed_subject")
                        session.update(mode=mode, claimed=claimed, compare=bool(message.get("compare_baseline", False)))
                        await websocket.send_json({"type": "config", "mode": mode.value, "claimed_subject": claimed,
                                                   "compare_baseline": session["compare"]})
                    except (ValueError, AttributeError, json.JSONDecodeError) as exc:
                        await websocket.send_json({"type": "error", "code": "bad-config", "message": str(exc)})
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            closed = True
            wake.set()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    # -- server-side camera (optional) ------------------------------------ #

    stream_tokens: dict[str, float] = {}

    def require_key_or_stream_token(request: Request, token: str = "") -> None:
        # <img> cannot send headers. Instead of putting the API key in a URL
        # (where logs would capture it), the preview takes a single-purpose
        # token that expires after a minute.
        if not settings.api_key:
            return
        if keys_equal(request.headers.get("x-api-key"), settings.api_key):
            return
        expiry = stream_tokens.pop(token, 0.0) if token else 0.0
        if expiry < time.time():
            raise HTTPException(401, detail="missing or expired stream token")

    @app.post("/api/server-camera/stream-token", dependencies=protected)
    async def server_camera_stream_token():
        now = time.time()
        for stale in [t for t, expiry in stream_tokens.items() if expiry < now]:
            stream_tokens.pop(stale, None)
        token = secrets.token_urlsafe(24)
        stream_tokens[token] = now + 60.0
        return {"token": token, "expires_in": 60}

    def camera_status(state: AppState) -> dict:
        configured = parse_source(settings.camera_source) is not None
        status = state.server_camera.status() if state.server_camera else {"running": False, "state": "stopped"}
        return {"configured": configured, **status}

    @app.get("/api/server-camera", dependencies=protected)
    async def server_camera_status(state: AppState = Depends(get_state)):
        return camera_status(state)

    @app.post("/api/server-camera/start", dependencies=protected)
    async def server_camera_start(body: CameraStartRequest, state: AppState = Depends(get_state)):
        source = parse_source(settings.camera_source)
        if source is None:
            raise HTTPException(409, detail="no server camera configured; set FAIRFACE_CAMERA_SOURCE")
        pipeline = state.require_pipeline()
        claimed = None
        if body.mode is Mode.VERIFY:
            try:
                claimed = normalize_subject_name(body.claimed_subject or "")
            except ValueError as exc:
                raise HTTPException(422, detail=f"verify mode requires claimed_subject ({exc})") from exc
            if (await state.aux(state.vault.gallery)).index_of(claimed) is None:
                raise HTTPException(404, detail="claimed subject is not enrolled")
        state.server_camera_session.update(mode=body.mode, claimed=claimed, compare=body.compare_baseline)
        if state.server_camera is None:
            throttle = StreamAuditThrottle(settings.stream_audit_interval_s)

            def analyse(frame: np.ndarray) -> dict:
                session = state.server_camera_session
                try:
                    result = pipeline.process(
                        frame, session["mode"], state.thresholds.get(), session["claimed"], session["compare"]
                    )
                except UnknownSubject:
                    return {"type": "error", "stream": "server", "code": "unknown-subject",
                            "message": "claimed subject is no longer enrolled"}
                except IncompleteTemplate as exc:
                    return {"type": "error", "stream": "server", "code": "incomplete-template", "message": str(exc)}
                audit_stream_frame(state, result, "server", throttle, frame)
                return {"type": "result", "stream": "server", **result.to_dict()}

            state.server_camera = ServerCamera(source, analyse)
        await state.aux(state.server_camera.start)
        return camera_status(state)

    @app.post("/api/server-camera/stop", dependencies=protected)
    async def server_camera_stop(state: AppState = Depends(get_state)):
        if state.server_camera is not None:
            await state.aux(state.server_camera.stop)
        return camera_status(state)

    @app.get("/api/server-camera/result", dependencies=protected)
    async def server_camera_result(state: AppState = Depends(get_state)):
        camera = state.server_camera
        return {"status": camera_status(state), "result": camera.latest_result() if camera else None}

    @app.get("/api/server-camera/stream.mjpg", dependencies=[Depends(require_key_or_stream_token)])
    def server_camera_stream(state: AppState = Depends(get_state)):
        camera = state.server_camera
        if camera is None or not camera.running:
            raise HTTPException(409, detail="server camera is not running")

        def frames():
            last = 0
            while camera.running:
                last, jpeg = camera.next_preview(last)
                if jpeg is None:
                    continue
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    # -- static dashboard -------------------------------------------------- #

    if config.FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")

    return app


app = create_app()
