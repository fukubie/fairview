"""Model provisioning: local discovery first, verified download second.

For every required network the manager

1. looks in the project's ``models/`` directory;
2. looks in well-known local caches (``~/.insightface``, ``~/.deepface``,
   sibling checkouts) and reuses any file whose SHA-256 matches the registry;
3. only then downloads it, with retries, resume and SHA-256 verification into
   a temporary file that is atomically renamed on success.

A registry file whose digest does not match is never loaded: ONNX models are
executable graphs, and a corrupted or substituted file must fail closed. The
one exception is an explicit operator override (``FAIRFACE_MODEL_<ROLE>``),
which is loaded as given and reported as unverified.

Licences are part of the registry (see ATTRIBUTIONS.md). The registry is the
single place that maps a role (detector, ArcFace, AdaFace, liveness) to a
concrete artefact, so swapping in commercially licensed weights is a matter of
editing one entry or setting ``FAIRFACE_MODEL_<ROLE>`` to a local path.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from backend import config
from backend.config import Settings

USER_AGENT = "fairface-guard-model-fetch/1.0"
CHUNK = 1 << 20
DOWNLOAD_ATTEMPTS = 4
MAX_ARCHIVE_BYTES = 400 * 1024 * 1024  # ceiling for zip fallbacks, whose size is not in the registry
CONNECT_TIMEOUT_S = 30

ProgressFn = Callable[..., None]


class ModelProvisioningError(RuntimeError):
    """A required model could not be found, downloaded or verified."""


@dataclass(frozen=True)
class ModelSpec:
    role: str  # detector | arcface | adaface | liveness_v2 | liveness_v1se
    filename: str  # name under models/
    sha256: str
    size: int  # bytes, for progress and a cheap pre-check
    urls: tuple[str, ...]  # tried in order
    license: str
    commercial_use: bool
    source: str  # upstream project / page
    description: str
    # Member to extract when a URL points at a zip archive instead of the file.
    archive_member: str | None = None
    # Other file names this artefact is known by in local caches.
    aliases: tuple[str, ...] = ()

    @property
    def env_override(self) -> str:
        return f"FAIRFACE_MODEL_{self.role.upper()}"


# Filled in below; kept as a function so the registry is defined after the
# dataclass and can be extended by deployments without editing this module.
REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> None:
    REGISTRY[spec.role] = spec


# --------------------------------------------------------------------------- #
# Hashing and discovery
# --------------------------------------------------------------------------- #


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _matches(path: Path, spec: ModelSpec) -> bool:
    try:
        return path.is_file() and path.stat().st_size == spec.size and sha256_file(path) == spec.sha256
    except OSError:
        return False


def local_search_dirs() -> list[Path]:
    """Places where compatible weights may already exist on this machine."""
    home = Path.home()
    candidates = [
        home / ".insightface" / "models",
        home / ".deepface" / "weights",
        home / ".cache" / "fairface-guard",
        config.BASE_DIR.parent / "models",
        config.BASE_DIR.parent / "insightface-master" / "models",
    ]
    extra = os.environ.get("FAIRFACE_MODEL_SEARCH_PATH", "")
    candidates.extend(Path(p).expanduser() for p in extra.split(os.pathsep) if p)
    return [c for c in candidates if c.is_dir()]


def find_local(spec: ModelSpec, search_dirs: list[Path] | None = None) -> Path | None:
    """Locate an already-present copy of ``spec`` (by name, verified by hash)."""
    names = {spec.filename, *spec.aliases}
    if spec.archive_member:
        names.add(Path(spec.archive_member).name)
    for root in search_dirs if search_dirs is not None else local_search_dirs():
        try:
            # Caches are shallow (models/<pack>/<file>); bound the walk.
            for path in root.glob("**/*"):
                if len(path.relative_to(root).parts) > 3:
                    continue
                if path.name in names and _matches(path, spec):
                    return path
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #


def _open(url: str, offset: int):
    if not url.lower().startswith("https://"):
        raise ModelProvisioningError(f"refusing non-HTTPS model URL: {url}")
    headers = {"User-Agent": USER_AGENT}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=CONNECT_TIMEOUT_S)


def _download_to(url: str, part: Path, expected_size: int | None, progress: ProgressFn, name: str) -> None:
    """Fetch ``url`` into ``part``, resuming a previous partial transfer."""
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        offset = part.stat().st_size if part.exists() else 0
        if expected_size and offset > expected_size:
            part.unlink()
            offset = 0
        if expected_size and offset == expected_size:
            return
        try:
            with _open(url, offset) as response:
                status = getattr(response, "status", 200)
                if offset and status != 206:
                    offset = 0  # server ignored the Range header: start over
                mode = "ab" if offset else "wb"
                total = expected_size or (offset + int(response.headers.get("Content-Length") or 0)) or None
                done, last_report = offset, time.monotonic()
                limit = expected_size or MAX_ARCHIVE_BYTES
                with open(part, mode) as handle:
                    while chunk := response.read(CHUNK):
                        done += len(chunk)
                        if done > limit:
                            # A mirror streaming more than the artefact's known
                            # size is serving something else; stop writing.
                            raise ModelProvisioningError(
                                f"{url} sent more than the expected {limit} bytes"
                            )
                        handle.write(chunk)
                        now = time.monotonic()
                        if total and now - last_report > 2.0:
                            progress("downloading", name, f"{done * 100 // total}% of {total / 1e6:.0f} MB")
                            last_report = now
            if expected_size and done < expected_size:
                # Connection dropped mid-body: keep the partial file and let
                # the next attempt resume it with a Range request.
                raise ConnectionError(f"transfer ended at {done} of {expected_size} bytes")
            return
        except ModelProvisioningError:
            part.unlink(missing_ok=True)
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():  # stale partial file
                part.unlink()
            last_error = exc
            if exc.code in (401, 403, 404, 410):
                break  # retrying will not help
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
        if attempt < DOWNLOAD_ATTEMPTS:
            delay = 2 ** attempt
            progress("retrying", name, f"in {delay}s ({type(last_error).__name__}: {last_error})")
            time.sleep(delay)
    raise ModelProvisioningError(f"download failed for {url}: {type(last_error).__name__}: {last_error}")


def _extract_member(archive: Path, member: str, destination: Path, expected_size: int) -> None:
    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        names = {Path(n).name: n for n in bundle.namelist() if not n.endswith("/")}
        target = member if member in bundle.namelist() else names.get(Path(member).name)
        if target is None:
            raise ModelProvisioningError(f"{member} not found inside {archive.name}")
        if bundle.getinfo(target).file_size != expected_size:
            raise ModelProvisioningError(f"{member} inside {archive.name} has an unexpected size")
        with bundle.open(target) as source, open(destination, "wb") as out:
            remaining = expected_size  # never trust the header alone (zip bombs)
            while remaining > 0 and (chunk := source.read(min(CHUNK, remaining))):
                out.write(chunk)
                remaining -= len(chunk)


def download(spec: ModelSpec, destination: Path, progress: ProgressFn) -> None:
    """Download ``spec`` to ``destination`` and verify it. Atomic on success."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in spec.urls:
        is_archive = spec.archive_member is not None and url.lower().split("?")[0].endswith(".zip")
        part = destination.with_name(destination.name + (".zip.part" if is_archive else ".part"))
        candidate = destination.with_name(destination.name + ".verify")
        try:
            progress("downloading", spec.role, url)
            _download_to(url, part, None if is_archive else spec.size, progress, spec.role)
            if is_archive:
                _extract_member(part, spec.archive_member, candidate, spec.size)
                part.unlink(missing_ok=True)
            else:
                part.replace(candidate)
            actual = sha256_file(candidate)
            if actual != spec.sha256:
                candidate.unlink(missing_ok=True)
                raise ModelProvisioningError(
                    f"SHA-256 mismatch for {spec.filename} from {url}: expected {spec.sha256}, got {actual}"
                )
            candidate.replace(destination)
            return
        except (ModelProvisioningError, OSError) as exc:
            errors.append(str(exc))
            # A partial file is only meaningful for the mirror that produced
            # it: never let the next URL resume from another server's bytes.
            for leftover in (candidate, part):
                leftover.unlink(missing_ok=True)
    raise ModelProvisioningError(
        f"could not provision the {spec.role} model ({spec.filename}).\n  "
        + "\n  ".join(errors)
        + f"\nPlace a verified copy at {destination} or point {spec.env_override} at one."
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def _noop_progress(*_args, **_kwargs) -> None:
    return None


def resolve(spec: ModelSpec, models_dir: Path, progress: ProgressFn, offline: bool) -> Path:
    # 0. Explicit operator override: used as-is (the operator vouches for it).
    override = os.environ.get(spec.env_override)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise ModelProvisioningError(f"{spec.env_override}={override} does not exist")
        progress("override", spec.role, str(path))
        return path
    # 1. Project models directory.
    target = models_dir / spec.filename
    if target.is_file():
        if _matches(target, spec):
            progress("present", spec.role, spec.filename)
            return target
        progress("corrupt", spec.role, f"{spec.filename} failed SHA-256 verification; replacing")
        target.unlink()
    # 2. Other local installations.
    found = find_local(spec)
    if found is not None:
        progress("reusing", spec.role, str(found))
        return found
    # 3. Network.
    if offline:
        raise ModelProvisioningError(
            f"{spec.filename} is missing and downloads are disabled. Expected at {target} "
            f"(SHA-256 {spec.sha256}); sources: {', '.join(spec.urls)}"
        )
    download(spec, target, progress)
    progress("verified", spec.role, spec.filename)
    return target


def ensure_models(
    models_dir: Path | None = None,
    progress: ProgressFn | None = None,
    offline: bool = False,
    roles: list[str] | None = None,
) -> dict[str, Path]:
    """Make every required model available; returns ``{role: path}``."""
    models_dir = models_dir or config.models_dir()
    progress = progress or _noop_progress
    paths = {}
    for role in roles or list(REGISTRY):
        paths[role] = resolve(REGISTRY[role], models_dir, progress, offline)
    return paths


def create_session(model: Path | bytes, settings: Settings, providers: list[str] | None = None):
    """Create an ONNX Runtime session, degrading to CPU if an accelerator fails.

    ``model`` is a file path or the serialised model itself.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    # Five sessions share the machine. By default every session's worker
    # threads busy-wait for new work, so the idle pools of four networks steal
    # cores from the one that is running (measured: 2.5x slower per frame on a
    # 2-core laptop CPU). Sleep instead of spin.
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    if settings.onnx_threads > 0:
        options.intra_op_num_threads = settings.onnx_threads
    providers = providers or config.select_providers()
    source = model if isinstance(model, bytes) else str(model)
    try:
        return ort.InferenceSession(source, sess_options=options, providers=providers)
    except Exception:
        if providers == ["CPUExecutionProvider"]:
            raise
        # Missing CUDA/cuDNN libraries, unsupported operator on CoreML, ...
        return ort.InferenceSession(source, sess_options=options, providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
# Every URL below was downloaded and its digest checked when the registry was
# compiled (zip fallbacks: member digests checked). Hugging Face URLs are
# pinned to a commit; GitHub release assets are mutable by their owners, which
# is why the SHA-256 is enforced on every load path.

_INSIGHTFACE_LICENSE = (
    "InsightFace pretrained models: non-commercial research purposes only "
    "(insightface README / model_zoo README). Code is MIT; the weights are not."
)

register(ModelSpec(
    role="detector",
    filename="scrfd_500m_kps.onnx",
    sha256="5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
    size=2_524_817,
    urls=(
        "https://huggingface.co/immich-app/buffalo_s/resolve/0ff1751885575e62e084dff70549ce24a11fa5dc/detection/model.onnx",
        "https://github.com/yakhyo/uniface/releases/download/weights/scrfd_500m_kps.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
    ),
    archive_member="det_500m.onnx",
    aliases=("det_500m.onnx",),
    license=_INSIGHTFACE_LICENSE,
    commercial_use=False,
    source="deepinsight/insightface (buffalo_s: SCRFD-500MF with 5 key points)",
    description="SCRFD face detector with 5 landmarks",
))

register(ModelSpec(
    role="arcface",
    filename="arcface_w600k_mbf.onnx",
    sha256="9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f",
    size=13_616_099,
    urls=(
        "https://huggingface.co/immich-app/buffalo_s/resolve/0ff1751885575e62e084dff70549ce24a11fa5dc/recognition/model.onnx",
        "https://github.com/yakhyo/face-reidentification/releases/download/v0.0.1/w600k_mbf.onnx",
        "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
    ),
    archive_member="w600k_mbf.onnx",
    aliases=("w600k_mbf.onnx",),
    license=_INSIGHTFACE_LICENSE + " Trained on WebFace600K.",
    commercial_use=False,
    source="deepinsight/insightface (buffalo_s: ArcFace MobileFaceNet, WebFace600K)",
    description="ArcFace 512-D embedding (RGB input)",
))

register(ModelSpec(
    role="adaface",
    filename="adaface_ir_18.onnx",
    sha256="6b6a35772fb636cdd4fa86520c1a259d0c41472a76f70f802b351837a00d9870",
    size=96_105_345,
    urls=("https://github.com/yakhyo/adaface-onnx/releases/download/weights/adaface_ir_18.onnx",),
    license=(
        "No licence is stated for AdaFace weights (code is MIT). Trained on WebFace4M, which is "
        "distributed for research under a licence agreement. Treat as research-only."
    ),
    commercial_use=False,
    source="mk-minchul/AdaFace IR-18 WebFace4M, ONNX export by yakhyo/adaface-onnx",
    description="AdaFace 512-D embedding (BGR input); feature norm is the quality proxy",
))

register(ModelSpec(
    role="liveness_v2",
    filename="MiniFASNetV2.onnx",
    sha256="b32929adc2d9c34b9486f8c4c7bc97c1b69bc0ea9befefc380e4faae4e463907",
    size=1_743_581,
    urls=("https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV2.onnx",),
    license="Apache-2.0 (minivision-ai/Silent-Face-Anti-Spoofing; training data undisclosed)",
    commercial_use=True,
    source="minivision-ai/Silent-Face-Anti-Spoofing 2.7_80x80_MiniFASNetV2, ONNX export by yakhyo",
    description="Passive liveness, 2.7x context crop",
))

register(ModelSpec(
    role="liveness_v1se",
    filename="MiniFASNetV1SE.onnx",
    sha256="ebab7f90c7833fbccd46d3a555410e78d969db5438e169b6524be444862b3676",
    size=1_742_335,
    urls=("https://github.com/yakhyo/face-anti-spoofing/releases/download/weights/MiniFASNetV1SE.onnx",),
    license="Apache-2.0 (minivision-ai/Silent-Face-Anti-Spoofing; training data undisclosed)",
    commercial_use=True,
    source="minivision-ai/Silent-Face-Anti-Spoofing 4_0_0_80x80_MiniFASNetV1SE, ONNX export by yakhyo",
    description="Passive liveness, 4x context crop",
))


# --------------------------------------------------------------------------- #
# AdaFace: expose the pre-normalisation feature norm
# --------------------------------------------------------------------------- #

ADAFACE_NORM_OUTPUT = "feat_norm"


def adaface_with_norm(source: Path) -> bytes:
    """Return the AdaFace graph with the feature norm as a second output.

    Published AdaFace ONNX exports end in ``Div(z, ReduceL2(z))`` and return
    only the unit vector, discarding ``||z||`` - the very signal used here as
    the image-quality proxy. ONNX Runtime cannot fetch intermediate tensors,
    so the graph is edited (weights untouched) to also output the ``ReduceL2``
    result.

    The edit is done in memory, from the digest-verified source, on every
    start, and the serialised bytes go straight to ONNX Runtime. Nothing
    derived is cached on disk, so there is no second, unverified model file
    that could be swapped out.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(str(source))
    graph = model.graph
    if not any(o.name == ADAFACE_NORM_OUTPUT for o in graph.output):
        final_name = graph.output[0].name
        final = [n for n in graph.node if n.op_type == "Div" and n.output[0] == final_name]
        producers = {out: node for node in graph.node for out in node.output}
        if len(final) != 1 or producers.get(final[0].input[1]) is None \
                or producers[final[0].input[1]].op_type != "ReduceL2" \
                or producers[final[0].input[1]].input[0] != final[0].input[0]:
            raise ModelProvisioningError(
                f"{source.name} does not end in Div(z, ReduceL2(z)); cannot expose the AdaFace feature norm"
            )
        batch = graph.output[0].type.tensor_type.shape.dim[0]
        graph.node.append(helper.make_node("Identity", [final[0].input[1]], [ADAFACE_NORM_OUTPUT]))
        graph.output.append(helper.make_tensor_value_info(
            ADAFACE_NORM_OUTPUT, TensorProto.FLOAT, [batch.dim_param or batch.dim_value or "batch", 1]))
        onnx.checker.check_model(model)
    return model.SerializeToString()


# --------------------------------------------------------------------------- #
# Pipeline assembly
# --------------------------------------------------------------------------- #

CALIBRATION_FILE = "calibration.json"
COHORT_FILE = "cohort.npz"


def load_calibration(assets_dir: Path | None = None):
    """Load the shipped calibrator, quality bounds and baseline threshold."""
    import json

    from backend.engine.calibrator import CalibrationParams, ScoreCalibrator
    from backend.engine.embedder import QualityConfig

    assets_dir = assets_dir or config.ASSETS_DIR
    params_path, cohort_path = assets_dir / CALIBRATION_FILE, assets_dir / COHORT_FILE
    if not params_path.is_file():
        # Conservative built-in fallback: global z-score, no subgroup
        # equalisation. describe() reports this mode to the UI and audit trail.
        a, b = ScoreCalibrator.solve_logistic(3.0, 5.5)
        params = CalibrationParams(a=a, b=b, mu0=0.0, sigma0=0.07, fmr_low=None, fmr_high=None,
                                   source="built-in fallback (uncalibrated)")
        return ScoreCalibrator(params), QualityConfig(norm_min=10.0, norm_max=24.0), None
    raw = json.loads(params_path.read_text(encoding="utf-8"))
    calibrator = ScoreCalibrator.load(params_path, cohort_path)
    quality = QualityConfig(**raw.get("quality", {"norm_min": 10.0, "norm_max": 24.0}))
    return calibrator, quality, raw.get("baseline_cosine_threshold")


def build_engine(settings: Settings, progress: ProgressFn | None = None, offline: bool = False):
    """Provision models and create the neural stages (no storage attached)."""
    from backend.engine.detector import FaceDetector
    from backend.engine.embedder import BackboneSpec, FaceEmbedder
    from backend.engine.liveness import LivenessDetector

    models_dir = config.models_dir()
    paths = ensure_models(models_dir, progress=progress, offline=offline)
    adaface_model = adaface_with_norm(paths["adaface"])

    providers = config.select_providers()
    sessions = {
        "detector": create_session(paths["detector"], settings, providers),
        "arcface": create_session(paths["arcface"], settings, providers),
        "adaface": create_session(adaface_model, settings, providers),
    }
    liveness = None
    if settings.liveness_enabled:
        sessions["liveness_v2"] = create_session(paths["liveness_v2"], settings, providers)
        sessions["liveness_v1se"] = create_session(paths["liveness_v1se"], settings, providers)
        liveness = LivenessDetector(
            sessions["liveness_v2"], sessions["liveness_v1se"], layout="NCHW", outputs_probabilities=False
        )
    ada_outputs = [o.name for o in sessions["adaface"].get_outputs()]
    detector = FaceDetector(sessions["detector"], settings.det_thresh, settings.det_nms, settings.det_size)
    embedder = FaceEmbedder(
        sessions["adaface"],
        sessions["arcface"],
        BackboneSpec(channel_order="BGR", embedding_output=0, norm_output=ada_outputs.index(ADAFACE_NORM_OUTPUT)),
        BackboneSpec(channel_order="RGB"),
    )
    def describe(role: str) -> dict:
        spec = REGISTRY[role]
        if os.environ.get(spec.env_override):
            # Operator-supplied file: its licence is whatever the operator
            # holds, and it is not checked against the registry digest.
            return {
                "file": paths[role].name, "description": spec.description,
                "source": f"operator override ({spec.env_override})",
                "license": "supplied by the operator; not known to this software",
                "commercial_use": None, "verified_sha256": False,
                "provider": sessions[role].get_providers()[0],
            }
        return {
            "file": paths[role].name, "description": spec.description, "source": spec.source,
            "license": spec.license, "commercial_use": spec.commercial_use, "verified_sha256": True,
            "provider": sessions[role].get_providers()[0],
        }

    models = {role: describe(role) for role in paths if role in sessions}
    flags = [m["commercial_use"] for m in models.values()]
    info = {
        "models": models,
        "providers": providers,
        # True / False when every licence is known; None when an operator
        # override makes it this software cannot answer.
        "commercial_use_permitted": False if False in flags else (None if None in flags else True),
    }
    return detector, embedder, liveness, info


def build_pipeline(vault, settings: Settings, progress: ProgressFn | None = None, offline: bool = False):
    """Assemble the production pipeline. Returns ``(pipeline, engine_info)``."""
    from dataclasses import replace

    from backend.engine.pipeline import RecognitionPipeline

    detector, embedder, liveness, info = build_engine(settings, progress, offline)
    calibrator, quality, baseline_threshold = load_calibration()
    if baseline_threshold is not None and "FAIRFACE_BASELINE_COSINE" not in os.environ:
        settings = replace(settings, baseline_cosine_threshold=float(baseline_threshold))
    pipeline = RecognitionPipeline(detector, embedder, liveness, calibrator, vault, settings, quality)
    info.update(
        calibration=calibrator.describe(),
        quality={"norm_min": quality.norm_min, "norm_max": quality.norm_max},
        fusion_mode=pipeline.fusion_mode,
        baseline_cosine_threshold=settings.baseline_cosine_threshold,
        liveness_enabled=liveness is not None,
    )
    return pipeline, info
