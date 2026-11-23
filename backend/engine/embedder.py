"""Stage 3: dual-backbone embeddings, intrinsic quality and adaptive fusion.

Two recognition backbones look at the same aligned 112x112 crop:

* **ArcFace** (InsightFace): RGB input, ``(pixel - 127.5) / 127.5``.
* **AdaFace** (Kim et al., CVPR 2022): BGR input, same scaling. AdaFace's
  training couples the feature norm to image quality, so the L2 norm of its
  *pre-normalisation* feature ``||z_ada||`` is a free quality signal: sharp,
  well-lit frontal faces produce long vectors, degraded ones short vectors.

Quality
-------
``q = q_norm^a * q_det^b * q_pose^c`` (weighted geometric mean, weights sum to
1) where ``q_norm`` is ``||z_ada||`` rescaled between calibrated bounds,
``q_det`` the SCRFD confidence rescaled above the detection threshold and
``q_pose`` the yaw/pitch symmetry of the five landmarks. A geometric mean lets
any single collapsed component veto the frame, which an arithmetic mean would
average away.

Fusion
------
Each embedding is L2-normalised and the pair is fused with a quality-adaptive
weight ``w_ada(q)`` that grows as quality falls, then re-normalised::

    e_fused = (w * e_ada (+) (1 - w) * e_arc) / || . ||

The two networks were trained independently, so their 512-D spaces share no
coordinate system. The default ``direct_sum`` mode therefore places them in
orthogonal subspaces (a 1024-D vector). The cosine of two fused vectors is then
exactly a weighted combination of the two per-backbone cosines. Adding the
vectors inside one shared 512-D space (``sum`` mode, the formula read
literally) introduces cross terms ``e_ada(a) . e_arc(b)`` between unrelated
coordinate systems; they carry no identity information and act as noise on
every comparison. ``sum`` remains available as a function so the two can be
compared (see the tests); the application itself always uses ``direct_sum``,
and the shipped cohort and thresholds are specific to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from backend.engine.detector import to_model_input

EMBEDDING_DIM = 512
FUSION_MODES = ("direct_sum", "sum")


@dataclass(frozen=True)
class QualityConfig:
    """Bounds and weights of the intrinsic quality score."""

    # ||z_ada|| at which the norm component is 0 / 1. Measured per AdaFace
    # checkpoint by tools/build_calibration.py (median norm of heavily
    # degraded crops / of clean crops) and shipped in assets/calibration.json.
    norm_min: float = 5.0
    norm_max: float = 25.0
    # SCRFD confidence mapped linearly from [det_floor, det_ceiling] to [0, 1].
    # The floor sits below the detection threshold on purpose: a face that was
    # detected at all is not disqualified by this term alone, it only loses a
    # little. Clean frontal portraits score about 0.8 with the 500MF detector.
    det_floor: float = 0.35
    det_ceiling: float = 0.75
    weight_norm: float = 0.5
    weight_det: float = 0.2
    weight_pose: float = 0.3

    def __post_init__(self) -> None:
        if not self.norm_max > self.norm_min:
            raise ValueError("norm_max must exceed norm_min")
        if not self.det_ceiling > self.det_floor:
            raise ValueError("det_ceiling must exceed det_floor")
        total = self.weight_norm + self.weight_det + self.weight_pose
        if min(self.weight_norm, self.weight_det, self.weight_pose) < 0 or abs(total - 1.0) > 1e-6:
            raise ValueError("quality weights must be non-negative and sum to 1")


@dataclass(frozen=True)
class QualityScore:
    score: float  # q in [0, 1]
    norm_component: float
    detection_component: float
    pose_component: float
    feature_norm: float  # raw ||z_ada||

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "norm": round(self.norm_component, 4),
            "detection": round(self.detection_component, 4),
            "pose": round(self.pose_component, 4),
            "feature_norm": round(self.feature_norm, 3),
        }


def _unit_interval(value: float, low: float, high: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max((value - low) / (high - low), 0.0), 1.0))


def compute_quality(
    feature_norm: float, det_score: float, pose_score: float, config: QualityConfig | None = None
) -> QualityScore:
    """Combine feature norm, detector confidence and pose into ``q in [0, 1]``."""
    config = config or QualityConfig()
    q_norm = _unit_interval(feature_norm, config.norm_min, config.norm_max)
    q_det = _unit_interval(det_score, config.det_floor, config.det_ceiling)
    q_pose = float(min(max(pose_score, 0.0), 1.0)) if np.isfinite(pose_score) else 0.0
    score = (q_norm**config.weight_norm) * (q_det**config.weight_det) * (q_pose**config.weight_pose)
    return QualityScore(
        score=float(score),
        norm_component=q_norm,
        detection_component=q_det,
        pose_component=q_pose,
        feature_norm=float(feature_norm) if np.isfinite(feature_norm) else 0.0,
    )


def fusion_weight(quality, w_min: float = 0.40, w_max: float = 0.70):
    """AdaFace weight ``w_ada(q)``: ``w_min`` at q = 1 rising linearly to ``w_max`` at q = 0."""
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("fusion weights must satisfy 0 <= w_min <= w_max <= 1")
    q = np.clip(np.nan_to_num(np.asarray(quality, dtype=np.float64), nan=0.0), 0.0, 1.0)
    weight = w_min + (w_max - w_min) * (1.0 - q)
    return float(weight) if weight.ndim == 0 else weight


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(unit_vectors, norms)`` along the last axis."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps), norms[..., 0]


def fuse_embeddings(e_ada: np.ndarray, e_arc: np.ndarray, w_ada, mode: str = "direct_sum") -> np.ndarray:
    """Quality-weighted fusion of two embeddings, L2-normalised.

    Accepts single vectors ``(D,)`` or batches ``(N, D)``; ``w_ada`` is a
    scalar or ``(N,)``. Inputs are normalised defensively. Output is ``2 * D``
    wide in ``direct_sum`` mode and ``D`` wide in ``sum`` mode.
    """
    if mode not in FUSION_MODES:
        raise ValueError(f"fusion mode must be one of {FUSION_MODES}")
    ada, _ = l2_normalize(e_ada)
    arc, _ = l2_normalize(e_arc)
    if ada.shape != arc.shape:
        raise ValueError("embeddings must have identical shapes")
    weight = np.asarray(w_ada, dtype=np.float64)
    if np.any(weight < 0) or np.any(weight > 1) or not np.all(np.isfinite(weight)):
        raise ValueError("w_ada must lie in [0, 1]")
    if ada.ndim == 2:
        weight = weight.reshape(-1, 1) if weight.ndim else weight
    if mode == "direct_sum":
        combined = np.concatenate([weight * ada, (1.0 - weight) * arc], axis=-1)
    else:
        combined = weight * ada + (1.0 - weight) * arc
    fused, norms = l2_normalize(combined)
    if np.any(norms < 1e-9):
        raise ValueError("fused embedding collapsed to zero")
    return fused.astype(np.float32)


def arcface_component(fused: np.ndarray, dim: int = EMBEDDING_DIM) -> np.ndarray | None:
    """Recover the unit ArcFace embedding from a ``direct_sum`` fused vector.

    Lets the single-backbone comparison engine reuse the stored templates.
    Returns ``None`` for ``sum``-mode vectors, where the parts are inseparable.
    """
    fused = np.asarray(fused, dtype=np.float64)
    if fused.shape[-1] != 2 * dim:
        return None
    part, norms = l2_normalize(fused[..., dim:])
    if np.any(norms < 1e-9):
        return None
    return part.astype(np.float32)


@dataclass(frozen=True)
class BackboneSpec:
    """How to feed one recognition network and read its output."""

    channel_order: str  # "RGB" or "BGR"
    # Index of the graph output holding the embedding, and (optionally) of an
    # output holding the pre-normalisation norm when the graph normalises
    # internally. With norm_output=None the embedding output is taken to be
    # the raw feature and its norm is computed here.
    embedding_output: int = 0
    norm_output: int | None = None


@dataclass(frozen=True)
class Embedding:
    ada: np.ndarray  # (512,) unit
    arc: np.ndarray  # (512,) unit
    ada_norm: float  # ||z_ada|| before normalisation (quality proxy)
    arc_norm: float


class FaceEmbedder:
    """Runs both ONNX backbones on an aligned crop."""

    def __init__(self, ada_session, arc_session, ada_spec: BackboneSpec, arc_spec: BackboneSpec) -> None:
        self._ada, self._arc = ada_session, arc_session
        self._ada_spec, self._arc_spec = ada_spec, arc_spec
        self._ada_input = ada_session.get_inputs()[0].name
        self._arc_input = arc_session.get_inputs()[0].name

    @staticmethod
    def _run(session, input_name: str, spec: BackboneSpec, aligned_bgr: np.ndarray) -> tuple[np.ndarray, float]:
        outputs = session.run(None, {input_name: to_model_input(aligned_bgr, spec.channel_order)})
        feature = np.asarray(outputs[spec.embedding_output], dtype=np.float64).reshape(-1)
        if feature.size != EMBEDDING_DIM:
            raise ValueError(f"backbone produced a {feature.size}-D feature, expected {EMBEDDING_DIM}")
        if not np.all(np.isfinite(feature)):
            raise ValueError("backbone produced non-finite features")
        unit, norm = l2_normalize(feature)
        if spec.norm_output is not None:
            norm = np.asarray(outputs[spec.norm_output], dtype=np.float64).reshape(-1)[0]
        norm = float(norm)
        if norm < 1e-9:
            raise ValueError("backbone produced a zero feature")
        return unit.astype(np.float32), norm

    def embed(self, aligned_bgr: np.ndarray) -> Embedding:
        if aligned_bgr.shape != (112, 112, 3) or aligned_bgr.dtype != np.uint8:
            raise ValueError("expected a 112x112x3 uint8 aligned crop")
        ada, ada_norm = self._run(self._ada, self._ada_input, self._ada_spec, aligned_bgr)
        arc, arc_norm = self._run(self._arc, self._arc_input, self._arc_spec, aligned_bgr)
        return Embedding(ada=ada, arc=arc, ada_norm=ada_norm, arc_norm=arc_norm)
