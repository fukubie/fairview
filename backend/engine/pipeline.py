"""End-to-end recognition pipeline: frame in, per-face three-zone decisions out.

Orchestrates the five stages (detect/align, liveness, embed + quality, calibrate,
decide) and the enrollment workflow. Frames exist only as in-memory arrays for
the duration of one call; nothing in this module writes pixels anywhere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from backend.config import Settings, Thresholds
from backend.engine.calibrator import ScoreCalibrator
from backend.engine.decision import Candidate, Decision, Mode, Reason, Zone, decide, decide_baseline
from backend.engine.detector import Detection, FaceDetector, align_face, estimate_pose
from backend.engine.embedder import (
    FaceEmbedder,
    QualityConfig,
    QualityScore,
    arcface_component,
    compute_quality,
    fuse_embeddings,
    fusion_weight,
)
from backend.engine.liveness import LivenessDetector
from backend.engine.visitors import VisitorRegistry, is_known_visitor
from backend.storage.vault import BiometricVault, Gallery, build_template

MAX_PROCESSING_SIDE = 1920
TOP_CANDIDATES = 5


class ImageError(ValueError):
    """The payload is not a decodable image within the configured limits."""


def declared_size(data: bytes) -> tuple[int, int] | None:
    """``(width, height)`` declared in a JPEG, PNG, WebP, BMP or GIF header.

    Parsed from the first bytes without decoding anything, so an image that
    claims an enormous canvas can be refused before a single pixel is
    allocated. Returns ``None`` for any other or malformed container.
    """
    n = len(data)
    if n >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if n >= 4 and data[:2] == b"\xff\xd8":
        position = 2
        while position + 4 <= n:
            if data[position] != 0xFF:
                return None
            marker = data[position + 1]
            if marker == 0xFF:  # fill byte
                position += 1
                continue
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # markers without a length
                position += 2
                continue
            length = int.from_bytes(data[position + 2 : position + 4], "big")
            if length < 2:
                return None
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # start of frame
                if position + 9 > n:
                    return None
                height = int.from_bytes(data[position + 5 : position + 7], "big")
                width = int.from_bytes(data[position + 7 : position + 9], "big")
                return width, height
            if marker == 0xDA:  # start of scan before any frame header
                return None
            position += 2 + length
        return None
    if n >= 30 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        chunk = data[12:16]
        if chunk == b"VP8X":
            return int.from_bytes(data[24:27], "little") + 1, int.from_bytes(data[27:30], "little") + 1
        if chunk == b"VP8L" and data[20] == 0x2F:
            bits = int.from_bytes(data[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            return (int.from_bytes(data[26:28], "little") & 0x3FFF,
                    int.from_bytes(data[28:30], "little") & 0x3FFF)
        return None
    if n >= 26 and data[:2] == b"BM":
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = int.from_bytes(data[22:26], "little", signed=True)
        return abs(width), abs(height)
    if n >= 10 and data[:6] in (b"GIF87a", b"GIF89a"):
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    return None


def decode_image(data: bytes, max_bytes: int, max_pixels: int) -> np.ndarray:
    """Decode an encoded image (JPEG, PNG, WebP, BMP, GIF) to BGR uint8, in memory."""
    if not data:
        raise ImageError("empty image payload")
    if len(data) > max_bytes:
        raise ImageError(f"image exceeds the {max_bytes // (1024 * 1024)} MB limit")
    # Refuse by declared size before decoding. This does not rely on OpenCV's
    # own limit, which only applies under specific import-order conditions.
    size = declared_size(data)
    if size is None:
        raise ImageError("unsupported or unreadable image format (use JPEG, PNG, WebP or BMP)")
    if size[0] <= 0 or size[1] <= 0 or size[0] * size[1] > max_pixels:
        raise ImageError("image resolution exceeds the configured limit")
    buffer = np.frombuffer(data, dtype=np.uint8)
    try:
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    except cv2.error as exc:
        raise ImageError("image could not be decoded") from exc
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ImageError("image could not be decoded")
    height, width = image.shape[:2]
    if height * width > max_pixels:
        raise ImageError("image resolution exceeds the configured limit")
    if height < 16 or width < 16:
        raise ImageError("image is too small")
    longest = max(height, width)
    if longest > MAX_PROCESSING_SIDE:
        scale = MAX_PROCESSING_SIDE / longest
        image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    return image


@dataclass
class FaceObservation:
    """Everything measured about one detected face."""

    detection: Detection
    primary: bool
    quality: QualityScore
    liveness: float | None  # None: stage disabled. NaN: enabled but not assessable.
    fused: np.ndarray | None  # unit fused embedding, None if the face could not be embedded
    arc: np.ndarray | None  # unit ArcFace embedding (comparison engine)
    w_ada: float | None
    yaw_offset: float = 0.0
    pitch_offset: float = 0.0
    error: str | None = None


@dataclass
class FaceResult:
    observation: FaceObservation
    decision: Decision
    candidates: list[Candidate]
    baseline: Decision | None = None
    # In-memory label for a face that matches nobody enrolled ("Visitor 3").
    visitor: int | None = None

    def to_dict(self) -> dict:
        obs = self.observation
        det = obs.detection
        liveness = obs.liveness
        payload = {
            "bbox": [round(float(v), 1) for v in det.bbox],
            "landmarks": [[round(float(x), 1), round(float(y), 1)] for x, y in det.landmarks],
            "det_score": round(det.score, 4),
            "primary": obs.primary,
            "quality": obs.quality.to_dict(),
            "pose": {"yaw_offset": round(obs.yaw_offset, 3), "pitch_offset": round(obs.pitch_offset, 3)},
            "liveness": {
                "enabled": liveness is not None,
                "score": None if liveness is None or not np.isfinite(liveness) else round(float(liveness), 4),
            },
            "w_ada": None if obs.w_ada is None else round(obs.w_ada, 3),
            "decision": self.decision.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates[:TOP_CANDIDATES]],
            "error": obs.error,
            "visitor": self.visitor,
        }
        if self.visitor is not None:
            payload["decision"]["visitor"] = self.visitor
        if self.baseline is not None:
            payload["baseline"] = self.baseline.to_dict()
        return payload


@dataclass
class FrameResult:
    width: int
    height: int
    mode: Mode
    faces: list[FaceResult]
    timing_ms: dict[str, float] = field(default_factory=dict)
    claimed_subject: str | None = None

    def to_dict(self) -> dict:
        return {
            "frame": {"width": self.width, "height": self.height},
            "mode": self.mode.value,
            "claimed_subject": self.claimed_subject,
            "faces": [f.to_dict() for f in self.faces],
            "timing_ms": {k: round(v, 1) for k, v in self.timing_ms.items()},
        }


class UnknownSubject(LookupError):
    pass


class IncompleteTemplate(LookupError):
    """The subject exists but has too few enrollment samples to be matched."""


class RecognitionPipeline:
    def __init__(
        self,
        detector: FaceDetector,
        embedder: FaceEmbedder,
        liveness: LivenessDetector | None,
        calibrator: ScoreCalibrator,
        vault: BiometricVault,
        settings: Settings,
        quality_config: QualityConfig | None = None,
        fusion_mode: str = "direct_sum",
    ) -> None:
        self.detector = detector
        self.embedder = embedder
        self.liveness = liveness if settings.liveness_enabled else None
        self.calibrator = calibrator
        self.vault = vault
        self.settings = settings
        self.quality_config = quality_config or QualityConfig()
        self.fusion_mode = fusion_mode
        self._stats_lock = threading.Lock()
        self._stats_version = -1
        self._template_stats: tuple[np.ndarray, np.ndarray] | None = None
        self._arc_templates: np.ndarray | None = None
        # Numbers for people who match nobody enrolled; memory only, never stored.
        self.visitors = VisitorRegistry()

    # -- observation ------------------------------------------------------- #

    def observe(self, image_bgr: np.ndarray, timing: dict[str, float] | None = None, max_faces: int | None = None) -> list[FaceObservation]:
        timing = timing if timing is not None else {}
        started = time.perf_counter()
        detections = self.detector.detect(
            image_bgr, max_faces=self.settings.max_faces if max_faces is None else max_faces
        )
        timing["detect"] = (time.perf_counter() - started) * 1000
        timing.setdefault("liveness", 0.0)
        timing.setdefault("embed", 0.0)
        observations = []
        for index, detection in enumerate(detections):
            observations.append(self._observe_face(image_bgr, detection, index == 0, timing))
        return observations

    def _observe_face(self, image_bgr, detection: Detection, primary: bool, timing: dict[str, float]) -> FaceObservation:
        pose = estimate_pose(detection.landmarks)
        liveness: float | None = None
        if self.liveness is not None:
            started = time.perf_counter()
            try:
                score = self.liveness.score(image_bgr, detection.bbox)
            except Exception:  # a failing liveness model must not pass as "live"
                score = None
            liveness = float("nan") if score is None else score
            timing["liveness"] += (time.perf_counter() - started) * 1000

        zero_quality = compute_quality(0.0, detection.score, pose.score, self.quality_config)
        aligned = align_face(image_bgr, detection.landmarks)
        if aligned is None:
            return FaceObservation(detection, primary, zero_quality, liveness, None, None, None,
                                   pose.yaw_offset, pose.pitch_offset, "DEGENERATE_LANDMARKS")
        started = time.perf_counter()
        try:
            embedding = self.embedder.embed(aligned)
        except ValueError:
            timing["embed"] += (time.perf_counter() - started) * 1000
            return FaceObservation(detection, primary, zero_quality, liveness, None, None, None,
                                   pose.yaw_offset, pose.pitch_offset, "EMBEDDING_FAILED")
        timing["embed"] += (time.perf_counter() - started) * 1000
        quality = compute_quality(embedding.ada_norm, detection.score, pose.score, self.quality_config)
        w_ada = fusion_weight(quality.score)
        fused = fuse_embeddings(embedding.ada, embedding.arc, w_ada, self.fusion_mode)
        return FaceObservation(detection, primary, quality, liveness, fused, embedding.arc, float(w_ada),
                               pose.yaw_offset, pose.pitch_offset)

    # -- gallery caches ---------------------------------------------------- #

    def _gallery_with_stats(self) -> tuple[Gallery, tuple[np.ndarray, np.ndarray] | None, np.ndarray | None]:
        """All enrolled templates with cached cohort statistics.

        Templates built from fewer than ``enroll_min_samples`` frames stay in
        the comparison on purpose: a half-enrolled lookalike must still be
        able to make the margin narrow. They are marked ineligible so the
        decision policy never asserts them as an identity.
        """
        gallery = self.vault.gallery()
        with self._stats_lock:
            if self._stats_version != gallery.version:
                if len(gallery):
                    self._template_stats = self.calibrator.cohort_stats(gallery.templates)
                    self._arc_templates = arcface_component(gallery.templates)
                else:
                    self._template_stats, self._arc_templates = None, None
                self._stats_version = gallery.version
            return gallery, self._template_stats, self._arc_templates

    # -- decisions --------------------------------------------------------- #

    def _decide(
        self,
        obs: FaceObservation,
        mode: Mode,
        claimed_index: int | None,
        thresholds: Thresholds,
        gallery: Gallery,
        stats,
        arc_templates,
        with_baseline: bool,
    ) -> FaceResult:
        if obs.fused is None:
            # Could not be embedded: abstain. quality is 0 so the policy lands
            # on LOW_IMAGE_QUALITY (or SPOOF if liveness failed first).
            decision = decide(mode, [], 0.0, obs.liveness, thresholds)
            if decision.zone is Zone.NO_MATCH:  # empty candidate list at q=0 cannot reach here, but be explicit
                decision = decide(mode, [], float("nan"), obs.liveness, thresholds)
            return FaceResult(obs, decision, [], None)

        candidates: list[Candidate] = []
        baseline_pairs: list[tuple[str, float]] = []
        if len(gallery):
            if mode is Mode.VERIFY:
                rows = [claimed_index] if claimed_index is not None else []
            else:
                rows = list(range(len(gallery)))
            if rows:
                templates = gallery.templates[rows]
                row_stats = (stats[0][rows], stats[1][rows]) if stats is not None else None
                cosine, scores = self.calibrator.score_matrix(obs.fused[None], templates, row_stats)
                minimum = self.settings.enroll_min_samples
                candidates = [
                    Candidate(gallery.names[row], float(scores[0, i]), float(cosine[0, i]),
                              eligible=gallery.sample_counts[row] >= minimum)
                    for i, row in enumerate(rows)
                ]
                candidates.sort(key=lambda c: c.score, reverse=True)
                if with_baseline and arc_templates is not None and obs.arc is not None:
                    arc_cosine = arc_templates[rows] @ obs.arc
                    baseline_pairs = [(gallery.names[row], float(arc_cosine[i])) for i, row in enumerate(rows)]

        decision = decide(mode, candidates, obs.quality.score, obs.liveness, thresholds)
        baseline = None
        if with_baseline:
            baseline = decide_baseline(mode, baseline_pairs, self.settings.baseline_cosine_threshold, thresholds)
        return FaceResult(obs, decision, candidates, baseline)

    def _apply_visitor_memory(self, face: FaceResult, gallery: Gallery, arc_templates) -> None:
        """Number un-enrolled faces, and keep a known visitor from being mistaken for a lookalike.

        A stranger who resembles an enrolled person scores "no match" on some
        frames and "might be X" on others. Once a clear frame has ruled them
        out, later borderline frames of the same face stay "no match": they
        match their own earlier frames far better than they match X.
        """
        obs, decision = face.observation, face.decision
        if obs.arc is None:
            return
        if decision.zone is Zone.MATCH_CONFIRMED:
            self.visitors.forget_like(obs.arc)  # confirmed as enrolled, so not a visitor
        elif decision.zone is Zone.NO_MATCH:
            # Only a clear, live face that matched nobody gets a number.
            face.visitor = self.visitors.assign(obs.arc)
        elif (decision.zone is Zone.UNCERTAIN_REVIEW and decision.reason is Reason.GRAY_ZONE_SIMILARITY
              and decision.top1 is not None and arc_templates is not None):
            known = self.visitors.match(obs.arc)
            row = gallery.index_of(decision.top1.subject)
            if known is None or row is None:
                return
            enrolled_cosine = float(arc_templates[row] @ obs.arc)
            if is_known_visitor(known[1], enrolled_cosine):
                face.decision = replace(decision, zone=Zone.NO_MATCH, reason=Reason.KNOWN_VISITOR,
                                        reasons=(Reason.KNOWN_VISITOR, *decision.reasons))
                face.visitor = self.visitors.assign(obs.arc)

    def process(
        self,
        image_bgr: np.ndarray,
        mode: Mode | str,
        thresholds: Thresholds,
        claimed_subject: str | None = None,
        with_baseline: bool = False,
    ) -> FrameResult:
        """Run the full pipeline on one frame."""
        mode = Mode(mode)
        started = time.perf_counter()
        gallery, stats, arc_templates = self._gallery_with_stats()
        claimed_index = None
        if mode is Mode.VERIFY:
            if not claimed_subject:
                raise ValueError("verify mode requires a claimed subject")
            claimed_index = gallery.index_of(claimed_subject)
            if claimed_index is None:
                raise UnknownSubject(claimed_subject)
            if gallery.sample_counts[claimed_index] < self.settings.enroll_min_samples:
                raise IncompleteTemplate(
                    f"'{claimed_subject}' has fewer than {self.settings.enroll_min_samples} "
                    "enrollment samples; capture more frames before verifying"
                )
        timing: dict[str, float] = {}
        observations = self.observe(image_bgr, timing)
        decide_started = time.perf_counter()
        faces = [
            self._decide(obs, mode, claimed_index, thresholds, gallery, stats, arc_templates, with_baseline)
            for obs in observations
        ]
        if mode is Mode.IDENTIFY:
            for face in faces:
                self._apply_visitor_memory(face, gallery, arc_templates)
        timing["decide"] = (time.perf_counter() - decide_started) * 1000
        timing["total"] = (time.perf_counter() - started) * 1000
        height, width = image_bgr.shape[:2]
        return FrameResult(width, height, mode, faces, timing, claimed_subject if mode is Mode.VERIFY else None)

    def compare(self, image_a: np.ndarray, image_b: np.ndarray, thresholds: Thresholds) -> dict:
        """1:1 comparison of the primary faces of two images (nothing is stored)."""
        obs_a = self.observe(image_a, max_faces=1)
        obs_b = self.observe(image_b, max_faces=1)
        if not obs_a or not obs_b:
            return {"faces_found": [len(obs_a), len(obs_b)], "decision": None}
        a, b = obs_a[0], obs_b[0]
        if a.fused is None or b.fused is None:
            return {"faces_found": [1, 1], "decision": None, "error": a.error or b.error}
        cosine, scores = self.calibrator.score_matrix(a.fused[None], b.fused[None])
        quality = min(a.quality.score, b.quality.score)
        liveness_values = [v for v in (a.liveness, b.liveness) if v is not None]
        liveness = min(liveness_values) if liveness_values and all(np.isfinite(liveness_values)) else (
            float("nan") if liveness_values else None
        )
        decision = decide(
            Mode.VERIFY, [Candidate("image_b", float(scores[0, 0]), float(cosine[0, 0]))],
            quality, liveness, thresholds,
        )
        return {
            "faces_found": [1, 1],
            "decision": decision.to_dict(),
            "quality": [a.quality.to_dict(), b.quality.to_dict()],
            "baseline_cosine": None if a.arc is None or b.arc is None else round(float(a.arc @ b.arc), 4),
        }

    # -- enrollment -------------------------------------------------------- #

    def enroll(self, subject: str, images: list[np.ndarray], thresholds: Thresholds, replace: bool = False) -> dict:
        """Enroll a subject from 1..N frames; stores only encrypted embeddings.

        A frame is rejected (with a reason) when it has no face, fails
        liveness, falls below the enrollment quality floor, or does not look
        like the same person as the other accepted frames.
        """
        reports: list[dict] = []
        accepted: list[tuple[int, FaceObservation]] = []
        for index, image in enumerate(images):
            observations = self.observe(image)
            if not observations:
                reports.append({"index": index, "accepted": False, "reason": "NO_FACE_DETECTED"})
                continue
            obs = observations[0]
            if len(observations) > 1:
                # Someone else of comparable size is in shot: refuse rather
                # than guess whose face is being enrolled.
                def area(o: FaceObservation) -> float:
                    return o.detection.width * o.detection.height

                # Faces are ranked by size *and* centrality, so a big face near
                # the edge can rank anywhere: check all of them.
                if max(area(other) for other in observations[1:]) >= 0.4 * area(obs):
                    reports.append({"index": index, "accepted": False, "reason": "MULTIPLE_FACES",
                                    "faces_in_frame": len(observations)})
                    continue
            report = {
                "index": index,
                "faces_in_frame": len(observations),
                "quality": obs.quality.to_dict(),
                "liveness": None if obs.liveness is None or not np.isfinite(obs.liveness) else round(obs.liveness, 4),
            }
            if obs.fused is None:
                report.update(accepted=False, reason=obs.error or "EMBEDDING_FAILED")
            elif obs.liveness is not None and not (np.isfinite(obs.liveness) and obs.liveness >= thresholds.l_min):
                report.update(accepted=False, reason="LIVENESS_BELOW_MINIMUM")
            elif obs.quality.score < max(self.settings.enroll_min_quality, thresholds.q_min):
                report.update(accepted=False, reason="LOW_IMAGE_QUALITY")
            else:
                report.update(accepted=True, reason=None)
                accepted.append((index, obs))
            reports.append(report)

        # Identity consistency: every accepted frame must match the template
        # formed by the remaining frames (and by the subject's existing
        # template when appending), otherwise two people are being mixed.
        gallery, _, _ = self._gallery_with_stats()
        existing_index = None if replace else gallery.index_of(subject)
        existing = gallery.templates[existing_index] if existing_index is not None else None
        changed = True
        while changed and accepted:
            changed = False
            for position, (index, obs) in enumerate(accepted):
                others = [o.fused for j, (_, o) in enumerate(accepted) if j != position]
                if existing is not None:
                    others.append(existing)
                reference = build_template(others) if others else None
                if reference is None:
                    continue
                _, score = self.calibrator.score_matrix(obs.fused[None], reference[None])
                if float(score[0, 0]) < thresholds.tau_low:
                    reports[index].update(accepted=False, reason="INCONSISTENT_IDENTITY")
                    accepted.pop(position)
                    changed = True
                    break
        # With exactly two mutually inconsistent frames the loop above drops
        # one arbitrarily; neither can be trusted, so drop the survivor too.
        if len(accepted) == 1 and existing is None:
            rejected_inconsistent = [r for r in reports if r.get("reason") == "INCONSISTENT_IDENTITY"]
            if rejected_inconsistent:
                reports[accepted[0][0]].update(accepted=False, reason="INCONSISTENT_IDENTITY")
                accepted = []

        if not accepted:
            return {"subject": subject, "enrolled": False, "accepted": 0, "frames": reports}

        embeddings = [obs.fused for _, obs in accepted]
        possible_duplicate = None
        if existing_index is None and len(gallery):
            candidate = build_template(embeddings)
            if candidate is not None:
                _, scores = self.calibrator.score_matrix(candidate[None], gallery.templates)
                best = int(np.argmax(scores[0]))
                if float(scores[0, best]) >= thresholds.tau_high and gallery.names[best] != subject:
                    possible_duplicate = gallery.names[best]

        info = self.vault.add_samples(
            subject,
            embeddings,
            [obs.quality.score for _, obs in accepted],
            replace=replace,
            max_samples=self.settings.enroll_retain_samples,
        )
        return {
            "subject": info.name,
            "enrolled": True,
            "accepted": len(accepted),
            "samples": info.samples,
            "template_complete": info.samples >= self.settings.enroll_min_samples,
            "active": info.samples >= self.settings.enroll_min_samples,
            "recommended_samples": [self.settings.enroll_min_samples, self.settings.enroll_max_samples],
            "possible_duplicate_of": possible_duplicate,
            "frames": reports,
        }
