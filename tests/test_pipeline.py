"""Alignment, fusion, three-zone policy, encryption, audit chain and API tests."""

from __future__ import annotations

import base64
import json
import math
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend import config
from backend.config import Thresholds, ThresholdStore
from backend.engine.decision import Candidate, Mode, Reason, Zone, decide, decide_baseline
from backend.engine.detector import (
    ARCFACE_REFERENCE,
    align_face,
    decode_scrfd_outputs,
    estimate_alignment,
    estimate_pose,
    nms,
    to_model_input,
    umeyama_similarity,
)
from backend.engine.embedder import (
    QualityConfig,
    arcface_component,
    compute_quality,
    fuse_embeddings,
    fusion_weight,
)
from backend.engine.liveness import crop_for_liveness, expanded_box
from backend.engine.pipeline import ImageError, decode_image
from backend.storage.audit import GENESIS_HASH, AuditLog, entry_hash
from backend.storage.vault import (
    BiometricVault,
    EmbeddingCipher,
    SubjectExists,
    SubjectNotFound,
    VaultError,
    VaultKeyError,
    build_template,
    generate_key,
    normalize_subject_name,
)
from tests.conftest import encode_png, frame_bytes, identity_vector, make_frame

T = Thresholds()  # tau_high 0.78, tau_low 0.48, margin 0.08, q_min 0.40, l_min 0.50


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


def cand(subject: str, score: float) -> Candidate:
    return Candidate(subject, score, score)


# --------------------------------------------------------------------------- #
# Stage 1: alignment and detection decoding
# --------------------------------------------------------------------------- #


class TestAlignment:
    def _transform(self, scale, degrees, tx, ty):
        theta = math.radians(degrees)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        return lambda pts: (scale * (rotation @ np.asarray(pts, dtype=np.float64).T)).T + [tx, ty]

    def test_recovers_known_similarity(self):
        forward = self._transform(2.7, 23.0, 140.0, -35.0)
        landmarks = forward(ARCFACE_REFERENCE)
        matrix = estimate_alignment(landmarks)
        mapped = (matrix[:, :2] @ landmarks.T).T + matrix[:, 2]
        assert np.allclose(mapped, ARCFACE_REFERENCE, atol=1e-4)
        # A similarity transform: equal singular values, positive determinant.
        s = np.linalg.svd(matrix[:, :2], compute_uv=False)
        assert s[0] == pytest.approx(s[1], rel=1e-9)
        assert s[0] == pytest.approx(1 / 2.7, rel=1e-6)
        assert np.linalg.det(matrix[:, :2]) > 0

    def test_never_returns_a_reflection(self):
        mirrored = ARCFACE_REFERENCE.copy()
        mirrored[:, 0] = 112 - mirrored[:, 0]  # left/right swapped geometry
        matrix = umeyama_similarity(mirrored, ARCFACE_REFERENCE)
        assert np.linalg.det(matrix[:, :2]) > 0

    def test_least_squares_with_noise(self):
        rng = np.random.default_rng(0)
        landmarks = self._transform(1.8, -12.0, 60, 40)(ARCFACE_REFERENCE) + rng.normal(0, 0.8, (5, 2))
        matrix = estimate_alignment(landmarks)
        mapped = (matrix[:, :2] @ landmarks.T).T + matrix[:, 2]
        assert np.abs(mapped - ARCFACE_REFERENCE).max() < 2.0

    def test_degenerate_landmarks_return_none(self):
        assert estimate_alignment(np.full((5, 2), 40.0)) is None
        assert estimate_alignment(np.full((5, 2), np.nan)) is None
        assert align_face(np.zeros((100, 100, 3), np.uint8), np.full((5, 2), 40.0)) is None

    def test_shape_validation(self):
        with pytest.raises(ValueError):
            estimate_alignment(np.zeros((4, 2)))

    def test_aligned_crop_lands_landmarks_on_template(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = self._transform(2.2, 15.0, 200, 90)(ARCFACE_REFERENCE)
        colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        for (x, y), colour in zip(landmarks, colours):
            cv2.circle(image, (int(round(x)), int(round(y))), 5, colour, -1)
        crop = align_face(image, landmarks)
        assert crop.shape == (112, 112, 3) and crop.dtype == np.uint8
        for (x, y), colour in zip(ARCFACE_REFERENCE, colours):
            assert tuple(int(v) for v in crop[int(round(y)), int(round(x))]) == colour

    def test_model_input_range_and_channel_order(self):
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        crop[..., 0] = 255  # pure blue in BGR
        rgb = to_model_input(crop, "RGB")
        bgr = to_model_input(crop, "BGR")
        assert rgb.shape == (1, 3, 112, 112) and rgb.dtype == np.float32
        assert rgb.min() == -1.0 and rgb.max() == 1.0
        assert rgb[0, 2].min() == 1.0 and rgb[0, 0].max() == -1.0  # blue is the last RGB plane
        assert bgr[0, 0].min() == 1.0
        with pytest.raises(ValueError):
            to_model_input(crop, "GRB")


class TestPose:
    def test_frontal_scores_high(self):
        pose = estimate_pose(ARCFACE_REFERENCE)
        assert pose.score > 0.95 and pose.yaw_offset < 0.05 and pose.pitch_offset < 0.05

    def test_yaw_lowers_score(self):
        turned = ARCFACE_REFERENCE.copy()
        turned[2, 0] = turned[1, 0] - 2  # nose almost under the right eye
        assert estimate_pose(turned).score < 0.15
        slight = ARCFACE_REFERENCE.copy()
        slight[2, 0] += 3  # a few degrees off frontal costs very little
        assert estimate_pose(slight).score > 0.9

    def test_pitch_lowers_score(self):
        nodding = ARCFACE_REFERENCE.copy()
        nodding[2, 1] = nodding[3, 1] - 2  # nose down at mouth level
        assert estimate_pose(nodding).score < 0.3

    def test_roll_is_not_penalised(self):
        theta = math.radians(30)
        rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        rolled = (rotation @ ARCFACE_REFERENCE.T).T
        pose = estimate_pose(rolled)
        assert pose.score > 0.95 and pose.roll_degrees == pytest.approx(30, abs=1)

    def test_garbage_scores_zero(self):
        assert estimate_pose(np.zeros((5, 2))).score == 0.0
        assert estimate_pose(np.full((5, 2), np.nan)).score == 0.0


class TestScrfdDecoding:
    def _outputs(self, size=64):
        outputs = []
        for kind_width in (1, 4, 10):
            for stride in (8, 16, 32):
                outputs.append(np.zeros(((size // stride) ** 2 * 2, kind_width), dtype=np.float32))
        return outputs

    def test_decodes_box_and_keypoints(self):
        outputs = self._outputs()
        # stride 16, cell (row 1, col 2), second anchor -> centre (32, 16)
        index = (1 * 4 + 2) * 2 + 1
        outputs[1][index, 0] = 0.9
        outputs[4][index] = [1.0, 0.5, 1.0, 1.5]  # left, top, right, bottom in strides
        outputs[7][index] = [0.25, -0.25] * 5
        boxes, scores, kps = decode_scrfd_outputs(outputs, 64, 64, 0.5)
        assert scores.tolist() == pytest.approx([0.9])
        assert boxes[0].tolist() == pytest.approx([32 - 16, 16 - 8, 32 + 16, 16 + 24])
        assert kps.shape == (1, 5, 2)
        assert kps[0, 0].tolist() == pytest.approx([32 + 4, 16 - 4])

    def test_threshold_and_empty(self):
        boxes, scores, kps = decode_scrfd_outputs(self._outputs(), 64, 64, 0.5)
        assert boxes.shape == (0, 4) and scores.shape == (0,) and kps.shape == (0, 5, 2)

    def test_rejects_models_without_keypoints(self):
        with pytest.raises(ValueError, match="key points"):
            decode_scrfd_outputs(self._outputs()[:6], 64, 64, 0.5)

    def test_nms_keeps_best_of_overlapping(self):
        boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105], [300, 300, 400, 400]], dtype=np.float32)
        scores = np.array([0.8, 0.9, 0.7], dtype=np.float32)
        assert nms(boxes, scores, 0.4) == [1, 2]
        assert nms(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32), 0.4) == []


class TestLivenessCrop:
    def test_box_expands_around_centre(self):
        assert expanded_box(1000, 1000, (400, 400, 100, 100), 2.7) == (315, 315, 585, 585)

    def test_box_is_shifted_back_into_frame(self):
        left, top, right, bottom = expanded_box(640, 480, (0, 0, 100, 100), 2.7)
        assert (left, top) == (0, 0) and right == 270 and bottom == 270

    def test_scale_is_capped_by_image_size(self):
        left, top, right, bottom = expanded_box(200, 200, (50, 50, 100, 100), 4.0)
        assert left >= 0 and top >= 0 and right <= 199 and bottom <= 199

    def test_crop_output_and_degenerate_input(self):
        image = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
        crop = crop_for_liveness(image, (200, 100, 300, 220), 2.7)
        assert crop.shape == (80, 80, 3) and crop.dtype == np.uint8
        assert crop_for_liveness(image, (200, 100, 200, 100), 2.7) is None


# --------------------------------------------------------------------------- #
# Stage 3: quality and fusion
# --------------------------------------------------------------------------- #


class TestQuality:
    cfg = QualityConfig(norm_min=5, norm_max=25)

    def test_bounds_and_monotonicity(self):
        assert compute_quality(25, 0.95, 1.0, self.cfg).score == pytest.approx(1.0)
        assert compute_quality(5, 0.95, 1.0, self.cfg).score == 0.0
        scores = [compute_quality(n, 0.9, 0.9, self.cfg).score for n in (6, 10, 15, 20, 30)]
        assert scores == sorted(scores) and all(0 <= s <= 1 for s in scores)

    def test_any_collapsed_component_vetoes(self):
        assert compute_quality(25, 0.95, 0.0, self.cfg).score == 0.0  # profile view
        assert compute_quality(25, 0.35, 1.0, self.cfg).score == 0.0  # detector has no confidence
        assert 0.8 < compute_quality(25, 0.5, 1.0, self.cfg).score < 0.9  # barely detected: mild penalty

    def test_non_finite_inputs_fail_low(self):
        assert compute_quality(float("nan"), 0.9, 1.0, self.cfg).score == 0.0
        assert compute_quality(20, 0.9, float("nan"), self.cfg).score == 0.0

    def test_config_validation(self):
        with pytest.raises(ValueError):
            QualityConfig(norm_min=10, norm_max=10)
        with pytest.raises(ValueError):
            QualityConfig(weight_norm=0.9, weight_det=0.9, weight_pose=0.1)


class TestFusion:
    def test_weight_rises_as_quality_falls(self):
        assert fusion_weight(1.0) == pytest.approx(0.40)
        assert fusion_weight(0.0) == pytest.approx(0.70)
        weights = fusion_weight(np.linspace(0, 1, 11))
        assert np.all(np.diff(weights) < 0)
        assert fusion_weight(float("nan")) == pytest.approx(0.70)  # unknown quality -> lean on AdaFace
        assert fusion_weight(7.0) == pytest.approx(0.40)

    def test_output_is_unit_length(self):
        for mode, width in (("direct_sum", 1024), ("sum", 512)):
            fused = fuse_embeddings(unit(1) * 9.0, unit(2) * 0.3, 0.55, mode)
            assert fused.shape == (width,) and fused.dtype == np.float32
            assert np.linalg.norm(fused) == pytest.approx(1.0, abs=1e-6)

    def test_direct_sum_matches_formula(self):
        a, r, w = unit(1), unit(2), 0.6
        fused = fuse_embeddings(a, r, w)
        norm = math.sqrt(w**2 + (1 - w) ** 2)
        assert np.allclose(fused[:512], w * a / norm, atol=1e-6)
        assert np.allclose(fused[512:], (1 - w) * r / norm, atol=1e-6)

    def test_sum_mode_matches_literal_formula(self):
        a, r, w = unit(1), unit(2), 0.6
        expected = w * a + (1 - w) * r
        assert np.allclose(fuse_embeddings(a, r, w, "sum"), expected / np.linalg.norm(expected), atol=1e-6)

    def test_direct_sum_cosine_is_weighted_mix_of_backbone_cosines(self):
        a1, r1, a2, r2 = unit(1), unit(2), unit(3), unit(4)
        w1, w2 = 0.45, 0.65
        cosine = float(fuse_embeddings(a1, r1, w1) @ fuse_embeddings(a2, r2, w2))
        n1, n2 = math.hypot(w1, 1 - w1), math.hypot(w2, 1 - w2)
        expected = (w1 * w2 * (a1 @ a2) + (1 - w1) * (1 - w2) * (r1 @ r2)) / (n1 * n2)
        assert cosine == pytest.approx(expected, abs=1e-6)

    def test_sum_mode_adds_cross_space_noise(self):
        """Why direct_sum is the default: same identity, independent spaces."""
        rng = np.random.default_rng(5)
        direct, summed = [], []
        for _ in range(300):
            a1, r1, a2, r2 = (v / np.linalg.norm(v) for v in rng.standard_normal((4, 512)))
            direct.append(float(fuse_embeddings(a1, r1, 0.5) @ fuse_embeddings(a2, r2, 0.5)))
            summed.append(float(fuse_embeddings(a1, r1, 0.5, "sum") @ fuse_embeddings(a2, r2, 0.5, "sum")))
        assert np.std(summed) > 1.25 * np.std(direct)

    def test_batch_matches_single(self):
        a = np.stack([unit(i) for i in range(4)])
        r = np.stack([unit(10 + i) for i in range(4)])
        w = np.array([0.4, 0.5, 0.6, 0.7])
        batch = fuse_embeddings(a, r, w)
        for i in range(4):
            assert np.allclose(batch[i], fuse_embeddings(a[i], r[i], float(w[i])), atol=1e-6)

    def test_arcface_component_roundtrip(self):
        a, r = unit(1), unit(2)
        assert np.allclose(arcface_component(fuse_embeddings(a, r, 0.63)), r, atol=1e-6)
        assert arcface_component(fuse_embeddings(a, r, 0.63, "sum")) is None

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            fuse_embeddings(unit(1), unit(2), 1.5)
        with pytest.raises(ValueError):
            fuse_embeddings(unit(1), unit(2, 256), 0.5)
        with pytest.raises(ValueError):
            fuse_embeddings(unit(1), unit(2), 0.5, "concat")

    def test_multi_sample_template(self):
        base = unit(1)
        rng = np.random.default_rng(0)
        samples = [base + 0.6 * rng.standard_normal(512) / math.sqrt(512) for _ in range(5)]
        template = build_template(samples)
        assert np.linalg.norm(template) == pytest.approx(1.0, abs=1e-6)
        single = samples[0] / np.linalg.norm(samples[0])
        assert float(template @ base) > float(single @ base)  # averaging removes pose noise
        assert build_template([base, -base]) is None
        assert build_template([]) is None


# --------------------------------------------------------------------------- #
# Stage 5: three-zone decision policy
# --------------------------------------------------------------------------- #


class TestThreeZonePolicy:
    def test_high_similarity_wide_margin_is_match(self):
        d = decide(Mode.IDENTIFY, [cand("alice", 0.93), cand("bob", 0.21)], 0.8, 0.95, T)
        assert d.zone is Zone.MATCH_CONFIRMED and d.subject == "alice"
        assert d.margin == pytest.approx(0.72)

    def test_twins_narrow_margin_is_uncertain(self):
        # The specification's example: s1 = 0.82, s2 = 0.79, margin 0.03 < 0.08.
        d = decide(Mode.IDENTIFY, [cand("twin_a", 0.82), cand("twin_b", 0.79)], 0.9, 0.99, T)
        assert d.zone is Zone.UNCERTAIN_REVIEW
        assert d.reason is Reason.NARROW_CANDIDATE_MARGIN
        assert d.subject is None, "no identity may be asserted in the uncertain zone"
        assert d.margin == pytest.approx(0.03)

    def test_low_quality_is_uncertain(self):
        d = decide(Mode.IDENTIFY, [cand("alice", 0.97), cand("bob", 0.1)], 0.39, 0.99, T)
        assert d.zone is Zone.UNCERTAIN_REVIEW and d.reason is Reason.LOW_IMAGE_QUALITY
        assert d.subject is None

    def test_low_quality_blocks_non_match_claim_too(self):
        d = decide(Mode.IDENTIFY, [cand("alice", 0.05)], 0.2, 0.99, T)
        assert d.zone is Zone.UNCERTAIN_REVIEW and d.reason is Reason.LOW_IMAGE_QUALITY

    def test_dissimilar_face_is_no_match(self):
        d = decide(Mode.IDENTIFY, [cand("alice", 0.12), cand("bob", 0.10)], 0.8, 0.9, T)
        assert d.zone is Zone.NO_MATCH and d.reason is Reason.SCORE_BELOW_LOW_THRESHOLD
        assert d.subject is None

    def test_two_weak_candidates_are_not_an_ambiguity(self):
        d = decide(Mode.IDENTIFY, [cand("a", 0.30), cand("b", 0.29)], 0.8, 0.9, T)
        assert d.zone is Zone.NO_MATCH

    def test_gray_zone(self):
        d = decide(Mode.IDENTIFY, [cand("alice", 0.60), cand("bob", 0.10)], 0.8, 0.9, T)
        assert d.zone is Zone.UNCERTAIN_REVIEW and d.reason is Reason.GRAY_ZONE_SIMILARITY
        both = decide(Mode.IDENTIFY, [cand("alice", 0.60), cand("bob", 0.58)], 0.8, 0.9, T)
        assert both.reasons == (Reason.GRAY_ZONE_SIMILARITY, Reason.NARROW_CANDIDATE_MARGIN)

    def test_spoof_takes_precedence(self):
        d = decide(Mode.IDENTIFY, [cand("alice", 0.99)], 0.9, 0.49, T)
        assert d.zone is Zone.SPOOF_DETECTED and d.reason is Reason.LIVENESS_BELOW_MINIMUM
        assert d.subject is None
        low_everything = decide(Mode.IDENTIFY, [cand("alice", 0.99)], 0.1, 0.1, T)
        assert low_everything.zone is Zone.SPOOF_DETECTED

    @pytest.mark.parametrize(
        "score,quality,liveness,zone",
        [
            (0.78, 0.40, 0.50, Zone.MATCH_CONFIRMED),  # every bound is inclusive on the pass side
            (0.7799, 0.40, 0.50, Zone.UNCERTAIN_REVIEW),
            (0.48, 0.40, 0.50, Zone.UNCERTAIN_REVIEW),  # tau_low itself is gray
            (0.4799, 0.40, 0.50, Zone.NO_MATCH),
            (0.90, 0.3999, 0.50, Zone.UNCERTAIN_REVIEW),
            (0.90, 0.40, 0.4999, Zone.SPOOF_DETECTED),
        ],
    )
    def test_boundaries(self, score, quality, liveness, zone):
        assert decide(Mode.VERIFY, [cand("alice", score)], quality, liveness, T).zone is zone

    def test_margin_boundary_is_inclusive(self):
        # Exactly representable values so the comparison is not at the mercy
        # of float rounding: margin 0.125 against delta 0.125.
        t = Thresholds(tau_high=0.75, tau_low=0.5, margin=0.125)
        exact = decide(Mode.IDENTIFY, [cand("a", 0.875), cand("b", 0.75)], 0.9, None, t)
        assert exact.zone is Zone.MATCH_CONFIRMED
        below = decide(Mode.IDENTIFY, [cand("a", 0.875), cand("b", 0.7501)], 0.9, None, t)
        assert below.zone is Zone.UNCERTAIN_REVIEW

    def test_verify_mode_ignores_margin(self):
        d = decide(Mode.VERIFY, [cand("alice", 0.82)], 0.9, 0.9, T)
        assert d.zone is Zone.MATCH_CONFIRMED and d.margin is None
        with pytest.raises(ValueError):
            decide(Mode.VERIFY, [cand("a", 0.9), cand("b", 0.8)], 0.9, 0.9, T)

    def test_single_enrolled_subject_and_empty_gallery(self):
        assert decide(Mode.IDENTIFY, [cand("alice", 0.9)], 0.9, 0.9, T).zone is Zone.MATCH_CONFIRMED
        empty = decide(Mode.IDENTIFY, [], 0.9, 0.9, T)
        assert empty.zone is Zone.NO_MATCH and empty.reason is Reason.EMPTY_GALLERY and empty.top1 is None

    def test_liveness_disabled_skips_spoof_check(self):
        assert decide(Mode.VERIFY, [cand("alice", 0.9)], 0.9, None, T).zone is Zone.MATCH_CONFIRMED

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_inputs_fail_safe(self, bad):
        for kwargs in (
            dict(candidates=[cand("a", bad)], quality=0.9, liveness=0.9),
            dict(candidates=[cand("a", 0.9)], quality=float("nan"), liveness=0.9),
            dict(candidates=[cand("a", 0.9)], quality=0.9, liveness=float("nan")),
        ):
            d = decide(Mode.IDENTIFY, thresholds=T, **kwargs)
            assert d.zone is Zone.UNCERTAIN_REVIEW and d.reason is Reason.INVALID_SCORE and d.subject is None

    def test_candidates_are_ranked_regardless_of_input_order(self):
        d = decide(Mode.IDENTIFY, [cand("bob", 0.2), cand("alice", 0.95), cand("eve", 0.5)], 0.9, 0.9, T)
        assert d.top1.subject == "alice" and d.top2.subject == "eve"

    def test_subject_only_set_on_match_exhaustive(self):
        rng = np.random.default_rng(0)
        for _ in range(2000):
            s1, s2, q, l = rng.uniform(0, 1, 4)
            d = decide(Mode.IDENTIFY, [cand("a", s1), cand("b", s2)], q, l, T)
            top, runner = max(s1, s2), min(s1, s2)
            if d.zone is Zone.MATCH_CONFIRMED:
                assert d.subject is not None
                assert top >= T.tau_high and top - runner >= T.margin and q >= T.q_min and l >= T.l_min
            else:
                assert d.subject is None
            if d.zone is Zone.NO_MATCH:
                assert top < T.tau_low and q >= T.q_min and l >= T.l_min

    def test_decision_serialises_to_strict_json(self):
        d = decide(Mode.IDENTIFY, [cand("a", float("nan"))], 0.9, 0.9, T)
        json.dumps(d.to_dict(), allow_nan=False)

    def test_baseline_forces_a_binary_answer(self):
        # The same twins the fair engine refuses to separate.
        d = decide_baseline(Mode.IDENTIFY, [("twin_a", 0.61), ("twin_b", 0.60)], 0.32, T)
        assert d.zone is Zone.MATCH_CONFIRMED and d.subject == "twin_a"
        assert decide_baseline(Mode.IDENTIFY, [("x", 0.1)], 0.32, T).zone is Zone.NO_MATCH
        assert decide_baseline(Mode.IDENTIFY, [], 0.32, T).reason is Reason.EMPTY_GALLERY


class TestThresholdConfig:
    def test_defaults_match_specification(self):
        assert T.to_dict() == {"tau_high": 0.78, "tau_low": 0.48, "margin": 0.08, "q_min": 0.40, "l_min": 0.50}

    @pytest.mark.parametrize(
        "kwargs", [dict(tau_low=0.8), dict(tau_high=1.2), dict(margin=-0.1), dict(q_min=float("nan"))]
    )
    def test_invalid_rejected(self, kwargs):
        with pytest.raises(ValueError):
            Thresholds(**kwargs).validate()

    def test_store_rejects_inconsistent_update_atomically(self):
        store = ThresholdStore(Thresholds())
        with pytest.raises(ValueError):
            store.update(tau_low=0.9)
        with pytest.raises(ValueError):
            store.update(bogus=0.1)
        assert store.get() == Thresholds()
        assert store.update(tau_high=0.9, margin=0.1).tau_high == 0.9

    def test_provider_selection_always_ends_on_cpu(self, monkeypatch):
        monkeypatch.delenv("FAIRFACE_ONNX_PROVIDERS", raising=False)
        assert config.select_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]
        assert config.select_providers(
            ["CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"]
        ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert config.select_providers(["CoreMLExecutionProvider", "CPUExecutionProvider"]) == [
            "CoreMLExecutionProvider", "CPUExecutionProvider"]
        monkeypatch.setenv("FAIRFACE_ONNX_PROVIDERS", "CUDAExecutionProvider")
        assert config.select_providers(["CPUExecutionProvider"]) == ["CPUExecutionProvider"]


# --------------------------------------------------------------------------- #
# Storage: encryption at rest and the audit chain
# --------------------------------------------------------------------------- #


class TestVaultEncryption:
    def test_roundtrip_and_template(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        samples = [unit(1) + 0.05 * unit(i) for i in (2, 3, 4)]
        info = vault.add_samples("Alice", samples, [0.9, 0.8, 0.7])
        assert info.samples == 3 and info.mean_quality == pytest.approx(0.8)
        gallery = vault.gallery()
        assert gallery.names == ("Alice",) and gallery.templates.shape == (1, 512)
        assert np.allclose(gallery.templates[0], build_template(samples), atol=1e-6)

    def test_database_holds_only_ciphertext(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        vector = unit(7).astype(np.float32)
        vault.add_samples("Alice", [vector], [0.9])
        vault.close()
        raw = b"".join(p.read_bytes() for p in tmp_path.glob("b.db*"))
        assert vector.tobytes() not in raw
        assert vector.tobytes()[:64] not in raw
        blob = sqlite3.connect(tmp_path / "b.db").execute("SELECT embedding FROM samples").fetchone()[0]
        assert isinstance(blob, bytes) and blob[:1] == b"\x01"  # AES-256-GCM envelope
        assert len(blob) == 1 + 12 + (8 + 512 * 4) + 16
        # Two encryptions of the same vector must differ (random nonce).
        vault2 = BiometricVault(tmp_path / "b.db")
        vault2.add_samples("Alice", [vector], [0.9])
        blobs = [r[0] for r in sqlite3.connect(tmp_path / "b.db").execute("SELECT embedding FROM samples")]
        assert blobs[0] != blobs[1]

    def test_default_cipher_uses_a_256_bit_key(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        assert vault.cipher_name == "aes256gcm"
        key = (tmp_path / "biometric.key").read_bytes().strip()
        assert len(base64.urlsafe_b64decode(key)) == 32
        assert (tmp_path / "biometric.key").stat().st_mode & 0o077 == 0

    def test_fernet_mode(self, tmp_path):
        key = generate_key()
        vault = BiometricVault(tmp_path / "b.db", key, cipher="fernet")
        vault.add_samples("Alice", [unit(1)])
        blob = sqlite3.connect(tmp_path / "b.db").execute("SELECT embedding FROM samples").fetchone()[0]
        assert blob[:1] == b"\x02"
        from cryptography.fernet import Fernet

        Fernet(key).decrypt(blob[1:])  # a standard Fernet token under the same key
        # Either cipher setting can read records written by the other.
        assert BiometricVault(tmp_path / "b.db", key, cipher="aes256gcm").gallery().names == ("Alice",)

    def test_wrong_key_is_rejected_at_open(self, tmp_path):
        BiometricVault(tmp_path / "b.db", generate_key()).add_samples("Alice", [unit(1)])
        with pytest.raises(VaultKeyError):
            BiometricVault(tmp_path / "b.db", generate_key())

    def test_key_from_environment(self, tmp_path, monkeypatch):
        key = generate_key().decode()
        monkeypatch.setenv("BIOMETRIC_ENCRYPTION_KEY", key)
        vault = BiometricVault(tmp_path / "b.db")
        assert vault.key_source == "environment" and not (tmp_path / "biometric.key").exists()
        monkeypatch.setenv("BIOMETRIC_ENCRYPTION_KEY", "not-a-key")
        with pytest.raises(VaultKeyError):
            BiometricVault(tmp_path / "c.db")

    def test_ciphertext_cannot_be_moved_between_subjects(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        vault.add_samples("Alice", [unit(1)])
        vault.add_samples("Mallory", [unit(2)])
        conn = sqlite3.connect(tmp_path / "b.db")
        mallory_blob = conn.execute(
            "SELECT embedding FROM samples WHERE subject_id=(SELECT id FROM subjects WHERE name='Mallory')"
        ).fetchone()[0]
        conn.execute(
            "UPDATE samples SET embedding=? WHERE subject_id=(SELECT id FROM subjects WHERE name='Alice')",
            (mallory_blob,),
        )
        conn.commit()
        conn.close()
        with pytest.raises(VaultError, match="authentication"):
            BiometricVault(tmp_path / "b.db").gallery()

    def test_relabelled_subject_rows_are_detected(self, tmp_path):
        """Swapping two names in SQL must not hand one person the other's template."""
        vault = BiometricVault(tmp_path / "b.db")
        vault.add_samples("Alice", [unit(1)])
        vault.add_samples("Mallory", [unit(2)])
        conn = sqlite3.connect(tmp_path / "b.db")
        conn.execute("UPDATE subjects SET name='tmp' WHERE name='Alice'")
        conn.execute("UPDATE subjects SET name='Alice' WHERE name='Mallory'")
        conn.execute("UPDATE subjects SET name='Mallory' WHERE name='tmp'")
        conn.commit()
        conn.close()
        with pytest.raises(VaultError, match="authentication"):
            BiometricVault(tmp_path / "b.db").gallery()

    def test_missing_key_check_with_templates_is_refused(self, tmp_path):
        key = generate_key()
        BiometricVault(tmp_path / "b.db", key).add_samples("Alice", [unit(1)])
        conn = sqlite3.connect(tmp_path / "b.db")
        conn.execute("DELETE FROM meta WHERE key='key_check'")
        conn.commit()
        conn.close()
        with pytest.raises(VaultKeyError, match="key check"):
            BiometricVault(tmp_path / "b.db", generate_key())

    def test_deleted_template_leaves_no_ciphertext_behind(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        vault.add_samples("Alice", [unit(1)])
        blob = sqlite3.connect(tmp_path / "b.db").execute("SELECT embedding FROM samples").fetchone()[0]
        vault.delete_subject("Alice")
        remaining = b"".join(p.read_bytes() for p in tmp_path.glob("b.db*"))
        assert blob[13:77] not in remaining

    def test_tampered_ciphertext_is_detected(self):
        cipher = EmbeddingCipher(generate_key())
        blob = bytearray(cipher.encrypt(b"payload", b"ctx"))
        blob[-1] ^= 1
        with pytest.raises(VaultError):
            cipher.decrypt(bytes(blob), b"ctx")
        with pytest.raises(VaultError):
            cipher.decrypt(b"\x09junk", b"ctx")

    def test_subject_lifecycle(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        vault.add_samples("Alice", [unit(1)])
        vault.add_samples("Bob", [unit(2)])
        assert [s.name for s in vault.list_subjects()] == ["Alice", "Bob"]
        vault.rename_subject("Bob", "Robert")
        assert vault.gallery().names == ("Alice", "Robert")  # AAD is the row id, so rename keeps blobs readable
        with pytest.raises(SubjectExists):
            vault.rename_subject("Robert", "Alice")
        with pytest.raises(SubjectNotFound):
            vault.delete_subject("Nobody")
        assert vault.delete_subject("Alice") == 1
        assert vault.gallery().names == ("Robert",)
        assert sqlite3.connect(tmp_path / "b.db").execute("SELECT COUNT(*) FROM samples").fetchone()[0] == 1

    def test_append_replace_and_cap(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        vault.add_samples("Alice", [unit(1), unit(2)])
        assert vault.add_samples("Alice", [unit(3)]).samples == 3
        assert vault.add_samples("Alice", [unit(4)], replace=True).samples == 1
        assert vault.add_samples("Alice", [unit(i) for i in range(5, 11)], max_samples=5).samples == 5

    def test_rejects_bad_embeddings_and_names(self, tmp_path):
        vault = BiometricVault(tmp_path / "b.db")
        with pytest.raises(VaultError):
            vault.add_samples("Alice", [np.full(512, np.nan)])
        vault.add_samples("Alice", [unit(1)])
        with pytest.raises(VaultError, match="dimension"):
            vault.add_samples("Bob", [unit(1, 1024)])
        for bad in ("", "   ", "x" * 200, "a\x00b"):
            with pytest.raises(ValueError):
                normalize_subject_name(bad)
        assert normalize_subject_name("  Ada   Lovelace ") == "Ada Lovelace"


class TestAuditChain:
    def _log(self, tmp_path) -> AuditLog:
        return AuditLog(tmp_path / "b.db", tmp_path / "audit.jsonl")

    def _record(self, log: AuditLog, **overrides):
        fields = dict(
            mode="identify", decision_zone="MATCH_CONFIRMED", abstention_reason="SCORE_ABOVE_HIGH_THRESHOLD",
            quality_score=0.81234567, liveness_score=0.97, top1_subject="alice", top1_score=0.93,
            top2_score=0.2, margin=0.73, thresholds=T.to_dict(),
        )
        fields.update(overrides)
        return log.record(**fields)

    def test_chain_links_and_required_fields(self, tmp_path):
        log = self._log(tmp_path)
        first, second = self._record(log), self._record(log, top1_subject="bob")
        assert first["prev_hash"] == GENESIS_HASH and second["prev_hash"] == first["hash"]
        assert (first["seq"], second["seq"]) == (1, 2)
        for name in ("timestamp", "mode", "quality_score", "liveness_score", "top1_subject", "top1_score",
                     "top2_score", "margin", "thresholds", "decision_zone", "abstention_reason"):
            assert name in first
        assert first["hash"] == entry_hash({k: v for k, v in first.items() if k != "hash"})
        report = log.verify()
        assert report["ok"] and report["entries"] == 2 and report["head_hash"] == second["hash"]

    def test_jsonl_mirrors_database(self, tmp_path):
        log = self._log(tmp_path)
        entries = [self._record(log) for _ in range(3)]
        lines = [json.loads(line) for line in (tmp_path / "audit.jsonl").read_text().splitlines()]
        assert lines == entries

    def test_non_finite_values_are_stored_as_null(self, tmp_path):
        log = self._log(tmp_path)
        entry = self._record(log, quality_score=float("nan"), liveness_score=None, margin=float("inf"))
        assert entry["quality_score"] is None and entry["liveness_score"] is None and entry["margin"] is None
        assert log.verify()["ok"]

    def test_edited_jsonl_entry_is_detected(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        path = tmp_path / "audit.jsonl"
        lines = path.read_text().splitlines()
        forged = json.loads(lines[1])
        forged["decision_zone"] = "NO_MATCH"
        lines[1] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n")
        report = log.verify()
        assert not report["ok"] and not report["jsonl_ok"] and report["sqlite_ok"]
        assert any(e["seq"] == 2 for e in report["errors"])

    def test_recomputed_hash_still_breaks_the_chain(self, tmp_path):
        """An attacker who re-hashes the edited entry is caught by the next link."""
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        path = tmp_path / "audit.jsonl"
        lines = path.read_text().splitlines()
        forged = json.loads(lines[0])
        forged["top1_subject"] = "mallory"
        forged["hash"] = entry_hash({k: v for k, v in forged.items() if k != "hash"})
        lines[0] = json.dumps(forged, sort_keys=True, separators=(",", ":"))
        path.write_text("\n".join(lines) + "\n")
        report = log.verify()
        assert not report["ok"]
        assert any("prev_hash" in e["error"] for e in report["errors"])
        assert not report["stores_consistent"]

    def test_edited_database_row_is_detected(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        conn = sqlite3.connect(tmp_path / "b.db")
        payload = json.loads(conn.execute("SELECT payload FROM audit_log WHERE seq=2").fetchone()[0])
        payload["top1_score"] = 0.01
        conn.execute("UPDATE audit_log SET payload=? WHERE seq=2", (json.dumps(payload, sort_keys=True, separators=(",", ":")),))
        conn.commit()
        conn.close()
        report = log.verify()
        assert not report["ok"] and not report["sqlite_ok"]

    def test_edited_index_column_is_detected(self, tmp_path):
        log = self._log(tmp_path)
        self._record(log)
        conn = sqlite3.connect(tmp_path / "b.db")
        conn.execute("UPDATE audit_log SET decision_zone='NO_MATCH' WHERE seq=1")
        conn.commit()
        conn.close()
        assert not log.verify()["ok"]

    def test_deleted_and_reordered_entries_are_detected(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(4):
            self._record(log)
        path = tmp_path / "audit.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0], lines[2], lines[3]]) + "\n")
        assert not log.verify()["ok"]
        path.write_text("\n".join([lines[1], lines[0], lines[2], lines[3]]) + "\n")
        assert not log.verify()["ok"]

    def test_truncated_file_is_detected_while_database_survives(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        path = tmp_path / "audit.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")
        report = log.verify()
        assert not report["ok"] and not report["stores_consistent"]

    def test_crash_between_stores_is_repaired_on_restart(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        log.close()
        path = tmp_path / "audit.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")  # last line never hit the disk
        reopened = self._log(tmp_path)
        assert reopened.recovered_entries == 1
        report = reopened.verify()
        assert report["ok"] and report["entries"] == 4  # 3 decisions + the repair, which is on the record
        assert reopened.entries()[-1]["decision_zone"] == "AUDIT_MIRROR_REPAIRED"
        self._record(reopened)
        assert reopened.verify()["entries"] == 5

    def test_torn_last_line_is_repaired(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        log.close()
        path = tmp_path / "audit.jsonl"
        path.write_bytes(path.read_bytes()[:-40])  # power cut in the middle of the last line
        reopened = self._log(tmp_path)
        assert reopened.recovered_entries == 1 and reopened.verify()["ok"]
        self._record(reopened)
        assert reopened.verify()["ok"]

    def test_missing_final_newline_does_not_corrupt_the_file(self, tmp_path):
        """A stripped trailing newline must be restored, not glued onto or NUL-padded."""
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        log.close()
        path = tmp_path / "audit.jsonl"
        path.write_bytes(path.read_bytes().rstrip(b"\n"))
        reopened = self._log(tmp_path)
        self._record(reopened)
        assert reopened.verify()["ok"] and b"\x00" not in path.read_bytes()
        assert len(path.read_bytes().splitlines()) == 4
        reopened.close()

        # ... and when the file is additionally one row behind.
        path.write_bytes(b"\n".join(path.read_bytes().splitlines()[:3]))
        again = self._log(tmp_path)
        assert again.recovered_entries == 1
        report = again.verify()
        assert report["ok"] and report["entries"] == 5 and b"\x00" not in path.read_bytes()

    def test_blank_line_is_reported_not_repaired_into_damage(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        log.close()
        path = tmp_path / "audit.jsonl"
        lines = path.read_bytes().splitlines()
        damaged = lines[0] + b"\n\n" + lines[1] + b"\n" + lines[2][:-30]  # blank line and torn tail
        path.write_bytes(damaged)
        reopened = self._log(tmp_path)
        assert reopened.recovered_entries == 0
        assert path.read_bytes() == damaged  # left exactly as found for verify to flag
        assert not reopened.verify()["ok"]

    def test_repair_is_on_record_even_if_the_file_write_fails(self, tmp_path, monkeypatch):
        log = self._log(tmp_path)
        for _ in range(2):
            self._record(log)
        log.close()
        path = tmp_path / "audit.jsonl"
        path.write_bytes(path.read_bytes()[:-25])  # torn last line
        import builtins

        real_open = builtins.open

        def failing_open(file, mode="r", *args, **kwargs):
            if str(file) == str(path) and "+" in mode:
                raise OSError(28, "No space left on device")
            return real_open(file, mode, *args, **kwargs)

        monkeypatch.setattr(builtins, "open", failing_open)
        reopened = self._log(tmp_path)  # repair entry committed, file write fails
        monkeypatch.setattr(builtins, "open", real_open)
        assert [e["decision_zone"] for e in reopened.entries()][-1] == "AUDIT_MIRROR_REPAIRED"
        self._record(reopened)  # dirty flag triggers a second, successful repair
        assert reopened.verify()["ok"]

    def test_failed_append_is_resynchronised(self, tmp_path, monkeypatch):
        log = self._log(tmp_path)
        self._record(log)
        real_append, calls = log._append_jsonl, {"n": 0}

        def flaky(entries):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError(28, "No space left on device")
            return real_append(entries)

        monkeypatch.setattr(log, "_append_jsonl", flaky)
        lost = self._record(log)  # committed to SQLite, mirror append fails
        assert lost["seq"] == 2 and not log.verify()["ok"]
        self._record(log)  # next write repairs the mirror first
        report = log.verify()
        assert report["ok"] and report["entries"] == 4
        assert [e["decision_zone"] for e in log.entries()][2] == "AUDIT_MIRROR_REPAIRED"

    def test_deleted_mirror_is_not_silently_rebuilt(self, tmp_path):
        """Rewriting SQLite and deleting the file must not yield a clean bill of health."""
        log = self._log(tmp_path)
        for _ in range(4):
            self._record(log)
        log.close()
        (tmp_path / "audit.jsonl").unlink()
        reopened = self._log(tmp_path)
        assert reopened.recovered_entries == 0
        report = reopened.verify()
        assert not report["ok"] and not report["stores_consistent"]

    def test_second_writer_is_refused(self, tmp_path):
        from backend.storage.audit import AuditLockError

        log = self._log(tmp_path)
        with pytest.raises(AuditLockError, match="single writer"):
            self._log(tmp_path)
        log.close()
        self._log(tmp_path).close()  # lock released on close

    def test_restart_does_not_paper_over_a_forged_file(self, tmp_path):
        log = self._log(tmp_path)
        for _ in range(3):
            self._record(log)
        log.close()
        path = tmp_path / "audit.jsonl"
        lines = path.read_text().splitlines()
        forged = json.loads(lines[1])
        forged["hash"] = "f" * 64
        path.write_text("\n".join([lines[0], json.dumps(forged)]) + "\n")
        reopened = self._log(tmp_path)
        assert reopened.recovered_entries == 0 and not reopened.verify()["ok"]

    def test_chain_continues_across_restart(self, tmp_path):
        log = self._log(tmp_path)
        first = self._record(log)
        log.close()
        second = self._record(self._log(tmp_path))
        assert second["prev_hash"] == first["hash"] and second["seq"] == 2

    def test_exports(self, tmp_path):
        log = self._log(tmp_path)
        self._record(log, top1_subject='=HYPERLINK("http://x","y")')
        self._record(log)
        exported = json.loads(log.export_json())
        assert len(exported["entries"]) == 2 and exported["verification"]["ok"]
        import csv
        import io

        rows = list(csv.DictReader(io.StringIO(log.export_csv())))
        assert len(rows) == 2 and rows[0]["seq"] == "1" and rows[1]["hash"] == exported["entries"][1]["hash"]
        assert rows[0]["top1_subject"].startswith("'=")  # spreadsheet formula neutralised
        assert json.loads(rows[0]["thresholds"])["tau_high"] == 0.78

    def test_no_biometric_material_in_log(self, tmp_path):
        log = self._log(tmp_path)
        entry = self._record(log)
        assert not any("embedding" in key or "image" in key for key in entry)


class TestImageDecoding:
    def test_valid_and_invalid_payloads(self):
        image = decode_image(frame_bytes(5), 8_000_000, 16_000_000)
        assert image.shape == (224, 224, 3)
        for bad in (b"", b"not an image", b"\xff\xd8\xff garbage"):
            with pytest.raises(ImageError):
                decode_image(bad, 8_000_000, 16_000_000)

    def test_limits(self):
        data = frame_bytes(5)
        with pytest.raises(ImageError, match="limit"):
            decode_image(data, 10, 16_000_000)
        with pytest.raises(ImageError, match="resolution"):
            decode_image(data, 8_000_000, 100)
        with pytest.raises(ImageError, match="small"):
            decode_image(encode_png(np.zeros((8, 8, 3), np.uint8)), 8_000_000, 16_000_000)

    def test_large_frames_are_downscaled_in_memory(self):
        big = encode_png(np.zeros((1200, 2400, 3), np.uint8))
        assert max(decode_image(big, 8_000_000, 16_000_000).shape[:2]) == 1920


# --------------------------------------------------------------------------- #
# API: the real pipeline, vault and audit log behind stand-in networks
# --------------------------------------------------------------------------- #


def enroll(client, subject: str, identity: int, frames: int = 3, **form):
    files = [("files", (f"{i}.png", frame_bytes(identity), "image/png")) for i in range(frames)]
    return client.post("/api/enroll", data={"subject": subject, **form}, files=files)


def identify(client, identity: int, quality: int = 230, liveness: int = 255, **form):
    return client.post(
        "/api/identify", data=form, files={"file": ("probe.png", frame_bytes(identity, quality, liveness), "image/png")}
    )


def primary(response) -> dict:
    assert response.status_code == 200, response.text
    faces = response.json()["faces"]
    assert len(faces) == 1
    return faces[0]["decision"]


class TestApiRecognition:
    def test_health_and_config(self, client):
        assert client.get("/api/health").json()["engine_ready"] is True
        cfg = client.get("/api/config").json()
        assert cfg["thresholds"] == T.to_dict()
        assert cfg["storage"] == {"cipher": "aes256gcm", "key_source": "generated key file", "raw_images_stored": False}

    def test_match_confirmed(self, client):
        assert enroll(client, "Alice", 11).json()["template_complete"] is True
        enroll(client, "Bob", 42)
        decision = primary(identify(client, 11))
        assert decision["zone"] == "MATCH_CONFIRMED" and decision["subject"] == "Alice"
        assert decision["top2"]["subject"] == "Bob" and decision["margin"] > 0.08

    def test_lookalikes_trigger_narrow_margin(self, client):
        enroll(client, "Twin A", 21)
        enroll(client, "Twin B", 22)  # same family -> cosine ~0.94
        decision = primary(identify(client, 21))
        assert decision["zone"] == "UNCERTAIN_REVIEW"
        assert decision["reason"] == "NARROW_CANDIDATE_MARGIN"
        assert decision["subject"] is None
        assert decision["top1"]["score"] >= 0.78 and decision["margin"] < 0.08

    def test_low_quality_frame_abstains(self, client):
        enroll(client, "Alice", 11)
        decision = primary(identify(client, 11, quality=70))
        assert decision["zone"] == "UNCERTAIN_REVIEW" and decision["reason"] == "LOW_IMAGE_QUALITY"
        assert decision["subject"] is None

    def test_unknown_face_is_no_match(self, client):
        enroll(client, "Alice", 11)
        decision = primary(identify(client, 95))
        assert decision["zone"] == "NO_MATCH" and decision["subject"] is None

    def test_spoof_is_rejected(self, client):
        enroll(client, "Alice", 11)
        decision = primary(identify(client, 11, liveness=40))
        assert decision["zone"] == "SPOOF_DETECTED" and decision["subject"] is None

    def test_no_face(self, client):
        response = identify(client, 0)
        assert response.status_code == 200 and response.json()["faces"] == []

    def test_empty_gallery(self, client):
        decision = primary(identify(client, 11))
        assert decision["zone"] == "NO_MATCH" and decision["reason"] == "EMPTY_GALLERY"

    def test_verify_mode(self, client):
        enroll(client, "Twin A", 21)
        enroll(client, "Twin B", 22)
        ok = client.post("/api/verify", data={"subject": "Twin A"},
                         files={"file": ("p.png", frame_bytes(21), "image/png")})
        assert primary(ok)["zone"] == "MATCH_CONFIRMED"  # 1:1 has no margin rule
        other = client.post("/api/verify", data={"subject": "Twin A"},
                            files={"file": ("p.png", frame_bytes(95), "image/png")})
        assert primary(other)["zone"] == "NO_MATCH"
        missing = client.post("/api/verify", data={"subject": "Nobody"},
                              files={"file": ("p.png", frame_bytes(21), "image/png")})
        assert missing.status_code == 404

    def test_baseline_comparison_exposes_the_false_positive(self, client):
        enroll(client, "Twin B", 22)  # only the lookalike is on file
        enroll(client, "Carol", 63)
        face = identify(client, 21, quality=70, compare_baseline="true").json()["faces"][0]
        assert face["baseline"]["zone"] == "MATCH_CONFIRMED" and face["baseline"]["subject"] == "Twin B"
        assert face["decision"]["zone"] == "UNCERTAIN_REVIEW"
        assert "baseline" not in identify(client, 21).json()["faces"][0]

    def test_pair_comparison_stores_nothing(self, client):
        response = client.post("/api/verify/pair", files={
            "file_a": ("a.png", frame_bytes(11), "image/png"), "file_b": ("b.png", frame_bytes(11), "image/png")})
        assert response.json()["decision"]["zone"] == "MATCH_CONFIRMED"
        assert client.get("/api/subjects").json()["count"] == 0
        # ... but the decision itself is a recognition event and is audited.
        entry = client.get("/api/audit").json()["entries"][0]
        assert entry["source"] == "api:verify-pair" and entry["seq"] == response.json()["audit"][0]["seq"]

    def test_large_uploads_stay_in_memory(self, client):
        """Starlette spools parts above 1 MB to a temp file unless told otherwise."""
        import tempfile

        from starlette.formparsers import MultiPartParser

        from backend.config import load_settings

        assert MultiPartParser.spool_max_size > load_settings().max_upload_bytes
        noisy = np.random.default_rng(0).integers(0, 255, (1100, 1100, 3), dtype=np.uint8)
        noisy[400:700, 400:700] = (11, 230, 255)  # the stand-in detector samples the centre
        payload = encode_png(noisy)
        assert len(payload) > 2 * 1024 * 1024
        rolled = []
        original = tempfile.SpooledTemporaryFile.rollover

        def spy(self):
            rolled.append(True)
            return original(self)

        tempfile.SpooledTemporaryFile.rollover = spy
        try:
            response = client.post("/api/identify", files={"file": ("big.png", payload, "image/png")})
        finally:
            tempfile.SpooledTemporaryFile.rollover = original
        assert response.status_code == 200 and rolled == []

    def test_oversized_request_is_refused_early(self, client):
        response = client.post("/api/identify", content=b"x", headers={
            "content-type": "multipart/form-data; boundary=x", "content-length": str(10**9)})
        assert response.status_code == 413

    def test_bad_uploads(self, client):
        response = client.post("/api/identify", files={"file": ("x.png", b"nope", "image/png")})
        assert response.status_code == 422
        assert client.post("/api/identify").status_code == 422

    def test_threshold_update_changes_decisions(self, client):
        enroll(client, "Twin A", 21)
        enroll(client, "Twin B", 22)
        assert client.put("/api/config/thresholds", json={"margin": 0.0}).status_code == 200
        assert primary(identify(client, 21))["zone"] == "MATCH_CONFIRMED"
        assert client.put("/api/config/thresholds", json={"tau_low": 0.99}).status_code == 422
        assert client.get("/api/config").json()["thresholds"]["tau_low"] == 0.48


class TestApiEnrollment:
    def test_rejections_are_explained(self, client):
        files = [
            ("files", ("good.png", frame_bytes(11), "image/png")),
            ("files", ("noface.png", frame_bytes(0), "image/png")),
            ("files", ("blurry.png", frame_bytes(11, quality=70), "image/png")),
            ("files", ("spoof.png", frame_bytes(11, liveness=30), "image/png")),
            ("files", ("good2.png", frame_bytes(11), "image/png")),
        ]
        body = client.post("/api/enroll", data={"subject": "Alice"}, files=files).json()
        assert body["enrolled"] and body["accepted"] == 2 and body["template_complete"] is False
        assert [f["reason"] for f in body["frames"]] == [
            None, "NO_FACE_DETECTED", "LOW_IMAGE_QUALITY", "LIVENESS_BELOW_MINIMUM", None]

    def test_mixed_identities_are_not_merged(self, client):
        files = [("files", (f"{i}.png", frame_bytes(identity), "image/png"))
                 for i, identity in enumerate((11, 11, 95))]
        body = client.post("/api/enroll", data={"subject": "Alice"}, files=files).json()
        assert body["accepted"] == 2
        assert body["frames"][2]["reason"] == "INCONSISTENT_IDENTITY"

    def test_two_conflicting_frames_enroll_nobody(self, client):
        files = [("files", ("a.png", frame_bytes(11), "image/png")), ("files", ("b.png", frame_bytes(95), "image/png"))]
        response = client.post("/api/enroll", data={"subject": "Alice"}, files=files)
        assert response.status_code == 422 and response.json()["enrolled"] is False
        assert client.get("/api/subjects").json()["count"] == 0

    def test_appending_a_different_person_is_refused(self, client):
        enroll(client, "Alice", 11)
        response = enroll(client, "Alice", 95, frames=1)
        assert response.status_code == 422
        assert response.json()["frames"][0]["reason"] == "INCONSISTENT_IDENTITY"

    def test_nothing_usable_is_a_422(self, client):
        response = enroll(client, "Ghost", 0)
        assert response.status_code == 422 and response.json()["enrolled"] is False

    def test_duplicate_identity_warning(self, client):
        enroll(client, "Alice", 11)
        assert enroll(client, "Alice Again", 11).json()["possible_duplicate_of"] == "Alice"

    def test_sample_cap_and_replace(self, client):
        enroll(client, "Alice", 11, frames=5)
        assert enroll(client, "Alice", 11, frames=3).json()["samples"] == 8
        for _ in range(3):  # later check-ins roll over the oldest photos
            enroll(client, "Alice", 11, frames=5)
        assert client.get("/api/subjects").json()["subjects"][0]["samples"] == 15
        assert enroll(client, "Alice", 11, frames=1, replace="true").json()["samples"] == 1

    def test_incomplete_template_takes_no_part_in_matching(self, client):
        """Fewer than three frames: stored, but not matched against (single-pose bias)."""
        body = enroll(client, "Alice", 11, frames=2).json()
        assert body["enrolled"] and body["template_complete"] is False and body["active"] is False
        pending = primary(identify(client, 11))
        assert pending["zone"] == "UNCERTAIN_REVIEW" and pending["reason"] == "INCOMPLETE_ENROLLMENT"
        assert pending["subject"] is None
        refused = client.post("/api/verify", data={"subject": "Alice"},
                              files={"file": ("p.png", frame_bytes(11), "image/png")})
        assert refused.status_code == 409 and "enrollment samples" in refused.json()["detail"]
        listed = client.get("/api/subjects").json()["subjects"][0]
        assert listed["samples"] == 2 and listed["active"] is False
        # One more frame completes the template and activates it.
        assert enroll(client, "Alice", 11, frames=1).json()["active"] is True
        assert primary(identify(client, 11))["subject"] == "Alice"

    def test_half_enrolled_lookalike_still_narrows_the_margin(self, client):
        """An incomplete template must not vanish from the ambiguity test."""
        enroll(client, "Twin A", 21, frames=3)
        second = enroll(client, "Twin B", 22, frames=2).json()
        assert second["possible_duplicate_of"] == "Twin A" and second["active"] is False
        decision = primary(identify(client, 22))  # Twin B's own face
        assert decision["zone"] == "UNCERTAIN_REVIEW" and decision["subject"] is None
        assert decision["reason"] == "NARROW_CANDIDATE_MARGIN"
        assert "INCOMPLETE_ENROLLMENT" in decision["reasons"]

    def test_bystander_anywhere_in_the_ranking_blocks_enrollment(self, data_dir, settings):
        """A large face near the edge ranks below small central ones; it must still count."""
        from backend.engine.detector import Detection
        from backend.engine.pipeline import RecognitionPipeline
        from backend.storage.vault import BiometricVault
        from tests.conftest import FakeEmbedder, FakeLiveness, test_calibrator

        def face(x1, y1, size):
            scale = size / 112.0
            return Detection(np.array([x1, y1, x1 + size, y1 + size], dtype=np.float32), 0.95,
                             (ARCFACE_REFERENCE * scale + [x1, y1]).astype(np.float32))

        class Crowd:
            def __init__(self, faces):
                self.faces = faces

            def detect(self, image, max_faces=0):
                return self.faces

        vault = BiometricVault(data_dir / "b.db")
        frame = np.full((480, 640, 3), (11, 230, 255), dtype=np.uint8)
        # Primary (central, 120 px), a small face, then a 280 px face ranked last.
        crowd = RecognitionPipeline(Crowd([face(260, 180, 120), face(100, 200, 60), face(350, 10, 280)]),
                                    FakeEmbedder(), FakeLiveness(), test_calibrator(), vault, settings)
        result = crowd.enroll("Alice", [frame], Thresholds())
        assert result["enrolled"] is False and result["frames"][0]["reason"] == "MULTIPLE_FACES"
        # Small faces in the background do not block enrollment.
        fine = RecognitionPipeline(Crowd([face(260, 180, 120), face(100, 200, 40)]),
                                   FakeEmbedder(), FakeLiveness(), test_calibrator(), vault, settings)
        assert fine.enroll("Alice", [frame], Thresholds())["enrolled"] is True

    def test_validation(self, client):
        assert enroll(client, "   ", 11).status_code == 422
        assert enroll(client, "Alice", 11, frames=6).status_code == 422

    def test_subject_management(self, client):
        enroll(client, "Alice", 11)
        enroll(client, "Bob", 42)
        assert client.put("/api/subjects/Bob", json={"subject": "Alice"}).status_code == 409
        assert client.put("/api/subjects/Bob", json={"subject": "Robert"}).json()["subject"] == "Robert"
        assert primary(identify(client, 42))["subject"] == "Robert"
        assert client.delete("/api/subjects/Robert").json()["samples_deleted"] == 3
        assert client.delete("/api/subjects/Robert").status_code == 404
        assert primary(identify(client, 42))["zone"] == "NO_MATCH"
        assert client.delete("/api/subjects").status_code == 400
        assert client.delete("/api/subjects", params={"confirm": "true"}).json() == {"deleted": 1}


class TestEndToEndPrivacy:
    def test_only_ciphertext_and_no_images_on_disk(self, client, data_dir):
        enroll(client, "Alice", 11)
        identify(client, 11)
        identify(client, 95)
        files = sorted(p.name for p in data_dir.rglob("*") if p.is_file())
        allowed = {"biometrics.db", "biometrics.db-wal", "biometrics.db-shm", "audit.jsonl", "audit.jsonl.lock",
                   "biometric.key"}
        assert set(files) <= allowed, files
        for path in data_dir.iterdir():
            assert path.stat().st_mode & 0o077 == 0, f"{path.name} is readable by other users"
        image_magic = (b"\x89PNG", b"\xff\xd8\xff", b"GIF8", b"RIFF", b"BM")
        plaintext = identity_vector(11, 0).astype(np.float32).tobytes()[:48]
        for path in data_dir.rglob("*"):
            if path.is_file():
                content = path.read_bytes()
                assert not content.startswith(image_magic), path
                assert plaintext not in content, path
                assert frame_bytes(11)[:64] not in content, path
        conn = sqlite3.connect(data_dir / "biometrics.db")
        columns = {row[1]: row[2] for row in conn.execute("PRAGMA table_info(samples)")}
        assert columns["embedding"] == "BLOB"
        blobs = [row[0] for row in conn.execute("SELECT embedding FROM samples")]
        assert len(blobs) == 3 and all(blob[:1] == b"\x01" for blob in blobs)
        # Ciphertext should look uniformly random, unlike a float32 vector.
        counts = np.bincount(np.frombuffer(b"".join(blobs), dtype=np.uint8), minlength=256)
        assert counts.min() > 0 and counts.max() < 4 * counts.mean()

    def test_audit_log_passes_chain_validation(self, client, data_dir):
        enroll(client, "Alice", 11)
        enroll(client, "Twin A", 21)
        enroll(client, "Twin B", 22)
        outcomes = [primary(identify(client, i, q))["zone"] for i, q in ((11, 230), (21, 230), (11, 70), (95, 230))]
        assert outcomes == ["MATCH_CONFIRMED", "UNCERTAIN_REVIEW", "UNCERTAIN_REVIEW", "NO_MATCH"]
        report = client.get("/api/audit/verify").json()
        assert report["ok"] and report["entries"] == 7  # 3 enrollments + 4 decisions
        # Independent re-verification straight from the JSONL file.
        prev = GENESIS_HASH
        lines = (data_dir / "audit.jsonl").read_text().splitlines()
        for line in lines:
            entry = json.loads(line)
            assert entry["prev_hash"] == prev
            assert entry["hash"] == entry_hash({k: v for k, v in entry.items() if k != "hash"})
            prev = entry["hash"]
        assert prev == report["head_hash"]
        everything = [json.loads(line) for line in lines]
        assert [e["decision_zone"] for e in everything[:3]] == ["SUBJECT_ENROLLED"] * 3
        recorded = [e for e in everything if e["mode"] != "admin"]
        assert [e["decision_zone"] for e in recorded] == outcomes
        assert recorded[1]["abstention_reason"] == "NARROW_CANDIDATE_MARGIN"
        assert recorded[1]["asserted_subject"] is None and recorded[0]["asserted_subject"] == "Alice"
        recent = client.get("/api/audit", params={"limit": 2}).json()
        assert [e["seq"] for e in recent["entries"]] == [7, 6]
        assert recent["total"] == 7 and recent["head_hash"] == report["head_hash"]

    def test_tampering_is_reported_by_the_api(self, client, data_dir):
        enroll(client, "Alice", 11)
        identify(client, 11)
        identify(client, 11)
        path = data_dir / "audit.jsonl"
        path.write_text(path.read_text().replace("MATCH_CONFIRMED", "NO_MATCH", 1))
        assert client.get("/api/audit/verify").json()["ok"] is False

    def test_exports(self, client):
        enroll(client, "Alice", 11)
        identify(client, 11)
        exported = client.get("/api/audit/export", params={"format": "json"})
        assert exported.headers["content-type"].startswith("application/json")
        assert [e["decision_zone"] for e in exported.json()["entries"]] == ["SUBJECT_ENROLLED", "MATCH_CONFIRMED"]
        as_csv = client.get("/api/audit/export", params={"format": "csv"})
        assert as_csv.headers["content-type"].startswith("text/csv") and as_csv.text.startswith("seq,")
        assert client.get("/api/audit/export", params={"format": "xml"}).status_code == 422


class TestApiFairnessAndRooms:
    def test_fairness_audit_and_stress(self, client):
        report = client.get("/api/fairness/audit", params={"tau_high": 0.8}).json()
        assert report["thresholds"]["tau_high"] == 0.8 and set(report["fair"]["groups"]) == set("I II III IV V VI".split())
        assert client.get("/api/fairness/audit", params={"tau_high": 0.3}).status_code == 422
        stress = client.post("/api/stress/lookalike", json={"similarity": 0.9, "trials": 50}).json()
        assert stress["fair"]["outcomes"]["MATCH_B"] == 0
        assert client.post("/api/stress/lookalike", json={"group": "X"}).status_code == 422
        rfw = client.get("/api/fairness/rfw").json()
        assert "available" in rfw

    def test_room_creation_and_qr(self, client):
        room = client.post("/api/rooms", json={"base_url": "https://faces.example.org"}).json()
        assert room["join_url"] == f"https://faces.example.org/?room={room['room_id']}"
        assert room["qr_svg"].lstrip().startswith("<?xml") or "<svg" in room["qr_svg"]
        assert room["warnings"] == []
        assert client.get(f"/api/rooms/{room['room_id']}").json()["peers"] == []
        assert client.get("/api/rooms/unknown").status_code == 404
        local = client.post("/api/rooms", json={"base_url": "http://localhost:8000"}).json()
        # Bound to loopback (the default): say so, and do not offer LAN links that cannot work.
        assert any("only listens on this machine" in w for w in local["warnings"]) and local["lan_origins"] == []
        plain = client.post("/api/rooms", json={"base_url": "http://192.168.1.20:8000"}).json()
        assert any("HTTPS" in w for w in plain["warnings"])
        assert client.post("/api/rooms", json={"base_url": "javascript:alert(1)"}).status_code == 422

    def test_security_headers(self, client):
        headers = client.get("/api/health").headers
        assert headers["x-content-type-options"] == "nosniff"
        assert "default-src 'self'" in headers["content-security-policy"]
        assert headers["cache-control"] == "no-store"
        # Never framed by default.
        assert headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in headers["content-security-policy"]

    def test_framing_is_allowed_only_for_named_https_origins(self, data_dir, monkeypatch):
        from fastapi.testclient import TestClient

        from backend.main import create_app

        monkeypatch.setenv("FAIRFACE_FRAME_ANCESTORS",
                           "https://huggingface.co http://plain.example https://x.example;script-src *")
        with TestClient(create_app(None)) as framed:
            headers = framed.get("/api/health").headers
        assert "x-frame-options" not in headers
        assert "frame-ancestors https://huggingface.co;" in headers["content-security-policy"]
        assert "plain.example" not in headers["content-security-policy"]  # not HTTPS
        assert "script-src *" not in headers["content-security-policy"]  # no header injection


class TestRequestGuard:
    """Browser-borne attacks on a localhost API (cross-site writes, DNS rebinding)."""

    def _enroll(self, client, **headers):
        files = [("files", (f"{i}.png", frame_bytes(95), "image/png")) for i in range(3)]
        return client.post("/api/enroll", data={"subject": "Alice", "replace": "true"}, files=files, headers=headers)

    def test_cross_site_page_cannot_replace_a_template(self, client):
        enroll(client, "Alice", 11)
        assert self._enroll(client, origin="https://evil.example").status_code == 403
        assert self._enroll(client, **{"sec-fetch-site": "cross-site"}).status_code == 403
        assert self._enroll(client, origin="null").status_code == 403
        assert primary(identify(client, 11))["subject"] == "Alice"  # template untouched
        assert primary(identify(client, 95))["zone"] == "NO_MATCH"

    def test_same_origin_and_non_browser_clients_still_work(self, client):
        assert self._enroll(client, origin="http://testserver").status_code == 200
        assert self._enroll(client, **{"sec-fetch-site": "same-origin"}).status_code == 200
        assert self._enroll(client).status_code == 200

    def test_cross_site_reads_of_public_status_are_harmless_but_writes_are_not(self, client):
        assert client.get("/api/health", headers={"origin": "https://evil.example"}).status_code == 200
        assert client.put("/api/config/thresholds", json={"tau_high": 0.5},
                          headers={"origin": "https://evil.example"}).status_code == 403
        assert client.delete("/api/subjects", params={"confirm": "true"},
                             headers={"origin": "https://evil.example"}).status_code == 403

    def test_unknown_host_header_is_refused(self, client):
        response = client.get("/api/health", headers={"host": "attacker-rebind.example"})
        assert response.status_code == 421 and "FAIRFACE_ALLOWED_HOSTS" in response.json()["detail"]
        assert client.get("/", headers={"host": "attacker-rebind.example"}).status_code == 421
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/camera", headers={
                    "host": "attacker-rebind.example", "origin": "http://attacker-rebind.example"}):
                pass

    def test_administrative_actions_are_audited(self, client):
        enroll(client, "Alice", 11)
        client.put("/api/config/thresholds", json={"tau_high": 0.9})
        client.put("/api/subjects/Alice", json={"subject": "Alicia"})
        client.delete("/api/subjects/Alicia")
        identify(client, 0)  # nobody in frame
        events = [(e["decision_zone"], e["top1_subject"]) for e in reversed(client.get("/api/audit").json()["entries"])]
        assert events == [("SUBJECT_ENROLLED", "Alice"), ("THRESHOLDS_CHANGED", None), ("SUBJECT_RENAMED", "Alicia"),
                          ("SUBJECT_DELETED", "Alicia"), ("NO_FACE_DETECTED", None)]
        assert client.get("/api/audit/verify").json()["ok"]

    def test_image_bombs_are_rejected_from_the_header(self, monkeypatch):
        """Huge declared canvases must be refused without decoding, in every format.

        The decoder is replaced by a tripwire: reaching it at all is the failure,
        so this cannot pass by accident through some later decode error.
        """
        import struct
        import zlib

        from backend.engine import pipeline

        def chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

        side = 30000
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(b"\x00" * 4096, 9)) + chunk(b"IEND", b""))
        jpeg = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
                + b"\xff\xc0" + struct.pack(">HBHHB", 17, 8, side, side, 3) + b"\x00" * 9 + b"\xff\xd9")
        bmp = b"BM" + b"\x00" * 16 + struct.pack("<ii", side, -side) + b"\x00" * 28
        gif = b"GIF89a" + struct.pack("<HH", side, side) + b"\x00" * 20
        webp = b"RIFF" + struct.pack("<I", 22) + b"WEBP" + b"VP8X" + struct.pack("<I", 10) + b"\x00" * 4 \
            + (side - 1).to_bytes(3, "little") + (side - 1).to_bytes(3, "little")

        def tripwire(*args, **kwargs):
            raise AssertionError("the decoder must not be reached for an over-limit image")

        monkeypatch.setattr(pipeline.cv2, "imdecode", tripwire)
        for name, payload in (("png", png), ("jpeg", jpeg), ("bmp", bmp), ("gif", gif), ("webp", webp)):
            assert pipeline.declared_size(payload) == (side, side), name
            with pytest.raises(ImageError, match="resolution"):
                decode_image(payload, 8_000_000, 4096 * 4096)
        # Containers whose dimensions cannot be read up front are refused outright.
        tiff = b"II*\x00" + b"\x00" * 64
        with pytest.raises(ImageError, match="unsupported"):
            decode_image(tiff, 8_000_000, 4096 * 4096)

    def test_declared_size_matches_real_encoders(self):
        from backend.engine.pipeline import declared_size

        image = np.random.default_rng(0).integers(0, 255, (37, 91, 3), dtype=np.uint8)
        for extension in (".png", ".jpg", ".bmp", ".webp"):
            ok, buffer = cv2.imencode(extension, image)
            assert ok and declared_size(buffer.tobytes()) == (91, 37), extension
            assert decode_image(buffer.tobytes(), 8_000_000, 4096 * 4096).shape == (37, 91, 3)
        progressive = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_PROGRESSIVE, 1])[1].tobytes()
        assert declared_size(progressive) == (91, 37)
        assert declared_size(b"") is None and declared_size(b"\xff\xd8\xff") is None

    def test_malformed_host_headers_are_refused(self, client):
        for host in ("testserver@evil.example", "testserver, evil.example", "test server", ""):
            assert client.get("/api/health", headers={"host": host}).status_code == 421, host
        assert client.get("/api/health", headers={"host": "testserver:8000"}).status_code == 200

    def test_public_origin_is_trusted_behind_a_proxy(self, data_dir, monkeypatch):
        """A TLS proxy that rewrites Host must not make every write look cross-site."""
        from fastapi.testclient import TestClient

        from backend.main import create_app
        from tests.conftest import fake_pipeline_factory

        monkeypatch.setenv("FAIRFACE_PUBLIC_URL", "https://faces.example.org")
        with TestClient(create_app(fake_pipeline_factory)) as proxied:
            ok = proxied.put("/api/config/thresholds", json={"tau_high": 0.8},
                             headers={"origin": "https://faces.example.org"})
            assert ok.status_code == 200
            assert proxied.put("/api/config/thresholds", json={"tau_high": 0.8},
                               headers={"origin": "https://evil.example"}).status_code == 403
            assert proxied.get("/api/health", headers={"host": "faces.example.org"}).status_code == 200


class TestApiKey:
    def test_key_is_enforced_when_configured(self, data_dir, monkeypatch):
        from fastapi.testclient import TestClient

        from backend.main import create_app
        from tests.conftest import fake_pipeline_factory

        monkeypatch.setenv("FAIRFACE_API_KEY", "s3cret")
        with TestClient(create_app(fake_pipeline_factory)) as client:
            assert client.get("/api/health").status_code == 200
            assert client.get("/api/subjects").status_code == 401
            assert client.get("/api/subjects", headers={"X-API-Key": "wrong"}).status_code == 401
            assert client.get("/api/subjects", headers={"X-API-Key": "s3cret"}).status_code == 200
            # ICE settings can carry TURN credentials: key or an open room id required.
            assert client.get("/api/rtc-config").status_code == 401
            room_id = client.post("/api/rooms", json={"base_url": "https://faces.example.org"},
                                  headers={"X-API-Key": "s3cret"}).json()["room_id"]
            assert client.get("/api/rtc-config", params={"room": room_id}).status_code == 200
            assert client.get("/api/subjects", headers={"X-API-Key": "caf\u00e9".encode("latin-1")}).status_code == 401
            assert client.get("/api/openapi.json").status_code == 401
            # The key is sent as the first message, never in the URL.
            with client.websocket_connect("/ws/camera") as ws:
                ws.send_bytes(b"\x00" + frame_bytes(11))
                assert ws.receive_json()["code"] == "unauthorized"
            with client.websocket_connect("/ws/camera?api_key=s3cret") as ws:
                ws.send_text(json.dumps({"type": "auth", "api_key": "wrong"}))
                assert ws.receive_json()["code"] == "unauthorized"
            with client.websocket_connect("/ws/camera") as ws:
                ws.send_text(json.dumps({"type": "auth", "api_key": "s3cret"}))
                assert ws.receive_json()["type"] == "ready"
            # Server-camera preview: header key or a short-lived single-use token.
            keyed = {"X-API-Key": "s3cret"}
            assert client.get("/api/server-camera/stream.mjpg").status_code == 401
            assert client.get("/api/server-camera/stream.mjpg", params={"token": "guess"}).status_code == 401
            token = client.post("/api/server-camera/stream-token", headers=keyed).json()["token"]
            assert client.get("/api/server-camera/stream.mjpg", params={"token": token}).status_code == 409
            assert client.get("/api/server-camera/stream.mjpg", params={"token": token}).status_code == 401


class TestWebSockets:
    def test_camera_stream_roundtrip(self, client):
        enroll(client, "Alice", 11)
        with client.websocket_connect("/ws/camera") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "ready" and ready["engine_ready"] is True
            ws.send_bytes(b"\x00" + frame_bytes(11))
            local = ws.receive_json()
            assert local["type"] == "result" and local["stream"] == "local" and local["seq"] == 1
            assert local["faces"][0]["decision"]["subject"] == "Alice"
            assert {"detect", "embed", "total"} <= set(local["timing_ms"])
            ws.send_bytes(b"\x01" + frame_bytes(95))
            remote = ws.receive_json()
            assert remote["stream"] == "remote" and remote["faces"][0]["decision"]["zone"] == "NO_MATCH"

    def test_camera_config_and_errors(self, client):
        enroll(client, "Alice", 11)
        with client.websocket_connect("/ws/camera") as ws:
            ws.receive_json()
            ws.send_text(json.dumps({"type": "config", "mode": "verify", "claimed_subject": "Alice",
                                     "compare_baseline": True}))
            assert ws.receive_json() == {"type": "config", "mode": "verify", "claimed_subject": "Alice",
                                         "compare_baseline": True}
            ws.send_bytes(b"\x00" + frame_bytes(11))
            face = ws.receive_json()["faces"][0]
            assert face["decision"]["mode"] == "verify" and "baseline" in face
            ws.send_text(json.dumps({"type": "config", "mode": "verify"}))
            assert ws.receive_json()["code"] == "bad-config"
            ws.send_text("not json")
            assert ws.receive_json()["code"] == "bad-config"
            ws.send_bytes(b"\x07" + frame_bytes(11))
            assert ws.receive_json()["code"] == "bad-frame"
            ws.send_bytes(b"\x00not-an-image")
            assert ws.receive_json()["code"] == "bad-frame"
            ws.send_text(json.dumps({"type": "config", "mode": "verify", "claimed_subject": "Nobody"}))
            ws.receive_json()
            ws.send_bytes(b"\x00" + frame_bytes(11))
            assert ws.receive_json()["code"] == "unknown-subject"
            # The socket survives every one of those errors.
            ws.send_text(json.dumps({"type": "config", "mode": "identify"}))
            ws.receive_json()
            ws.send_bytes(b"\x00" + frame_bytes(11))
            assert ws.receive_json()["type"] == "result"

    def test_stream_decisions_are_audited_without_flooding(self, client):
        enroll(client, "Alice", 11)
        with client.websocket_connect("/ws/camera") as ws:
            ws.receive_json()
            for _ in range(5):
                ws.send_bytes(b"\x00" + frame_bytes(11))
                ws.receive_json()
            ws.send_bytes(b"\x00" + frame_bytes(95))
            ws.receive_json()
        entries = [e for e in client.get("/api/audit").json()["entries"] if e["mode"] != "admin"]
        assert [e["decision_zone"] for e in entries] == ["NO_MATCH", "MATCH_CONFIRMED"]
        assert entries[0]["source"] == "stream:local"
        assert client.get("/api/audit/verify").json()["ok"]

    def test_cross_origin_websocket_is_refused(self, client):
        from starlette.websockets import WebSocketDisconnect

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/ws/camera", headers={"origin": "https://evil.example"}):
                pass

    def test_room_signaling_relay(self, client):
        created = client.post("/api/rooms", json={"base_url": "https://faces.example.org"}).json()
        room_id, token = created["room_id"], created["host_token"]
        # The join link alone must not be enough to take the host seat.
        with client.websocket_connect(f"/ws/room/{room_id}?role=host") as impostor:
            assert impostor.receive_json()["code"] == "not-host"
        with client.websocket_connect(f"/ws/room/{room_id}?role=host&token={token}") as host:
            assert host.receive_json() == {"type": "joined", "role": "host", "peer_present": False}
            host.send_text(json.dumps({"type": "offer", "sdp": "x"}))
            assert host.receive_json()["code"] == "peer-absent"
            with client.websocket_connect(f"/ws/room/{room_id}?role=guest") as guest:
                assert guest.receive_json() == {"type": "joined", "role": "guest", "peer_present": True}
                assert host.receive_json() == {"type": "peer-joined", "role": "guest"}
                host.send_text(json.dumps({"type": "offer", "sdp": "v=0..."}))
                assert guest.receive_json() == {"type": "offer", "sdp": "v=0...", "from": "host"}
                guest.send_text(json.dumps({"type": "answer", "sdp": "v=0!"}))
                assert host.receive_json() == {"type": "answer", "sdp": "v=0!", "from": "guest"}
                guest.send_text(json.dumps({"type": "candidate", "candidate": {"candidate": "c"}}))
                assert host.receive_json()["type"] == "candidate"
                guest.send_text(json.dumps({"type": "shutdown-server"}))
                assert guest.receive_json()["code"] == "unsupported-message"
                guest.send_text("x" * 70000)
                assert guest.receive_json()["code"] == "message-too-large"
                with client.websocket_connect(f"/ws/room/{room_id}?role=guest") as third:
                    assert third.receive_json()["code"] == "room-full"
            assert host.receive_json() == {"type": "peer-left", "role": "guest"}

    def test_closing_a_room_revokes_the_link(self, client):
        room_id = client.post("/api/rooms", json={"base_url": "https://faces.example.org"}).json()["room_id"]
        with client.websocket_connect(f"/ws/room/{room_id}?role=guest") as guest:
            guest.receive_json()
            assert client.delete(f"/api/rooms/{room_id}").json() == {"room_id": room_id, "closed": True}
            assert guest.receive_json() == {"type": "error", "code": "room-closed"}
        assert client.get(f"/api/rooms/{room_id}").status_code == 404
        assert client.delete(f"/api/rooms/{room_id}").status_code == 404
        with client.websocket_connect(f"/ws/room/{room_id}?role=guest") as late:
            assert late.receive_json()["code"] == "room-not-found"

    def test_unknown_room(self, client):
        with client.websocket_connect("/ws/room/does-not-exist?role=guest") as ws:
            assert ws.receive_json()["code"] == "room-not-found"


class TestServerCamera:
    """The optional OpenCV/RTSP worker, driven by a scripted capture device."""

    class ScriptedCapture:
        def __init__(self, frames, fail_after=None):
            self.frames, self.index, self.fail_after, self.released = frames, 0, fail_after, False

        def isOpened(self):
            return True

        def read(self):
            import time

            time.sleep(0.002)
            if self.fail_after is not None and self.index >= self.fail_after:
                return False, None
            frame = self.frames[self.index % len(self.frames)]
            self.index += 1
            return True, frame

        def release(self):
            self.released = True

    def _wait(self, predicate, timeout=5.0):
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_latest_frame_wins_and_frames_are_skipped(self):
        import time

        from backend.camera import ServerCamera

        capture = self.ScriptedCapture([make_frame(11)])
        analysed = []

        def analyse(frame):
            time.sleep(0.05)  # slower than the camera
            analysed.append(int(frame[0, 0, 0]))
            return {"type": "result", "faces": []}

        camera = ServerCamera(0, analyse, open_capture=lambda source: capture)
        camera.start()
        assert self._wait(lambda: camera.frames_analysed >= 3)
        status = camera.status()
        assert status["running"] and status["state"] == "running"
        assert status["frames_read"] > status["frames_analysed"] and status["frames_skipped"] > 0
        assert camera.latest_result()["seq"] >= 3
        _, jpeg = camera.next_preview(0)
        assert jpeg[:3] == b"\xff\xd8\xff"
        camera.stop()
        assert not camera.running and capture.released and camera.latest_result() is None

    def test_unavailable_source_reports_instead_of_crashing(self):
        from backend.camera import ServerCamera

        class Closed:
            def isOpened(self):
                return False

            def release(self):
                pass

        camera = ServerCamera("rtsp://user:secret@cam.local/stream", lambda f: {}, open_capture=lambda s: Closed())
        camera.start()
        assert self._wait(lambda: camera.state == "unavailable")
        status = camera.status()
        assert status["error"] and "secret" not in status["source"]
        camera.stop()

    def test_stream_drop_triggers_reconnect(self):
        from backend.camera import ServerCamera

        opened = []

        def open_capture(source):
            opened.append(self.ScriptedCapture([make_frame(11)], fail_after=3 if not opened else None))
            return opened[-1]

        camera = ServerCamera(0, lambda f: {"type": "result"}, open_capture=open_capture)
        camera.start()
        assert self._wait(lambda: camera.state == "reconnecting")
        assert opened[0].released
        camera.stop()

    def test_api_requires_configuration(self, client):
        assert client.get("/api/server-camera").json() == {"configured": False, "running": False, "state": "stopped"}
        assert client.post("/api/server-camera/start", json={}).status_code == 409
        assert client.get("/api/server-camera/stream.mjpg").status_code == 409
        assert client.post("/api/server-camera/stop").json()["running"] is False


# --------------------------------------------------------------------------- #
# Model provisioning
# --------------------------------------------------------------------------- #


class TestModelProvisioning:
    def _spec(self, payload: bytes, **overrides):
        import hashlib

        from backend.engine.model_manager import ModelSpec

        fields = dict(
            role="detector", filename="tiny.onnx", sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
            urls=("https://models.invalid/tiny.onnx",), license="test", commercial_use=True, source="test",
            description="test", aliases=("other-name.onnx",),
        )
        fields.update(overrides)
        return ModelSpec(**fields)

    @pytest.fixture()
    def no_network(self, monkeypatch):
        from backend.engine import model_manager

        calls = []

        def refuse(url, offset):
            calls.append(url)
            raise OSError("network disabled in tests")

        monkeypatch.setattr(model_manager, "_open", refuse)
        monkeypatch.setattr(model_manager, "DOWNLOAD_ATTEMPTS", 1)
        monkeypatch.setattr(model_manager, "local_search_dirs", lambda: [])
        return calls

    def test_registry_is_well_formed(self):
        from backend.engine.model_manager import REGISTRY

        assert set(REGISTRY) == {"detector", "arcface", "adaface", "liveness_v2", "liveness_v1se"}
        for spec in REGISTRY.values():
            assert len(spec.sha256) == 64 and int(spec.sha256, 16) >= 0
            assert spec.size > 100_000 and spec.urls and spec.license and spec.source
            assert all(url.startswith("https://") for url in spec.urls)
        # Licence flags must not overstate: only the liveness nets are permissive.
        assert {r for r, s in REGISTRY.items() if s.commercial_use} == {"liveness_v2", "liveness_v1se"}

    def test_present_file_is_used_without_network(self, tmp_path, no_network):
        from backend.engine.model_manager import resolve

        payload = b"model-bytes" * 100
        (tmp_path / "tiny.onnx").write_bytes(payload)
        events = []
        path = resolve(self._spec(payload), tmp_path, lambda *a: events.append(a), offline=False)
        assert path == tmp_path / "tiny.onnx" and events[0][0] == "present" and no_network == []

    def test_existing_installation_is_reused_by_hash(self, tmp_path, no_network, monkeypatch):
        from backend.engine import model_manager

        payload = b"model-bytes" * 100
        cache = tmp_path / "cache" / "models" / "pack"
        cache.mkdir(parents=True)
        (cache / "other-name.onnx").write_bytes(payload)
        (cache / "tiny.onnx").write_bytes(b"same name, different content")
        monkeypatch.setattr(model_manager, "local_search_dirs", lambda: [tmp_path / "cache"])
        path = model_manager.resolve(self._spec(payload), tmp_path / "models", lambda *a: None, offline=False)
        assert path == cache / "other-name.onnx" and no_network == []

    def test_corrupt_file_is_never_loaded(self, tmp_path, no_network):
        from backend.engine.model_manager import ModelProvisioningError, resolve

        payload = b"model-bytes" * 100
        (tmp_path / "tiny.onnx").write_bytes(payload[:-1] + b"X")
        with pytest.raises(ModelProvisioningError):
            resolve(self._spec(payload), tmp_path, lambda *a: None, offline=False)
        assert not (tmp_path / "tiny.onnx").exists() and len(no_network) == 1

    def test_offline_mode_explains_what_is_missing(self, tmp_path, no_network):
        from backend.engine.model_manager import ModelProvisioningError, resolve

        with pytest.raises(ModelProvisioningError, match="downloads are disabled"):
            resolve(self._spec(b"x" * 10), tmp_path, lambda *a: None, offline=True)
        assert no_network == []

    def test_download_verifies_digest(self, tmp_path, monkeypatch):
        import io

        from backend.engine import model_manager

        payload = b"genuine-model" * 50
        served = {"body": payload}

        class Response(io.BytesIO):
            status = 200
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(model_manager, "_open", lambda url, offset: Response(served["body"]))
        monkeypatch.setattr(model_manager, "local_search_dirs", lambda: [])
        spec = self._spec(payload)
        path = model_manager.resolve(spec, tmp_path, lambda *a: None, offline=False)
        assert path.read_bytes() == payload
        assert sorted(p.name for p in tmp_path.iterdir()) == ["tiny.onnx"]  # no temp files left behind

        path.unlink()
        served["body"] = b"tampered-mode" * 50  # same length, different bytes
        with pytest.raises(model_manager.ModelProvisioningError, match="SHA-256 mismatch"):
            model_manager.resolve(spec, tmp_path, lambda *a: None, offline=False)
        assert list(tmp_path.iterdir()) == []

    def test_dropped_connection_resumes_instead_of_failing(self, tmp_path, monkeypatch):
        import io

        from backend.engine import model_manager

        payload = bytes(range(256)) * 400
        requests = []

        class Response(io.BytesIO):
            headers = {}

            def __init__(self, body, status):
                super().__init__(body)
                self.status = status

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def serve(url, offset):
            requests.append(offset)
            if len(requests) == 1:
                return Response(payload[:30000], 200)  # server hangs up a third of the way in
            return Response(payload[offset:], 206)

        monkeypatch.setattr(model_manager, "_open", serve)
        monkeypatch.setattr(model_manager, "local_search_dirs", lambda: [])
        monkeypatch.setattr(model_manager.time, "sleep", lambda seconds: None)
        path = model_manager.resolve(self._spec(payload), tmp_path, lambda *a: None, offline=False)
        assert path.read_bytes() == payload and requests == [0, 30000]

    def test_oversized_response_is_cut_off(self, tmp_path, monkeypatch):
        import io

        from backend.engine import model_manager

        payload = b"m" * 5000
        written = []

        class Endless(io.BytesIO):
            status, headers = 200, {}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self, n=-1):
                written.append(n)
                return b"x" * min(n, 1 << 20)

        monkeypatch.setattr(model_manager, "_open", lambda url, offset: Endless())
        monkeypatch.setattr(model_manager, "local_search_dirs", lambda: [])
        with pytest.raises(model_manager.ModelProvisioningError, match="more than the expected"):
            model_manager.resolve(self._spec(payload), tmp_path, lambda *a: None, offline=False)
        assert len(written) <= 2 and list(tmp_path.iterdir()) == []

    def test_plain_http_sources_are_refused(self):
        from backend.engine import model_manager

        with pytest.raises(model_manager.ModelProvisioningError, match="non-HTTPS"):
            model_manager._open("http://models.invalid/x.onnx", 0)

    def test_operator_override(self, tmp_path, no_network, monkeypatch):
        from backend.engine.model_manager import ModelProvisioningError, resolve

        custom = tmp_path / "licensed.onnx"
        custom.write_bytes(b"commercially licensed weights")
        monkeypatch.setenv("FAIRFACE_MODEL_DETECTOR", str(custom))
        assert resolve(self._spec(b"x" * 10), tmp_path, lambda *a: None, offline=True) == custom
        monkeypatch.setenv("FAIRFACE_MODEL_DETECTOR", str(tmp_path / "missing.onnx"))
        with pytest.raises(ModelProvisioningError):
            resolve(self._spec(b"x" * 10), tmp_path, lambda *a: None, offline=True)

    def test_adaface_norm_output_is_exposed(self, tmp_path):
        """Graph edit on a miniature network with AdaFace's ``Div(z, ReduceL2(z))`` tail."""
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper

        from backend.engine.model_manager import ADAFACE_NORM_OUTPUT, ModelProvisioningError, adaface_with_norm

        weight = np.random.default_rng(0).standard_normal((4, 6)).astype(np.float32)
        graph = helper.make_graph(
            [
                helper.make_node("MatMul", ["input", "w"], ["z"]),
                helper.make_node("ReduceL2", ["z"], ["z_norm"], axes=[1], keepdims=1),
                helper.make_node("Div", ["z", "z_norm"], ["output"]),
            ],
            "tiny-adaface",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 6])],
            [numpy_helper.from_array(weight, "w")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 9
        source = tmp_path / "ada.onnx"
        onnx.save(model, source)

        derived = adaface_with_norm(source)
        assert isinstance(derived, bytes)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["ada.onnx"]  # nothing derived is written to disk
        session = ort.InferenceSession(derived, providers=["CPUExecutionProvider"])
        assert [o.name for o in session.get_outputs()] == ["output", ADAFACE_NORM_OUTPUT]
        x = np.array([[1.0, -2.0, 0.5, 3.0]], dtype=np.float32)
        unit_vec, norm = session.run(None, {"input": x})
        assert np.linalg.norm(unit_vec) == pytest.approx(1.0, abs=1e-5)
        assert norm.shape == (1, 1) and norm[0, 0] == pytest.approx(np.linalg.norm(x @ weight), rel=1e-5)

        # A graph that does not normalise in-graph is rejected, not silently accepted.
        plain = helper.make_graph(
            [helper.make_node("MatMul", ["input", "w"], ["output"])], "plain",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 4])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 6])],
            [numpy_helper.from_array(weight, "w")],
        )
        plain_model = helper.make_model(plain, opset_imports=[helper.make_opsetid("", 13)])
        plain_model.ir_version = 9
        onnx.save(plain_model, tmp_path / "plain.onnx")
        with pytest.raises(ModelProvisioningError):
            adaface_with_norm(tmp_path / "plain.onnx")


class TestMultiResolutionDetection:
    """Close-up faces are only found by the low-resolution pass."""

    class FakeSession:
        """Reports one face at the image centre, but only at small input sizes."""

        def __init__(self):
            self.sizes = []

        def get_inputs(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name="input", shape=[1, 3, "?", "?"])]

        def get_outputs(self):
            from types import SimpleNamespace

            return [SimpleNamespace(name=str(i)) for i in range(9)]

        def run(self, names, feed):
            size = feed["input"].shape[2]
            self.sizes.append(size)
            outputs = []
            for width in (1, 4, 10):
                for stride in (8, 16, 32):
                    outputs.append(np.zeros(((size // stride) ** 2 * 2, width), dtype=np.float32))
            if size <= 160:
                cells = size // 32
                index = (cells // 2 * cells + cells // 2) * 2
                outputs[2][index, 0] = 0.9
                outputs[5][index] = [2.0, 2.0, 2.0, 2.0]
            return outputs

    def test_small_pass_recovers_frame_filling_face(self):
        from backend.engine.detector import FaceDetector

        session = self.FakeSession()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = FaceDetector(session, det_size=640, det_size_small=160).detect(image)
        assert session.sizes == [640, 160]
        assert len(detections) == 1 and detections[0].score == pytest.approx(0.9)
        # Coordinates are mapped back to the source frame (160 px pass, scale 0.25).
        assert detections[0].width == pytest.approx(4 * 32 * 4, rel=0.01)

        single = self.FakeSession()
        assert FaceDetector(single, det_size=640, det_size_small=0).detect(image) == []
        assert single.sizes == [640]


# --------------------------------------------------------------------------- #
# The real ONNX networks (skipped when weights or sample images are absent)
# --------------------------------------------------------------------------- #

SAMPLES = Path(__file__).resolve().parents[2] / "insightface-master"
GROUP_PHOTO = SAMPLES / "python-package/insightface/data/images/t1.jpg"
BULK = SAMPLES / "cpp-package/inspireface/test_res/data/bulk"


def _real_models_available() -> bool:
    from backend.engine.model_manager import REGISTRY

    models = config.models_dir()
    return all((models / spec.filename).is_file() for spec in REGISTRY.values()) and GROUP_PHOTO.is_file() \
        and (BULK / "Rob_Lowe_0001.jpg").is_file()


@pytest.fixture(scope="module")
def real_engine():
    if not _real_models_available():
        pytest.skip("model weights or sample images are not available locally")
    from backend.engine.model_manager import build_engine

    detector, embedder, liveness, info = build_engine(config.load_settings(), offline=True)
    return {"detector": detector, "embedder": embedder, "liveness": liveness, "info": info}


def _load(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    assert image is not None, path
    return image


class TestRealModels:
    def _embed(self, engine, image):
        detection = engine["detector"].detect(image, max_faces=1)[0]
        crop = align_face(image, detection.landmarks)
        return detection, crop, engine["embedder"].embed(crop)

    def test_detects_every_face_in_a_group_photo(self, real_engine):
        image = _load(GROUP_PHOTO)
        faces = real_engine["detector"].detect(image)
        assert len(faces) == 6
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            assert 0.5 <= face.score <= 1.0 and x2 > x1 and y2 > y1
            assert np.all(face.landmarks[:, 0] > x1 - 5) and np.all(face.landmarks[:, 0] < x2 + 5)
            left_eye, right_eye, nose, left_mouth, right_mouth = face.landmarks
            assert left_eye[0] < right_eye[0] and left_mouth[1] > left_eye[1]  # landmark order

    def test_close_up_face_is_found(self, real_engine):
        """Regression: a face filling the frame is missed by the 640 px pass alone."""
        from backend.engine.detector import FaceDetector

        image = _load(BULK / "Rob_Lowe_0001.jpg")
        detection = real_engine["detector"].detect(image, max_faces=1)[0]
        x1, y1, x2, y2 = (int(v) for v in detection.bbox)
        pad = int(0.15 * (x2 - x1))
        tight = image[max(y1 - pad, 0): y2 + pad, max(x1 - pad, 0): x2 + pad]
        close_up = cv2.resize(tight, (640, 640), interpolation=cv2.INTER_CUBIC)
        assert len(real_engine["detector"].detect(close_up)) == 1
        assert real_engine["detector"].input_sizes == [(640, 640), (160, 160)]
        assert isinstance(real_engine["detector"], FaceDetector)

    def test_embeddings_separate_identities(self, real_engine):
        _, _, rob1 = self._embed(real_engine, _load(BULK / "Rob_Lowe_0001.jpg"))
        _, _, rob2 = self._embed(real_engine, _load(BULK / "Rob_Lowe_0002.jpg"))
        _, _, other = self._embed(real_engine, _load(BULK / "Nathalie_Baye_0002.jpg"))
        for name in ("ada", "arc"):
            a, b, c = getattr(rob1, name), getattr(rob2, name), getattr(other, name)
            assert a.shape == (512,) and np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)
            assert float(a @ b) > 0.45, name  # same person, different photos
            assert float(a @ c) < 0.25, name  # different people
        fused_same = float(fuse_embeddings(rob1.ada, rob1.arc, 0.5) @ fuse_embeddings(rob2.ada, rob2.arc, 0.5))
        fused_diff = float(fuse_embeddings(rob1.ada, rob1.arc, 0.5) @ fuse_embeddings(other.ada, other.arc, 0.5))
        assert fused_same - fused_diff > 0.4

    def test_backbone_channel_orders_are_the_documented_ones(self, real_engine):
        """Feeding a backbone the other channel order must not look better."""
        image = _load(BULK / "Rob_Lowe_0001.jpg")
        _, crop1, right1 = self._embed(real_engine, image)
        _, crop2, right2 = self._embed(real_engine, _load(BULK / "Rob_Lowe_0002.jpg"))
        swapped1 = real_engine["embedder"].embed(np.ascontiguousarray(crop1[:, :, ::-1]))
        swapped2 = real_engine["embedder"].embed(np.ascontiguousarray(crop2[:, :, ::-1]))
        assert float(right1.arc @ right2.arc) >= float(swapped1.arc @ swapped2.arc) - 0.02
        assert float(right1.ada @ right2.ada) >= float(swapped1.ada @ swapped2.ada) - 0.02

    def test_feature_norm_tracks_image_quality(self, real_engine):
        _, crop, clean = self._embed(real_engine, _load(BULK / "Rob_Lowe_0001.jpg"))
        blurred = real_engine["embedder"].embed(cv2.GaussianBlur(crop, (0, 0), 6.0))
        dark = real_engine["embedder"].embed((crop.astype(np.float32) * 0.06).astype(np.uint8))
        assert clean.ada_norm > 20
        assert blurred.ada_norm < 0.7 * clean.ada_norm
        assert dark.ada_norm < 0.8 * clean.ada_norm

    def test_liveness_scores_real_face_above_replayed_one(self, real_engine):
        if real_engine["liveness"] is None:
            pytest.skip("liveness disabled")
        real = _load(BULK / "Rob_Lowe_0001.jpg")
        real_score = real_engine["liveness"].score(real, real_engine["detector"].detect(real, max_faces=1)[0].bbox)
        assert 0.5 < real_score <= 1.0
        fake_path = BULK / "rgb_fake.jpg"
        if fake_path.is_file():
            fake = _load(fake_path)
            fake_score = real_engine["liveness"].score(fake, real_engine["detector"].detect(fake, max_faces=1)[0].bbox)
            assert 0.0 <= fake_score < 0.5

    def test_licence_flags_reach_the_engine_info(self, real_engine):
        info = real_engine["info"]
        assert info["commercial_use_permitted"] is False
        assert info["models"]["liveness_v2"]["commercial_use"] is True
        assert info["models"]["adaface"]["commercial_use"] is False
        assert all(m["verified_sha256"] for m in info["models"].values())


class TestRealEndToEnd:
    """Acceptance path on the real networks: enroll, identify, verify, privacy, audit."""

    @pytest.fixture()
    def live(self, data_dir, monkeypatch):
        if not _real_models_available():
            pytest.skip("model weights or sample images are not available locally")
        from fastapi.testclient import TestClient

        from backend.main import create_app

        with TestClient(create_app()) as test_client:
            import time

            deadline = time.monotonic() + 120
            while time.monotonic() < deadline and not test_client.get("/api/health").json()["engine_ready"]:
                time.sleep(0.2)
            health = test_client.get("/api/health").json()
            assert health["engine_ready"], health
            yield test_client

    @staticmethod
    def _jpeg(image: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        assert ok
        return buffer.tobytes()

    def test_enroll_identify_verify_and_audit(self, live, data_dir):
        rob1, rob2 = _load(BULK / "Rob_Lowe_0001.jpg"), _load(BULK / "Rob_Lowe_0002.jpg")
        other = _load(BULK / "Nathalie_Baye_0002.jpg")
        frames = [rob1, cv2.flip(rob1, 1), cv2.convertScaleAbs(rob1, alpha=1.1, beta=8)]
        enrolled = live.post("/api/enroll", data={"subject": "Rob"}, files=[
            ("files", (f"{i}.jpg", self._jpeg(f), "image/jpeg")) for i, f in enumerate(frames)])
        assert enrolled.status_code == 200, enrolled.text
        assert enrolled.json()["template_complete"] is True

        calibration = live.get("/api/config").json()["engine"]["calibration"]
        assert calibration["mode"] == "adaptive_cohort_snorm" and calibration["equalises_subgroups"] is True
        assert live.get("/api/health").json() == {**live.get("/api/health").json(), "status": "ok", "calibrated": True}

        same = live.post("/api/identify", data={"compare_baseline": "true"},
                         files={"file": ("p.jpg", self._jpeg(rob2), "image/jpeg")}).json()["faces"][0]
        assert same["decision"]["zone"] == "MATCH_CONFIRMED" and same["decision"]["subject"] == "Rob"
        assert same["baseline"]["zone"] == "MATCH_CONFIRMED"

        stranger = live.post("/api/identify", files={"file": ("p.jpg", self._jpeg(other), "image/jpeg")}).json()
        assert stranger["faces"][0]["decision"]["zone"] in ("NO_MATCH", "UNCERTAIN_REVIEW")
        assert stranger["faces"][0]["decision"]["subject"] is None

        wrong_claim = live.post("/api/verify", data={"subject": "Rob"},
                                files={"file": ("p.jpg", self._jpeg(other), "image/jpeg")}).json()["faces"][0]
        assert wrong_claim["decision"]["subject"] is None

        # A degraded capture of the enrolled person must abstain, not match.
        degraded = (cv2.GaussianBlur(rob2, (0, 0), 5.0).astype(np.float32) * 0.2).astype(np.uint8)
        poor = live.post("/api/identify", files={"file": ("p.jpg", self._jpeg(degraded), "image/jpeg")}).json()["faces"]
        if poor:  # the detector may find nothing at all in a frame this bad
            assert poor[0]["decision"]["subject"] is None
            assert poor[0]["decision"]["zone"] in ("UNCERTAIN_REVIEW", "SPOOF_DETECTED")

        group = live.post("/api/identify", files={"file": ("g.jpg", GROUP_PHOTO.read_bytes(), "image/jpeg")}).json()
        assert len(group["faces"]) == 6 and sum(f["primary"] for f in group["faces"]) == 1
        assert all(f["decision"]["subject"] is None for f in group["faces"])

        # Privacy: nothing but ciphertext, key, lock and logs in the data directory.
        for path in data_dir.rglob("*"):
            if path.is_file():
                assert not path.read_bytes().startswith((b"\xff\xd8\xff", b"\x89PNG")), path
        blob = sqlite3.connect(data_dir / "biometrics.db").execute("SELECT embedding FROM samples").fetchone()[0]
        assert blob[:1] == b"\x01" and len(blob) == 1 + 12 + (8 + 1024 * 4) + 16

        assert live.get("/api/audit/verify").json()["ok"] is True
        prev = GENESIS_HASH
        for line in (data_dir / "audit.jsonl").read_text().splitlines():
            entry = json.loads(line)
            assert entry["prev_hash"] == prev and entry["hash"] == entry_hash(
                {k: v for k, v in entry.items() if k != "hash"})
            prev = entry["hash"]
