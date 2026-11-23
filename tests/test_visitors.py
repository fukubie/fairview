"""Visitor numbers: in-memory labels for faces that match nobody enrolled."""
from __future__ import annotations

import numpy as np

from backend.engine.visitors import VisitorRegistry


def _unit(seed: int, dim: int = 64) -> np.ndarray:
    vector = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


def _near(vector: np.ndarray, seed: int, noise: float = 0.05) -> np.ndarray:
    jitter = np.random.default_rng(seed).normal(scale=noise, size=vector.shape).astype(np.float32)
    moved = vector + jitter
    return moved / np.linalg.norm(moved)


def test_the_same_face_keeps_its_number_and_a_new_face_gets_the_next():
    registry = VisitorRegistry()
    first, second = _unit(1), _unit(2)
    assert registry.assign(first) == 1
    assert registry.assign(_near(first, 10)) == 1
    assert registry.assign(second) == 2
    # Stepping away and coming back keeps the number.
    assert registry.assign(_near(first, 11)) == 1
    assert registry.assign(_near(second, 12)) == 2
    assert len(registry) == 2


def test_a_visitor_is_forgotten_after_the_time_limit():
    registry = VisitorRegistry(ttl_s=60)
    face = _unit(3)
    assert registry.assign(face, now=0.0) == 1
    assert registry.assign(face, now=30.0) == 1
    # Not seen for longer than the limit: forgotten, so a fresh number.
    assert registry.assign(face, now=200.0) == 2
    assert len(registry) == 1


def test_the_registry_never_grows_past_its_cap():
    registry = VisitorRegistry(max_visitors=3)
    for seed in range(10):
        registry.assign(_unit(100 + seed), now=float(seed))
    assert len(registry) == 3


def test_unusable_embeddings_get_no_number_and_clear_starts_over():
    registry = VisitorRegistry()
    assert registry.assign(np.zeros(64, dtype=np.float32)) is None
    assert registry.assign(np.array([], dtype=np.float32)) is None
    assert registry.assign(_unit(4)) == 1
    registry.clear()
    assert len(registry) == 0 and registry.assign(_unit(5)) == 1


def test_match_reads_without_changing_and_forget_like_drops_a_confirmed_face():
    registry = VisitorRegistry()
    face, other = _unit(6), _unit(7)
    assert registry.match(face) is None
    assert registry.assign(face) == 1
    number, cosine = registry.match(_near(face, 20))
    assert number == 1 and cosine > 0.8
    assert registry.match(other)[1] < 0.3 and len(registry) == 1  # looking does not add anyone
    assert registry.forget_like(other) == 0 and len(registry) == 1
    assert registry.forget_like(_near(face, 21)) == 1 and len(registry) == 0


def test_known_visitor_rule_needs_a_strong_and_clearly_better_match():
    from backend.engine.visitors import is_known_visitor

    assert is_known_visitor(0.85, 0.40)
    assert not is_known_visitor(0.45, 0.10)  # not sure it is that visitor at all
    assert not is_known_visitor(0.70, 0.62)  # about as much like the enrolled person


class TestVisitorMemoryInThePipeline:
    """A stranger who resembles an employee: borderline frames stay 'no match' once a clear frame ruled them out."""

    EMPLOYEE, STRANGER_CLEAR, STRANGER_BORDERLINE = 50, 60, 61

    @staticmethod
    def _vectors():
        e1, e2, e3 = np.eye(512, dtype=np.float32)[:3]
        employee = e1
        clear = 0.2 * e1 + np.sqrt(1 - 0.04) * e2  # cosine 0.2 with the employee: no match
        borderline = 0.4 * e1 + 0.85 * e2 + np.sqrt(1 - 0.16 - 0.7225) * e3  # cosine 0.4: gray zone
        return {50: employee, 60: clear / np.linalg.norm(clear), 61: borderline / np.linalg.norm(borderline)}

    def _pipeline(self, settings):
        from conftest import FakeDetector, FakeLiveness, test_calibrator

        from backend.engine.embedder import Embedding, QualityConfig
        from backend.engine.pipeline import RecognitionPipeline
        from backend.storage.vault import BiometricVault

        vectors = self._vectors()

        class Embedder:
            def embed(self, aligned_bgr):
                identity, quality, _ = (int(v) for v in aligned_bgr[56, 56])
                vector = vectors[identity].astype(np.float32)
                return Embedding(ada=vector, arc=vector, ada_norm=quality / 10.0, arc_norm=20.0)

        from backend import config

        vault = BiometricVault(config.data_dir() / "biometrics.db")
        return RecognitionPipeline(FakeDetector(), Embedder(), FakeLiveness(), test_calibrator(), vault, settings,
                                   QualityConfig(norm_min=5.0, norm_max=25.0)), vault

    def test_borderline_frames_of_a_known_visitor_are_not_sent_for_review(self, settings):
        from conftest import make_frame

        from backend.config import ThresholdStore

        pipeline, vault = self._pipeline(settings)
        thresholds = ThresholdStore().get()
        try:
            enrolled = pipeline.enroll("Dev", [make_frame(self.EMPLOYEE)] * 3, thresholds, False)
            assert enrolled["enrolled"], enrolled

            def see(identity):
                (face,) = pipeline.process(make_frame(identity), "identify", thresholds).faces
                return face.decision.zone.value, face.decision.reason.value, face.visitor

            # Nobody has ruled the stranger out yet: a borderline frame asks for review, by design.
            assert see(self.STRANGER_BORDERLINE) == ("UNCERTAIN_REVIEW", "GRAY_ZONE_SIMILARITY", None)
            # A clear frame: not enrolled, so a visitor number.
            assert see(self.STRANGER_CLEAR) == ("NO_MATCH", "SCORE_BELOW_LOW_THRESHOLD", 1)
            # The same borderline frame now stays "no match", as the visitor already seen.
            assert see(self.STRANGER_BORDERLINE) == ("NO_MATCH", "KNOWN_VISITOR", 1)
            # The employee is still confirmed, and is never numbered as a visitor.
            assert see(self.EMPLOYEE) == ("MATCH_CONFIRMED", "SCORE_ABOVE_HIGH_THRESHOLD", None)
            assert len(pipeline.visitors) == 1
        finally:
            vault.close()
