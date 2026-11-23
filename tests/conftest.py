"""Shared fixtures.

The API and pipeline tests run the *real* pipeline, decision engine, vault and
audit log. Only the three neural stages are replaced by deterministic stand-ins
so the suite needs no model weights and stays fast:

* a test image is a flat-colour PNG whose colour encodes what the stand-ins
  should "see": blue = identity id (0 means no face), green = capture quality,
  red = liveness;
* identities sharing ``id // 10`` are lookalikes (highly similar embeddings).

Tests that exercise the actual ONNX networks live in ``TestRealModels``
(``test_pipeline.py``) and are skipped when the weights or the sample images
are not available locally.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import Settings, load_settings  # noqa: E402
from backend.engine.calibrator import CalibrationParams, ScoreCalibrator  # noqa: E402
from backend.engine.detector import ARCFACE_REFERENCE, Detection  # noqa: E402
from backend.engine.embedder import Embedding, QualityConfig  # noqa: E402
from backend.engine.pipeline import RecognitionPipeline  # noqa: E402

FRAME_SIZE = 224
LOOKALIKE_BLEND = 0.25  # weight of the individual part within a lookalike family


def make_frame(identity: int, quality: int = 230, liveness: int = 255) -> np.ndarray:
    """Flat BGR frame encoding (identity, quality, liveness) in its colour."""
    frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    frame[:] = (identity, quality, liveness)
    return frame


def encode_png(frame: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", frame)
    assert ok
    return buffer.tobytes()


def frame_bytes(identity: int, quality: int = 230, liveness: int = 255) -> bytes:
    return encode_png(make_frame(identity, quality, liveness))


def identity_vector(identity: int, space: int) -> np.ndarray:
    """Unit vector for an identity; ids in the same decade are near-duplicates."""
    family = np.random.default_rng([space, 1, identity // 10]).standard_normal(512)
    unique = np.random.default_rng([space, 2, identity]).standard_normal(512)
    family /= np.linalg.norm(family)
    unique /= np.linalg.norm(unique)
    vector = family + LOOKALIKE_BLEND * unique
    return vector / np.linalg.norm(vector)


class FakeDetector:
    """One centred face per frame unless the identity channel is 0."""

    def detect(self, image_bgr: np.ndarray, max_faces: int = 0):
        height, width = image_bgr.shape[:2]
        if int(image_bgr[height // 2, width // 2, 0]) == 0:
            return []
        scale = min(height, width) / 112.0
        landmarks = (ARCFACE_REFERENCE * scale).astype(np.float32)
        bbox = np.array([0.15 * width, 0.1 * height, 0.85 * width, 0.9 * height], dtype=np.float32)
        return [Detection(bbox=bbox, score=0.95, landmarks=landmarks)]


class FakeEmbedder:
    def embed(self, aligned_bgr: np.ndarray) -> Embedding:
        identity, quality, _ = (int(v) for v in aligned_bgr[56, 56])
        # Feature norm tracks the quality channel: 0..255 -> 0..25.5.
        return Embedding(
            ada=identity_vector(identity, 0).astype(np.float32),
            arc=identity_vector(identity, 1).astype(np.float32),
            ada_norm=quality / 10.0,
            arc_norm=20.0,
        )


class FakeLiveness:
    def score(self, image_bgr: np.ndarray, bbox) -> float:
        height, width = image_bgr.shape[:2]
        return float(image_bgr[height // 2, width // 2, 2]) / 255.0


def test_calibrator() -> ScoreCalibrator:
    # Global z-score on cosine (mu 0, sigma 0.1): s = 0.48 at cos 0.30 and
    # s = 0.78 at cos 0.50.
    a, b = ScoreCalibrator.solve_logistic(3.0, 5.0)
    return ScoreCalibrator(CalibrationParams(a=a, b=b, mu0=0.0, sigma0=0.1, source="test"))


test_calibrator.__test__ = False  # helper, not a test


def fake_pipeline_factory(vault, settings: Settings):
    pipeline = RecognitionPipeline(
        FakeDetector(), FakeEmbedder(), FakeLiveness(), test_calibrator(), vault, settings,
        QualityConfig(norm_min=5.0, norm_max=25.0),
    )
    return pipeline, {"models": {"stand_in": True}, "providers": ["CPUExecutionProvider"],
                      "calibration": pipeline.calibrator.describe()}


@pytest.fixture()
def data_dir(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "data"
    monkeypatch.setenv("FAIRFACE_DATA_DIR", str(path))
    monkeypatch.setenv("FAIRFACE_ALLOWED_HOSTS", "testserver")  # the test client's Host header
    for name in ("BIOMETRIC_ENCRYPTION_KEY", "BIOMETRIC_CIPHER", "FAIRFACE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return path


@pytest.fixture()
def settings(data_dir) -> Settings:
    return load_settings()


@pytest.fixture()
def client(data_dir):
    from fastapi.testclient import TestClient

    from backend.main import create_app

    with TestClient(create_app(fake_pipeline_factory)) as test_client:
        # Model loading runs on the inference executor; wait for it.
        for _ in range(200):
            if test_client.get("/api/health").json()["engine_ready"]:
                break
            import time

            time.sleep(0.01)
        yield test_client


def pytest_configure(config):
    os.environ.setdefault("FAIRFACE_LOG_LEVEL", "WARNING")
