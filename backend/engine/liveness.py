"""Stage 2: passive liveness (presentation-attack detection) with MiniFASNet.

Re-implements the inference path DeepFace uses for anti-spoofing
(``deepface/models/spoofing``), which itself wraps Minivision's
Silent-Face-Anti-Spoofing models:

* two networks see the face at two context scales: ``MiniFASNetV2`` on a crop
  2.7x the detected box and ``MiniFASNetV1SE`` on a crop 4x the box;
* each crop is resized to 80x80 and fed as **BGR, raw 0..255 floats** (the
  reference code deliberately skips the ``/255`` normalisation);
* each network yields 3 class probabilities; class index 1 is "real".

DeepFace reports the confidence of whichever class wins the summed vote.
Here ``liveness_score`` is defined as the mean probability of the *real* class,
``(p1[1] + p2[1]) / 2``, a monotone score in ``[0, 1]`` that can be thresholded
by ``l_min`` directly.

Passive single-frame liveness is a deterrent, not a guarantee: these small
networks were trained on specific print/replay attack sets and generalise
imperfectly to unseen cameras, screens and masks. Treat a pass as "no attack
detected", never as proof of presence.
"""

from __future__ import annotations

import cv2
import numpy as np

INPUT_SIZE = 80
REAL_CLASS_INDEX = 1
SCALE_V2 = 2.7
SCALE_V1SE = 4.0


def expanded_box(src_w: int, src_h: int, bbox_xywh: tuple[float, float, float, float], scale: float):
    """Square-ish context window around a face box, shifted to stay in frame.

    Port of ``_get_new_box`` from Silent-Face-Anti-Spoofing / DeepFace. The
    scale is reduced when the image is too small to hold the requested context.
    Returns inclusive integer corners ``(left, top, right, bottom)``.
    """
    x, y, box_w, box_h = (float(v) for v in bbox_xywh)
    if box_w <= 0 or box_h <= 0:
        raise ValueError("face box must have positive width and height")
    scale = min((src_h - 1) / box_h, (src_w - 1) / box_w, scale)
    new_w, new_h = box_w * scale, box_h * scale
    centre_x, centre_y = box_w / 2 + x, box_h / 2 + y
    left, top = centre_x - new_w / 2, centre_y - new_h / 2
    right, bottom = centre_x + new_w / 2, centre_y + new_h / 2
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > src_w - 1:
        left -= right - src_w + 1
        right = src_w - 1
    if bottom > src_h - 1:
        top -= bottom - src_h + 1
        bottom = src_h - 1
    return int(left), int(top), int(right), int(bottom)


def crop_for_liveness(image_bgr: np.ndarray, bbox_xyxy, scale: float) -> np.ndarray | None:
    """Context crop resized to 80x80 (BGR uint8), or ``None`` if degenerate."""
    src_h, src_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox_xyxy)
    if x2 - x1 < 1 or y2 - y1 < 1 or src_h < 2 or src_w < 2:
        return None
    left, top, right, bottom = expanded_box(src_w, src_h, (x1, y1, x2 - x1, y2 - y1), scale)
    left, top = max(left, 0), max(top, 0)
    patch = image_bgr[top : bottom + 1, left : right + 1]
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return None
    return cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exp = np.exp(shifted)
    return exp / exp.sum()


class LivenessDetector:
    """Two-scale MiniFASNet ensemble returning P(real)."""

    def __init__(self, v2_session, v1se_session, layout: str = "NHWC", outputs_probabilities: bool = True) -> None:
        if layout not in ("NHWC", "NCHW"):
            raise ValueError("layout must be 'NHWC' or 'NCHW'")
        self._models = ((v2_session, SCALE_V2), (v1se_session, SCALE_V1SE))
        self._input_names = [s.get_inputs()[0].name for s, _ in self._models]
        self.layout = layout
        self.outputs_probabilities = outputs_probabilities

    def _probabilities(self, index: int, crop_bgr: np.ndarray) -> np.ndarray:
        session, _ = self._models[index]
        blob = crop_bgr.astype(np.float32)  # raw 0..255, BGR, as in the reference
        if self.layout == "NCHW":
            blob = blob.transpose(2, 0, 1)
        raw = np.asarray(
            session.run(None, {self._input_names[index]: np.ascontiguousarray(blob[None])})[0],
            dtype=np.float64,
        ).reshape(-1)
        if raw.size != 3 or not np.all(np.isfinite(raw)):
            raise ValueError("liveness model returned an unexpected output")
        return raw if self.outputs_probabilities else _softmax(raw)

    def score(self, image_bgr: np.ndarray, bbox_xyxy) -> float | None:
        """Liveness score in ``[0, 1]`` for one face, ``None`` if it cannot be assessed."""
        total = 0.0
        for index, (_, scale) in enumerate(self._models):
            crop = crop_for_liveness(image_bgr, bbox_xyxy, scale)
            if crop is None:
                return None
            total += float(self._probabilities(index, crop)[REAL_CLASS_INDEX])
        return float(min(max(total / len(self._models), 0.0), 1.0))
