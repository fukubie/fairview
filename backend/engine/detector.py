"""Stage 1: SCRFD face detection and 5-point similarity alignment.

Re-implementation of the inference path of InsightFace's SCRFD
(``python-package/insightface/model_zoo/scrfd.py``) and its ArcFace alignment
(``python-package/insightface/utils/face_align.py``) on plain NumPy + OpenCV:

* letterbox to a square canvas anchored top-left, ``(pixel - 127.5) / 128``,
  BGR -> RGB, NCHW;
* anchor-free decoding over strides 8/16/32 with 2 anchors per cell: boxes are
  ``centre -/+ distance * stride``, key points ``centre + offset * stride``;
* greedy IoU NMS (threshold 0.4, pixel-inclusive ``+1`` areas as upstream);
* two input resolutions per frame, merged by NMS. Upstream does the same with
  ``DEFAULT_DET_SIZES = [(128, 128), (640, 640)]``; here the small pass is
  160 px. SCRFD's largest anchors cannot cover a face that fills a 640 px
  frame, so close-up faces (the normal case at a webcam) are only found by the
  low-resolution pass. On 100 close-up 512 px portraits from the calibration
  set the 640 px pass alone detected 40 faces and a 160 px pass all 100; at
  128 px small faces in a group photo started to be missed, hence 160;
* alignment by the Umeyama least-squares similarity transform onto the five
  canonical ArcFace reference points of a 112x112 crop.

The Umeyama solver is implemented here directly so the runtime does not need
scikit-image (upstream calls ``skimage.transform.SimilarityTransform``).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

ALIGNED_SIZE = 112

# Canonical landmark positions in a 112x112 aligned face, in order:
# left eye, right eye, nose tip, left mouth corner, right mouth corner
# ("left" = image left). Source: insightface/utils/face_align.py::arcface_dst.
ARCFACE_REFERENCE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

SCRFD_INPUT_MEAN = 127.5
SCRFD_INPUT_STD = 128.0
SCRFD_STRIDES = (8, 16, 32)
SCRFD_ANCHORS_PER_CELL = 2


@dataclass(frozen=True)
class Detection:
    bbox: np.ndarray  # (4,) float32: x1, y1, x2, y2 in source-image pixels
    score: float  # SCRFD confidence in [0, 1]
    landmarks: np.ndarray  # (5, 2) float32 in source-image pixels

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


def umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray | None:
    """Least-squares similarity transform (rotation, uniform scale, translation).

    Returns the 2x3 matrix ``M`` minimising ``sum ||M @ [src, 1] - dst||^2``
    (Umeyama, 1991), or ``None`` when ``src`` is degenerate (all points
    coincident or non-finite) and no transform is defined.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 2 or src.shape[0] < 2:
        raise ValueError("src and dst must both be (N, 2) with N >= 2")
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        return None

    num, dim = src.shape
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_demean, dst_demean = src - src_mean, dst - dst_mean
    src_var = float(src_demean.var(axis=0).sum())
    if src_var < 1e-12:
        return None

    covariance = dst_demean.T @ src_demean / num
    d = np.ones(dim)
    if np.linalg.det(covariance) < 0:
        d[dim - 1] = -1.0  # forbid reflections

    u, s, vt = np.linalg.svd(covariance)
    rank = np.linalg.matrix_rank(covariance)
    if rank == 0:
        return None
    if rank == dim - 1:
        if np.linalg.det(u) * np.linalg.det(vt) > 0:
            rotation = u @ vt
        else:
            d_fix = d.copy()
            d_fix[dim - 1] = -1.0
            rotation = u @ np.diag(d_fix) @ vt
    else:
        rotation = u @ np.diag(d) @ vt

    scale = float(s @ d) / src_var
    if not np.isfinite(scale) or scale <= 1e-9:
        return None
    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * (rotation @ src_mean)
    return matrix


def estimate_alignment(landmarks: np.ndarray, size: int = ALIGNED_SIZE) -> np.ndarray | None:
    """2x3 matrix mapping the 5 detected landmarks onto the ArcFace template."""
    landmarks = np.asarray(landmarks, dtype=np.float64)
    if landmarks.shape != (5, 2):
        raise ValueError("landmarks must have shape (5, 2)")
    reference = ARCFACE_REFERENCE.astype(np.float64) * (size / float(ALIGNED_SIZE))
    return umeyama_similarity(landmarks, reference)


def align_face(image_bgr: np.ndarray, landmarks: np.ndarray, size: int = ALIGNED_SIZE) -> np.ndarray | None:
    """Warp a face to the canonical ``size`` x ``size`` crop (BGR uint8).

    Returns ``None`` if the landmarks are degenerate. Colour order and
    ``[-1, 1]`` scaling are applied per backbone by :func:`to_model_input`,
    because the recognition backbones disagree on channel order.
    """
    matrix = estimate_alignment(landmarks, size)
    if matrix is None:
        return None
    return cv2.warpAffine(image_bgr, matrix.astype(np.float32), (size, size), borderValue=0.0)


def to_model_input(aligned_bgr: np.ndarray, channel_order: str = "RGB") -> np.ndarray:
    """Aligned uint8 BGR crop -> ``(1, 3, H, W)`` float32 in ``[-1, 1]``."""
    if aligned_bgr.ndim != 3 or aligned_bgr.shape[2] != 3:
        raise ValueError("expected an HxWx3 image")
    if channel_order not in ("RGB", "BGR"):
        raise ValueError("channel_order must be 'RGB' or 'BGR'")
    image = aligned_bgr[:, :, ::-1] if channel_order == "RGB" else aligned_bgr
    blob = (image.astype(np.float32) - 127.5) / 127.5
    return np.ascontiguousarray(blob.transpose(2, 0, 1)[None])


@dataclass(frozen=True)
class PoseEstimate:
    yaw_offset: float  # 0 frontal .. ~1 full profile (nose slides onto an eye)
    pitch_offset: float  # 0 neutral .. ~1 extreme up/down tilt
    roll_degrees: float  # in-plane rotation; removed by alignment, informational
    score: float  # frontal-ness in [0, 1]


# Vertical position of the nose between the eye line and the mouth line in the
# canonical template: (71.74 - 51.60) / (92.28 - 51.60).
_NEUTRAL_NOSE_RATIO = 0.495
# Offsets at which the pose score reaches zero. A horizontal nose offset of
# 0.35 inter-ocular distances corresponds to roughly 45-50 degrees of yaw.
# Pitch is given a much wider tolerance: the vertical ratio also moves with
# expression (a smile lifts the mouth corners) and with individual facial
# proportions, so a tight bound would penalise anatomy rather than pose.
_YAW_ZERO = 0.35
_PITCH_ZERO = 0.50


def estimate_pose(landmarks: np.ndarray) -> PoseEstimate:
    """Coarse yaw/pitch symmetry from the 5 landmarks (no extra model).

    Yaw: on a frontal face the nose tip projects onto the midpoint of the eye
    segment; turning the head slides it toward one eye. Pitch: nodding changes
    the ratio between the eye-line-to-nose and nose-to-mouth-line distances.
    Both are measured in a frame aligned to the eye line, so in-plane roll
    (which alignment removes anyway) does not count against the score.
    """
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.shape != (5, 2) or not np.all(np.isfinite(pts)):
        return PoseEstimate(1.0, 1.0, 0.0, 0.0)
    left_eye, right_eye, nose, left_mouth, right_mouth = pts
    eye_vec = right_eye - left_eye
    eye_dist = float(np.linalg.norm(eye_vec))
    if eye_dist < 1e-6:
        return PoseEstimate(1.0, 1.0, 0.0, 0.0)
    axis_x = eye_vec / eye_dist
    axis_y = np.array([-axis_x[1], axis_x[0]])  # points down the face
    eye_mid = (left_eye + right_eye) / 2.0
    mouth_mid = (left_mouth + right_mouth) / 2.0

    nose_along = float((nose - left_eye) @ axis_x) / eye_dist  # 0.5 when frontal
    yaw_offset = min(abs(nose_along - 0.5) / _YAW_ZERO, 1.0)

    eye_to_nose = float((nose - eye_mid) @ axis_y)
    eye_to_mouth = float((mouth_mid - eye_mid) @ axis_y)
    if eye_to_mouth <= 1e-6:
        pitch_offset = 1.0  # mouth above the eyes: upside-down or garbage
    else:
        pitch_offset = min(abs(eye_to_nose / eye_to_mouth - _NEUTRAL_NOSE_RATIO) / _PITCH_ZERO, 1.0)

    roll = float(np.degrees(np.arctan2(eye_vec[1], eye_vec[0])))
    # Quadratic fall-off: small deviations from frontal are harmless to the
    # recognisers and cost almost nothing; the score collapses near profile.
    score = float((1.0 - yaw_offset**2) * (1.0 - pitch_offset**2))
    return PoseEstimate(float(yaw_offset), float(pitch_offset), roll, score)


# --------------------------------------------------------------------------- #
# SCRFD
# --------------------------------------------------------------------------- #


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Greedy non-maximum suppression; returns kept indices, best first."""
    if boxes.shape[0] == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return keep


def decode_scrfd_outputs(
    outputs: list[np.ndarray],
    input_height: int,
    input_width: int,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw SCRFD heads into boxes, scores and key points (input pixels).

    ``outputs`` is the 9-tensor layout of the ``*_kps`` models: three score
    maps, three box-distance maps, three key-point-offset maps, one per stride.
    Tensors may or may not carry a leading batch dimension.
    """
    levels = len(SCRFD_STRIDES)
    if len(outputs) != levels * 3:
        raise ValueError(
            f"expected a SCRFD model with key points ({levels * 3} outputs), got {len(outputs)} outputs"
        )
    all_boxes, all_scores, all_kps = [], [], []
    for index, stride in enumerate(SCRFD_STRIDES):
        scores = np.asarray(outputs[index]).reshape(-1)
        distances = np.asarray(outputs[index + levels]).reshape(-1, 4) * stride
        offsets = np.asarray(outputs[index + 2 * levels]).reshape(-1, 10) * stride

        height, width = input_height // stride, input_width // stride
        expected = height * width * SCRFD_ANCHORS_PER_CELL
        if scores.shape[0] != expected:
            raise ValueError(
                f"SCRFD stride {stride}: expected {expected} anchors, model produced {scores.shape[0]}"
            )
        grid = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
        centres = (grid * stride).reshape(-1, 2)
        centres = np.repeat(centres, SCRFD_ANCHORS_PER_CELL, axis=0)

        keep = np.where(scores >= score_threshold)[0]
        if keep.size == 0:
            continue
        c, d, o = centres[keep], distances[keep], offsets[keep]
        boxes = np.stack([c[:, 0] - d[:, 0], c[:, 1] - d[:, 1], c[:, 0] + d[:, 2], c[:, 1] + d[:, 3]], axis=-1)
        kps = o.reshape(-1, 5, 2) + c[:, None, :]
        all_boxes.append(boxes)
        all_scores.append(scores[keep])
        all_kps.append(kps)

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 5, 2), dtype=np.float32),
        )
    return (
        np.concatenate(all_boxes).astype(np.float32),
        np.concatenate(all_scores).astype(np.float32),
        np.concatenate(all_kps).astype(np.float32),
    )


class FaceDetector:
    """SCRFD detector returning boxes, confidences and 5 landmarks per face."""

    def __init__(
        self,
        session,
        det_thresh: float = 0.5,
        nms_thresh: float = 0.4,
        det_size: int = 640,
        min_face_size: float = 20.0,
        det_size_small: int = 160,
    ) -> None:
        self.session = session
        self.det_thresh = float(det_thresh)
        self.nms_thresh = float(nms_thresh)
        self.min_face_size = float(min_face_size)
        model_input = session.get_inputs()[0]
        self.input_name = model_input.name
        self.output_names = [o.name for o in session.get_outputs()]
        # Honour a fixed-resolution export; otherwise use the configured size.
        shape = model_input.shape
        fixed_h, fixed_w = (shape[2], shape[3]) if len(shape) == 4 else (None, None)
        if isinstance(fixed_h, int) and isinstance(fixed_w, int) and fixed_h > 0 and fixed_w > 0:
            # A fixed-resolution export supports exactly one size.
            self.input_sizes = [(int(fixed_h), int(fixed_w))]
        else:
            sizes = [int(det_size)]
            if det_size_small and int(det_size_small) < int(det_size):
                sizes.append(int(det_size_small))
            self.input_sizes = [(size, size) for size in sizes]
        if any(h % 32 or w % 32 for h, w in self.input_sizes):
            raise ValueError("SCRFD input sizes must be multiples of 32")

    @staticmethod
    def _preprocess(image_bgr: np.ndarray, input_height: int, input_width: int) -> tuple[np.ndarray, float]:
        height, width = image_bgr.shape[:2]
        image_ratio = height / width
        model_ratio = input_height / input_width
        if image_ratio > model_ratio:
            new_height = input_height
            new_width = max(1, int(new_height / image_ratio))
        else:
            new_width = input_width
            new_height = max(1, int(new_width * image_ratio))
        scale = new_height / height
        resized = cv2.resize(image_bgr, (new_width, new_height))
        canvas = np.zeros((input_height, input_width, 3), dtype=np.uint8)
        canvas[:new_height, :new_width] = resized
        blob = (canvas[:, :, ::-1].astype(np.float32) - SCRFD_INPUT_MEAN) / SCRFD_INPUT_STD
        return np.ascontiguousarray(blob.transpose(2, 0, 1)[None]), scale

    def detect(self, image_bgr: np.ndarray, max_faces: int = 0) -> list[Detection]:
        """Detect faces. Results are ordered primary-subject first.

        "Primary" follows InsightFace's default ranking: large and centred
        (``area - 2 * squared_distance_to_image_centre``).
        """
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("expected an HxWx3 BGR image")
        height, width = image_bgr.shape[:2]
        if height < 8 or width < 8:
            return []
        box_parts, score_parts, kps_parts = [], [], []
        for input_height, input_width in self.input_sizes:
            blob, scale = self._preprocess(image_bgr, input_height, input_width)
            outputs = self.session.run(self.output_names, {self.input_name: blob})
            part_boxes, part_scores, part_kps = decode_scrfd_outputs(
                outputs, input_height, input_width, self.det_thresh
            )
            if part_boxes.shape[0]:
                box_parts.append(part_boxes / scale)
                score_parts.append(part_scores)
                kps_parts.append(part_kps / scale)
        if not box_parts:
            return []
        boxes, scores, kps = np.concatenate(box_parts), np.concatenate(score_parts), np.concatenate(kps_parts)
        # One NMS over both resolutions: the same face seen twice collapses to
        # its higher-confidence detection.
        keep = nms(boxes, scores, self.nms_thresh)
        boxes, scores, kps = boxes[keep], scores[keep], kps[keep]

        # Drop slivers and faces too small to carry identity information.
        sizes = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        valid = (sizes >= self.min_face_size) & np.all(np.isfinite(boxes), axis=1)
        boxes, scores, kps = boxes[valid], scores[valid], kps[valid]
        if boxes.shape[0] == 0:
            return []

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        centre_x = (boxes[:, 0] + boxes[:, 2]) / 2 - width / 2
        centre_y = (boxes[:, 1] + boxes[:, 3]) / 2 - height / 2
        rank = areas - 2.0 * (centre_x**2 + centre_y**2)
        order = np.argsort(-rank, kind="stable")
        if max_faces > 0:
            order = order[:max_faces]
        return [
            Detection(
                bbox=np.clip(boxes[i], [0, 0, 0, 0], [width, height, width, height]).astype(np.float32),
                score=float(np.clip(scores[i], 0.0, 1.0)),
                landmarks=kps[i].astype(np.float32),
            )
            for i in order
        ]
