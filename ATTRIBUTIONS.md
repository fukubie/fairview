# Attributions, provenance and licences

This document records where every algorithm, constant and model weight in
Fairview comes from, under which licence, and how it was verified. The
[README](README.md#credits-and-licences) has the short version.

> **Licence status of the default model set: non-commercial research only.**
> Three of the five networks downloaded by default (detector, ArcFace,
> AdaFace) carry research-only terms; see [Model weights](#model-weights). Only
> the two liveness networks are permissively licensed. No commercially
> licensed AdaFace or InsightFace SCRFD/ArcFace weights were found. Commercial
> deployment requires replacing the three research-only files with weights you
> hold a licence for; see
> [Using commercially licensed weights](#using-commercially-licensed-weights).

## How the reference projects were used

The projects below were read as source code during development (local
checkouts or shallow clones). **None of them is included in this repository,
and no code was copied from any of them.** Every module here is a new
implementation written against their documented behaviour. Each extracted
constant was cited to a file and, where it matters, re-verified against the
real weights (channel order, value range, output semantics, class indices).
Unknowns are listed in [Unverified](#unverified-or-unknown) rather than
filled in.

Model weights are not stored in the repository either. At run time
`backend/engine/model_manager.py` looks for existing copies (project
`models/`, `~/.insightface`, `~/.deepface`, `FAIRFACE_MODEL_SEARCH_PATH`),
reuses any file whose SHA-256 matches, and downloads only what is missing.

## Reverse-engineered components

### 1. InsightFace (`deepinsight/insightface`) — code MIT

Read from a local source checkout of the upstream repository.

| What | Source | Where it lives here |
|---|---|---|
| SCRFD preprocessing: letterbox to a top-left-anchored square, `(pixel - 127.5) / 128`, BGR→RGB | `python-package/insightface/model_zoo/scrfd.py` (`_prepare_input_blob`, `_detect_candidates`) | `backend/engine/detector.py::FaceDetector._preprocess` |
| SCRFD decoding: strides 8/16/32, 2 anchors per cell, 9 outputs (scores, box distances, key-point offsets), `centre ∓ distance × stride` | same file (`forward`, `distance2bbox`, `distance2kps`) | `decode_scrfd_outputs` |
| NMS: greedy IoU, threshold 0.4, `+1` pixel-inclusive areas; `det_thresh = 0.5` | same file (`nms`, `__init__`) | `nms`, `Settings.det_thresh` |
| Primary-face ranking `area − 2·dist²(centre)` | same file (`detect`) | `FaceDetector.detect` |
| Five ArcFace reference landmarks for a 112×112 crop | `python-package/insightface/utils/face_align.py::arcface_dst` | `ARCFACE_REFERENCE` |
| Alignment by similarity transform (`skimage.SimilarityTransform`, i.e. Umeyama) and `warpAffine(borderValue=0)` | `face_align.py::estimate_norm`, `norm_crop` | `umeyama_similarity`, `align_face` (own NumPy solver, no scikit-image) |
| ArcFace input `(pixel − 127.5) / 127.5`, RGB | `model_zoo/arcface_onnx.py` | `to_model_input(..., "RGB")` |
| Execution-provider priority with CPU fallback | `model_zoo/onnxruntime_utils.py` | `backend/config.py::select_providers` |

### 2. AdaFace (`mk-minchul/AdaFace`, commit `c60eaa7`) — code MIT

| What | Source | Where it lives here |
|---|---|---|
| Input is **BGR**, `(x/255 − 0.5)/0.5`, 112×112 NCHW | `inference.py::to_input`; README | `to_model_input(..., "BGR")`, `BackboneSpec` in `model_manager.build_engine` |
| Backbone returns `(z/‖z‖, ‖z‖)`; the norm is taken after the final `BatchNorm1d` | `net.py` (`forward`) | `model_manager.adaface_with_norm`, `embedder.FaceEmbedder` |
| Feature norm as image-quality signal, normalised and clipped | `head.py::AdaFace.forward` (`margin_scaler`) | `embedder.compute_quality` (min/max rescaling with bounds measured on data, since the head's running statistics are not part of inference checkpoints) |
| Norm-aware fusion of several features of one identity | `utils.py::fuse_features_with_norm`, `validation_mixed/validate_IJB_BC.py` | concept behind quality-weighted fusion and multi-sample templates (`embedder.fuse_embeddings`, `vault.build_template`) |

Verified on the weights: the published ONNX export ends in
`Div(z, ReduceL2(z))` and returns only the unit vector. The norm is exposed by
adding the `ReduceL2` result as a second graph output (weights untouched). The
export expects BGR; this was confirmed at weight level by comparing its first
convolution with the author's RGB-convention checkpoint (channels reversed).

### 3. DeepFace (`serengil/deepface`, commit `4df8db1`) — code MIT

| What | Source | Where it lives here |
|---|---|---|
| Two-scale anti-spoofing: MiniFASNetV2 on a 2.7× box, MiniFASNetV1SE on a 4× box, 80×80, **BGR raw 0–255 (no /255)**, 3 classes, index 1 = real | `deepface/models/spoofing/FasNetUtils.py` (`crop`, `_get_new_box`), `.../onnx/FasNet.py` | `backend/engine/liveness.py` |
| Cosine / Euclidean-L2 distance and L2 normalisation | `deepface/modules/verification.py` | `embedder.l2_normalize`; cosine throughout |
| Fixed per-model cosine thresholds as the conventional decision rule (ArcFace cosine distance 0.68 ⇒ similarity 0.32) | `deepface/config/threshold.py` | default of the naive comparison engine (`Settings.baseline_cosine_threshold`), replaced by a measured value when calibration assets are present |
| Multi-model ensembling | DeepFace no longer ships an in-library ensemble; only a boosted-trees experiment under `boosted/` | the two-backbone fusion here is this project's own design |

Deliberate difference: DeepFace reports the confidence of whichever class wins
the vote. Here `liveness_score` is the mean probability of the *real* class,
which is monotone and directly thresholdable.

### 4. CompreFace (`exadel-inc/CompreFace`, commit `ddf32da`) — Apache-2.0

| What | Source | Where it lives here |
|---|---|---|
| Subject / example resource model, `x-api-key` header, multipart `file` uploads, per-face result objects with box, landmarks and candidates | `docs/Rest-API-description.md` | `backend/main.py` (`/api/enroll`, `/api/identify`, `/api/verify`, `/api/subjects`) |
| Stored embeddings normalised at save, probes normalised at query | `EuclideanDistanceClassifier.java`, `SubjectService.java` | `vault.build_template`, `embedder.fuse_embeddings` |
| Distance → similarity in `[0,1]` through a fitted sigmoid-shaped map (`(tanh((c0 − d)·c1) + 1)/2`) | `embedding-calculator/.../plugins/base.py`, `EuclideanDistanceClassifier.java` | concept behind the logistic score map in `calibrator.py` (anchored to measured false-match rates instead of fixed coefficients) |

Deliberate difference: CompreFace leaves the match decision to the client;
here the three-zone policy is enforced and audited server-side.

### 5. Fairness evaluation references

| Project | Licence | What was taken |
|---|---|---|
| `idiap/mllm-fairness` (commit `a101eb8`) | MIT | per-group FMR/FNMR at a threshold; max/min ratio of group error rates (reported here as Skewed Error Ratio); reporting at several FMR targets (`fairness/metrics.py`) |
| `Marco210210/Synthetic-Fairness-Faces` (commit `3c82453`) | CC BY-NC-SA 4.0 per README (no LICENSE file) | metric set only: FMR/FNMR per group, Equalized Odds via fairlearn, α-weighted FMR/FNMR gap (`fairness/accuracy.py`). **No code or data from this repository is used.** Note that it evaluates attribute classification, not verification, and its grouping makes its "FDR" degenerate |
| `fairlearn` (0.16.0.dev0 source checkout) | MIT | exact semantics of `equalized_odds_difference` (between-groups, worst case); re-implemented in NumPy. The expected values in `tests/test_fairness.py` were obtained by running a local copy of fairlearn on the same worked example during analysis; fairlearn is not a dependency and is not invoked by the tests |
| `FaVFA` (source checkout) | CeCILL 2.1 (copyleft) | methodology only: hard same-demographic pairs, L2-distance→cosine conversion. No code copied. Its RFW pair scores (`data/rfw.csv`, `model_results/BUPT_RFW.csv`) are read only from a FaVFA checkout you supply (`FAIRFACE_RFW_DIR`, a git-ignored `FaVFA-main/` inside the project, or `FaVFA-main/` next to it) and are not redistributed in this repository. The Hugging Face build script copies them, with their `LICENCE`, into the Space only when present locally |
| `facerec-bias-bfw` | BSD-3-Clause | consulted and **not** used: its TAR@FAR and negative-rate code is incorrect (e.g. "FAR" computed as `1 − TAR`) |

Formulas implemented in `backend/fairness/audit.py` that come from the
literature rather than from these repositories: Fairness Discrepancy Rate
`1 − (α·ΔFMR + (1−α)·ΔFNMR)` (de Freitas Pereira & Marcel, 2021) and adaptive
symmetric score normalisation (AS-norm, from speaker verification) used for
the cohort calibration.

Departure from fairlearn worth knowing: a group with an empty denominator
yields `NaN` here and is excluded from gaps. fairlearn (via scikit-learn)
returns `0.0`, which can fabricate a disparity or hide one.

## Model weights

Downloaded on first start by `backend/engine/model_manager.py`; every registry
file is verified against the SHA-256 below before it is loaded. (A file
supplied through a `FAIRFACE_MODEL_<ROLE>` override is the operator's
responsibility and is loaded as given.)

| Role | File | Origin | SHA-256 | Licence of the weights | Commercial use |
|---|---|---|---|---|---|
| Face detector | `scrfd_500m_kps.onnx` (2.5 MB) | InsightFace `buffalo_s` / `det_500m.onnx` | `5e4447f5…b4ea3a` | InsightFace: "ALL models are available for non-commercial research purposes only" | **No** |
| ArcFace embedding | `arcface_w600k_mbf.onnx` (13.6 MB) | InsightFace `buffalo_s` / `w600k_mbf.onnx` (WebFace600K) | `9cc6e4a7…19eb4f` | same InsightFace statement | **No** |
| AdaFace embedding | `adaface_ir_18.onnx` (96 MB) | AdaFace IR-18 WebFace4M, ONNX export by `yakhyo/adaface-onnx` | `6b6a3577…0d9870` | none stated for the weights (code MIT); trained on WebFace4M, distributed for research under agreement | **No** (treat as research-only) |
| Liveness, 2.7× crop | `MiniFASNetV2.onnx` (1.7 MB) | `minivision-ai/Silent-Face-Anti-Spoofing`, ONNX export by `yakhyo/face-anti-spoofing` | `b32929ad…463907` | Apache-2.0 (training data undisclosed) | Yes |
| Liveness, 4× crop | `MiniFASNetV1SE.onnx` (1.7 MB) | same | `ebab7f90…2b3676` | Apache-2.0 (training data undisclosed) | Yes |

Full digests and download URLs are in the registry at the bottom of
`backend/engine/model_manager.py`. Hugging Face URLs are pinned to a commit.
GitHub release assets can be replaced by their owners, which is one reason the
digest is enforced for downloads and for locally discovered copies alike.

Provenance checks made while the registry was compiled (not repeated by the
test suite): the SCRFD and ArcFace files had the same SHA-256 as the
`det_500m.onnx` / `w600k_mbf.onnx` members of InsightFace's official
`buffalo_s.zip` and `buffalo_sc.zip`; the AdaFace and MiniFASNet exports were
matched to the original authors' checkpoints on fingerprint tensors (final
layers), not on every weight.

### Using commercially licensed weights

Each role can be pointed at a local file you are licensed to use:

```
FAIRFACE_MODEL_DETECTOR=/path/to/scrfd.onnx
FAIRFACE_MODEL_ARCFACE=/path/to/arcface.onnx
FAIRFACE_MODEL_ADAFACE=/path/to/adaface.onnx
```

Replacement files must keep the same interface (SCRFD with key points;
112×112 → 512-D recognisers). After changing a recogniser, rebuild the
calibration assets with `tools/build_calibration.py`: the cohort, thresholds
and quality bounds are specific to the embedding spaces they were measured in.
`GET /api/config` reports the licence of every registry model. For an
overridden role it reports "supplied by the operator" with
`commercial_use: null`, since the software cannot know what licence you hold.

Candidates examined and not adopted as defaults: YuNet (MIT) as detector, which
is not SCRFD and needs a different decoder; and AuraFace (Apache-2.0, with the
vendor's assertion that its training data permits commercial use) for the
ArcFace slot, which is a drop-in candidate for `FAIRFACE_MODEL_ARCFACE` but
measured about 20x slower on a laptop CPU (ResNet-100) and noticeably weaker on
a small sanity set than the default. Even with both, the AdaFace slot has no
permissive option, so the default set would still not be commercially clean. The ONNX Model
Zoo `arcfaceresnet100-8` is tagged Apache-2.0 yet was trained on MS-Celeb-1M
and is not safe to treat as commercial.

## Calibration data

`backend/assets/cohort.npz` and `calibration.json` are derived from
**SynthFaces** (`shredder67/synthfaces-faces-balanced`, Hugging Face revision
`732b1715`), a set of generated faces. No photograph of a real person was used,
and the cohort contains no images, only embedding vectors.

* Licence: the dataset card is tagged MIT. The accompanying paper
  (arXiv:2610.40119v1) says the data licence was still "in the process of
  development". Obtain written confirmation from the authors before relying on
  it commercially.
* Labels: seven race/ethnicity categories, which are the generator's
  conditioning labels, not per-image annotations and **not skin tone**. They
  are used offline only, to stratify splits and to anchor thresholds on the
  worst-off group. Nothing in the running system predicts or reads them.
* The generator (Arc2Face) was itself trained on real face data. The dataset
  authors claim separation from training identities; that claim was not
  verified here.

## Software dependencies

| Package | Licence | Use |
|---|---|---|
| ONNX Runtime | MIT | inference |
| ONNX | Apache-2.0 | one-time graph edit exposing the AdaFace norm |
| NumPy | BSD-3-Clause | numerics |
| OpenCV (headless) | Apache-2.0 | image decode, resize, warp |
| FastAPI / Starlette / Uvicorn | MIT / BSD-3-Clause / BSD-3-Clause | API server |
| websockets | BSD-3-Clause | WebSocket transport |
| python-multipart | Apache-2.0 | uploads |
| cryptography | Apache-2.0 / BSD-3-Clause | AES-256-GCM, Fernet, TLS certificate |
| qrcode | BSD-3-Clause | room QR codes |
| pytest, httpx (development) | MIT / BSD-3-Clause | tests |

No PyTorch, TensorFlow or diffusion model is required at run time or to build
the calibration assets.

## Unverified or unknown

* Training-data terms for WebFace4M/WebFace600K, MS-Celeb-1M and the MiniFASNet
  training set were not confirmed at an authoritative source.
* AdaFace/MiniFASNet export provenance rests on fingerprint tensors, not a
  full weight comparison.
* How CompreFace derived its per-model similarity coefficients is not
  documented upstream.
* SynthFaces label accuracy, identity consistency between images of one
  synthetic identity, and its non-leakage claim are unverified.
* Fitzpatrick-labelled face data under a usable licence was not found. The
  skin-type benchmark in the dashboard is a simulation and says so.
