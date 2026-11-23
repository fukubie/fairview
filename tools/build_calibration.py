#!/usr/bin/env python3
"""Build the calibration assets from a labelled face-image set (offline tool).

    python tools/build_calibration.py --images DIR --out backend/assets

``DIR`` holds one sub-directory per demographic group, one image per person:
``DIR/<group>/<id>.jpg``. ``--pairs DIR2`` optionally adds identities with
several images each (``DIR2/<group>/<identity>/<n>.jpg``); their first image
joins the identity pool and the others serve as genuine probes. Group labels are used here, offline, to split the
data evenly and to anchor thresholds on the worst-off group. They are not
written into the cohort and nothing at run time predicts or reads them.

Produces
* ``cohort.npz``        reference embeddings for adaptive cohort normalisation;
* ``calibration.json``  logistic anchors, quality-norm bounds, baseline threshold;
* ``calibration_report.json``  measured per-group false-match rates on a
  held-out split, with and without cohort normalisation.

Identities are split three ways, disjointly: cohort / calibration / evaluation.
With ``--cohort-clusters K`` the cohort is replaced by K centroids, each the
mean of several people, so that no individual's embedding is distributed.

Image features are cached in ``--cache`` so the (slow) network pass runs once.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.engine.calibrator import CalibrationParams, ScoreCalibrator  # noqa: E402
from backend.engine.detector import align_face, estimate_pose  # noqa: E402
from backend.engine.embedder import QualityConfig, compute_quality, fuse_embeddings, fusion_weight  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def degrade(crop: np.ndarray) -> np.ndarray:
    """Heavy blur plus under-exposure: the reference for 'unusable' quality."""
    return (cv2.GaussianBlur(crop, (0, 0), 6.0).astype(np.float32) * 0.25).astype(np.uint8)


def extract(images_dir: Path, cache: Path, degrade_every: int = 4, pattern: str = "*/*") -> dict[str, np.ndarray]:
    if cache.is_file():
        with np.load(cache, allow_pickle=False) as archive:
            print(f"loaded cached features for {archive['ada'].shape[0]} faces from {cache}")
            return {k: archive[k] for k in archive.files}
    from backend.engine.model_manager import build_engine

    detector, embedder, _, _ = build_engine(config.load_settings())
    files = sorted(p for p in images_dir.glob(pattern) if p.suffix.lower() in IMAGE_SUFFIXES)
    group_of = (lambda p: p.parent.name) if pattern == "*/*" else (lambda p: p.parent.parent.name)
    name_of = (lambda p: p.stem) if pattern == "*/*" else (lambda p: f"{p.parent.name}/{p.stem}")
    rows: dict[str, list] = {k: [] for k in (
        "ada", "arc", "ada_norm", "det", "pose", "yaw", "pitch", "group", "name", "degraded_norm")}
    started = time.time()
    for index, path in enumerate(files):
        image = cv2.imread(str(path))
        if image is None:
            continue
        detections = detector.detect(image, max_faces=1)
        if not detections:
            continue
        det = detections[0]
        crop = align_face(image, det.landmarks)
        if crop is None:
            continue
        try:
            emb = embedder.embed(crop)
        except ValueError:
            continue
        pose = estimate_pose(det.landmarks)
        degraded_norm = np.nan
        if index % degrade_every == 0:
            try:
                degraded_norm = embedder.embed(degrade(crop)).ada_norm
            except ValueError:
                pass
        for key, value in (
            ("ada", emb.ada), ("arc", emb.arc), ("ada_norm", emb.ada_norm), ("det", det.score),
            ("pose", pose.score), ("yaw", pose.yaw_offset), ("pitch", pose.pitch_offset),
            ("group", group_of(path)), ("name", name_of(path)), ("degraded_norm", degraded_norm),
        ):
            rows[key].append(value)
        if (index + 1) % 200 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{len(files)} images, {rate:.1f}/s", flush=True)
    data = {
        "ada": np.stack(rows["ada"]).astype(np.float32),
        "arc": np.stack(rows["arc"]).astype(np.float32),
        **{k: np.asarray(rows[k], dtype=np.float32) for k in ("ada_norm", "det", "pose", "yaw", "pitch", "degraded_norm")},
        "group": np.asarray(rows["group"]),
        "name": np.asarray(rows["name"]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **data)
    print(f"extracted {data['ada'].shape[0]} faces from {len(files)} images in {time.time() - started:.0f}s")
    return data


def kmeans_centroids(vectors: np.ndarray, k: int, seed: int = 0, iterations: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Spherical k-means. Returns ``(unit centroids, members per centroid)``."""
    rng = np.random.default_rng(seed)
    centroids = vectors[rng.choice(len(vectors), size=k, replace=False)].copy()
    assignment = np.zeros(len(vectors), dtype=np.int64)
    for _ in range(iterations):
        assignment = np.argmax(vectors @ centroids.T, axis=1)
        for j in range(k):
            members = vectors[assignment == j]
            if len(members):
                mean = members.mean(axis=0)
                centroids[j] = mean / np.linalg.norm(mean)
            else:  # re-seed an empty cluster
                centroids[j] = vectors[rng.integers(len(vectors))]
    counts = np.bincount(assignment, minlength=k)
    return centroids.astype(np.float32), counts


def pair_scores(calibrator: ScoreCalibrator, vectors: np.ndarray):
    """All unordered impostor pairs within a set: ``(z, i_index, j_index)``."""
    stats = calibrator.cohort_stats(vectors)
    cosine = np.clip(vectors @ vectors.T, -1, 1)
    z = calibrator.normalise(cosine, stats, stats)
    i, j = np.triu_indices(len(vectors), k=1)
    return z[i, j], cosine[i, j], i, j


def group_fmr(scores: np.ndarray, threshold: float, pair_group: np.ndarray, groups: list[str]) -> dict:
    out = {}
    for g in groups:
        member = scores[pair_group == g]
        out[g] = {"pairs": int(member.size), "false_matches": int((member >= threshold).sum()),
                  "fmr": float((member >= threshold).mean()) if member.size else None}
    return out


def gap_summary(per_group: dict) -> dict:
    rates = [v["fmr"] for v in per_group.values() if v["fmr"] is not None]
    low, high = min(rates), max(rates)
    return {"max_fmr": high, "min_fmr": low, "gap": high - low,
            "ser": (high / low) if low > 0 else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=config.ASSETS_DIR)
    parser.add_argument("--cache", type=Path, default=None, help="feature cache (.npz)")
    parser.add_argument("--pairs", type=Path, default=None, help="multi-image identities: DIR/<group>/<identity>/<n>.jpg")
    parser.add_argument("--pairs-cache", type=Path, default=None)
    parser.add_argument("--dataset-name", default="unspecified")
    parser.add_argument("--dataset-license", default="unspecified")
    parser.add_argument("--cohort-fraction", type=float, default=0.3)
    parser.add_argument("--calibration-fraction", type=float, default=0.4)
    parser.add_argument("--cohort-clusters", type=int, default=0,
                        help="replace the cohort by this many k-means centroids (0 = keep individual embeddings)")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--fmr-low", type=float, default=1e-2)
    parser.add_argument("--fmr-high", type=float, default=1e-4)
    parser.add_argument("--baseline-fmr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Feature caches hold one embedding per person. Keep them out of the
    # package directory so they can never be shipped by accident.
    cache_dir = Path(tempfile.gettempdir()) / "fairface-guard-calibration"
    cache_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def dataset_key(directory: Path, pattern: str) -> str:
        """Fingerprint of a dataset (location plus file list) for the cache name."""
        import hashlib

        digest = hashlib.sha256(str(directory.resolve()).encode())
        for path in sorted(directory.glob(pattern)):
            digest.update(f"{path.relative_to(directory)}:{path.stat().st_size}".encode())
        return digest.hexdigest()[:16]

    cache = args.cache or (cache_dir / f"features_cache_{dataset_key(args.images, '*/*')}.npz")
    data = extract(args.images, cache)
    probes = None
    if args.pairs is not None:
        extra = extract(args.pairs, args.pairs_cache or (cache_dir / f"pairs_cache_{dataset_key(args.pairs, '*/*/*')}.npz"), pattern="*/*/*")
        identity = np.array([name.rsplit("/", 1)[0] for name in extra["name"]])
        image_no = np.array([name.rsplit("/", 1)[1] for name in extra["name"]])
        first = image_no == "0"
        # First images become ordinary pool members (named by identity) ...
        pool = {k: v[first] for k, v in extra.items()}
        pool["name"] = np.array([f"pair:{i}" for i in identity[first]])
        data = {k: np.concatenate([data[k], pool[k]]) for k in data}
        # ... the remaining images are genuine probes for those identities.
        probes = {k: v[~first] for k, v in extra.items()}
        probes["identity"] = np.array([f"pair:{i}" for i in identity[~first]])
    groups = sorted(set(data["group"].tolist()))
    n = data["ada"].shape[0]
    print(f"{n} faces, groups: " + ", ".join(f"{g}={int((data['group'] == g).sum())}" for g in groups))

    # -- quality bounds from data ------------------------------------------ #
    clean_norm = data["ada_norm"]
    degraded = data["degraded_norm"][np.isfinite(data["degraded_norm"])]
    norm_min = float(np.median(degraded))
    norm_max = float(np.median(clean_norm))
    quality_config = QualityConfig(norm_min=round(norm_min, 2), norm_max=round(norm_max, 2))
    quality = np.array([
        compute_quality(nm, d, p, quality_config).score
        for nm, d, p in zip(data["ada_norm"], data["det"], data["pose"])
    ])
    quality_by_group = {
        g: {
            "feature_norm_median": float(np.median(clean_norm[data["group"] == g])),
            "detector_score_median": float(np.median(data["det"][data["group"] == g])),
            "pose_score_median": float(np.median(data["pose"][data["group"] == g])),
            "quality_median": float(np.median(quality[data["group"] == g])),
            "below_q_min_0.40": float((quality[data["group"] == g] < 0.40).mean()),
        }
        for g in groups
    }
    print(f"feature norm: clean median {norm_max:.2f}, degraded median {norm_min:.2f}")

    fused = fuse_embeddings(data["ada"], data["arc"], fusion_weight(quality))

    # -- three-way disjoint split, stratified by group --------------------- #
    rng = np.random.default_rng(args.seed)
    part = np.empty(n, dtype="<U12")
    for g in groups:
        idx = rng.permutation(np.where(data["group"] == g)[0])
        a = int(len(idx) * args.cohort_fraction)
        b = a + int(len(idx) * args.calibration_fraction)
        part[idx[:a]], part[idx[a:b]], part[idx[b:]] = "cohort", "calibration", "evaluation"
    cohort_individuals = fused[part == "cohort"]
    cohort_note = {"kind": "individual embeddings", "size": int(len(cohort_individuals))}
    cohort = cohort_individuals
    if args.cohort_clusters > 0:
        cohort, counts = kmeans_centroids(cohort_individuals, args.cohort_clusters, args.seed)
        cohort_note = {"kind": "k-means centroids", "size": int(len(cohort)),
                       "source_individuals": int(len(cohort_individuals)),
                       "members_per_centroid_min": int(counts.min()),
                       "members_per_centroid_median": float(np.median(counts))}
    top_k = min(args.top_k, max(2, len(cohort) // 4))

    def same_group_labels(index_set: np.ndarray, i: np.ndarray, j: np.ndarray) -> np.ndarray:
        g = data["group"][index_set]
        return np.where(g[i] == g[j], g[i], "cross-group")

    # -- fit anchors on the calibration split ------------------------------ #
    normaliser = ScoreCalibrator(CalibrationParams(a=1.0, b=0.0, mu0=0.0, sigma0=1.0, top_k=top_k), cohort)
    cal_idx = np.where(part == "calibration")[0]
    cal_z, cal_cos, ci, cj = pair_scores(normaliser, fused[cal_idx])
    cal_groups = same_group_labels(cal_idx, ci, cj)
    # Thresholds are anchored on same-group pairs only. Cross-group pairs are
    # much easier impostors; pooling them in would dilute the tail and let
    # every individual group overshoot its target.
    hard = cal_groups != "cross-group"
    cal_z, cal_cos, cal_groups, ci, cj = cal_z[hard], cal_cos[hard], cal_groups[hard], ci[hard], cj[hard]
    mu0, sigma0 = float(cal_cos.mean()), float(cal_cos.std())
    fit_kwargs = dict(mu0=mu0, sigma0=sigma0, top_k=top_k, fmr_low=args.fmr_low, fmr_high=args.fmr_high)
    calibrator = ScoreCalibrator.fit(
        cohort, cal_z, cal_groups, source=f"{args.dataset_name}: cohort AS-norm, worst-group anchors", **fit_kwargs)
    global_cal = ScoreCalibrator.fit(
        None, (cal_cos - mu0) / sigma0, None, source=f"{args.dataset_name}: global z-score", **fit_kwargs)
    arc_cal = data["arc"][cal_idx]
    arc_cos_cal = np.clip(arc_cal @ arc_cal.T, -1, 1)[ci, cj]
    baseline_threshold = float(np.quantile(arc_cos_cal, 1.0 - args.baseline_fmr))

    # -- measure on the held-out evaluation split -------------------------- #
    ev_idx = np.where(part == "evaluation")[0]
    ev_z, ev_cos, ei, ej = pair_scores(calibrator, fused[ev_idx])
    ev_groups = same_group_labels(ev_idx, ei, ej)
    arc_ev = data["arc"][ev_idx]
    arc_cos_ev = np.clip(arc_ev @ arc_ev.T, -1, 1)[ei, ej]
    s_cohort = calibrator.to_score(ev_z)
    s_global = global_cal.to_score((ev_cos - mu0) / sigma0)
    report_groups = groups
    evaluation = {}
    for label, anchor in (("tau_low", calibrator.params.anchor_low), ("tau_high", calibrator.params.anchor_high)):
        with_cohort = group_fmr(s_cohort, anchor, ev_groups, report_groups)
        without = group_fmr(s_global, anchor, ev_groups, report_groups)
        evaluation[label] = {
            "score_threshold": anchor,
            "target_fmr": args.fmr_low if label == "tau_low" else args.fmr_high,
            "with_cohort": {"groups": with_cohort, **gap_summary(with_cohort)},
            "global_zscore": {"groups": without, **gap_summary(without)},
        }
    baseline_eval = group_fmr(arc_cos_ev, baseline_threshold, ev_groups, report_groups)
    evaluation["baseline_arcface"] = {
        "cosine_threshold": baseline_threshold, "target_fmr": args.baseline_fmr,
        "groups": baseline_eval, **gap_summary(baseline_eval),
    }
    impostor_means = {g: float(arc_cos_ev[ev_groups == g].mean()) for g in report_groups}
    # Intermediate operating point with enough events per group to be read.
    def safe_quantile(values: np.ndarray, fmr: float) -> float | None:
        try:
            return ScoreCalibrator.impostor_quantile(values, fmr)
        except ValueError:  # group too small to support this tail
            return None

    z_mid = max(
        q for q in [safe_quantile(cal_z, 1e-3)] + [safe_quantile(cal_z[cal_groups == g], 1e-3) for g in groups]
        if q is not None
    )
    cos_mid = ScoreCalibrator.impostor_quantile((cal_cos - mu0) / sigma0, 1e-3)
    mid_cohort = group_fmr(ev_z, z_mid, ev_groups, report_groups)
    mid_global = group_fmr((ev_cos - mu0) / sigma0, cos_mid, ev_groups, report_groups)
    evaluation["fmr_1e-3"] = {
        "target_fmr": 1e-3,
        "with_cohort": {"groups": mid_cohort, **gap_summary(mid_cohort)},
        "global_zscore": {"groups": mid_global, **gap_summary(mid_global)},
    }

    # -- genuine pairs (identities in the evaluation split only) ----------- #
    genuine = None
    if probes is not None:
        name_to_row = {name: row for row, name in enumerate(data["name"])}
        rows = np.array([name_to_row.get(i, -1) for i in probes["identity"]])
        usable = (rows >= 0) & (part[np.maximum(rows, 0)] == "evaluation")
        probe_quality = np.array([
            compute_quality(nm, d, p, quality_config).score
            for nm, d, p in zip(probes["ada_norm"][usable], probes["det"][usable], probes["pose"][usable])
        ])
        probe_fused = fuse_embeddings(probes["ada"][usable], probes["arc"][usable], fusion_weight(probe_quality))
        template = fused[rows[usable]]
        cos_g = np.clip(np.sum(probe_fused * template, axis=1), -1, 1)
        mu_p, sd_p = calibrator.cohort_stats(probe_fused)
        mu_t, sd_t = calibrator.cohort_stats(template)
        s_g = calibrator.to_score(0.5 * ((cos_g - mu_p) / sd_p + (cos_g - mu_t) / sd_t))
        s_g_global = global_cal.to_score((cos_g - mu0) / sigma0)
        arc_g = np.sum(probes["arc"][usable] * data["arc"][rows[usable]], axis=1)
        probe_group = probes["group"][usable]
        lo, hi = calibrator.params.anchor_low, calibrator.params.anchor_high
        genuine = {"note": (
            "Genuine pairs are two generated images of one synthetic identity. Identity consistency "
            "of the generator is not validated, so read these as relative differences between "
            "groups, not as absolute FNMR."), "groups": {}}
        for g in report_groups:
            m = probe_group == g
            if not m.any():
                continue
            ok = m & (probe_quality >= 0.40)
            genuine["groups"][g] = {
                "pairs": int(m.sum()),
                "fused_cosine_mean": float(cos_g[m].mean()),
                "low_quality_abstention": float((probe_quality[m] < 0.40).mean()),
                "fnmr_with_cohort": float((s_g[ok] < lo).mean()),
                "not_confirmed_with_cohort": float((s_g[ok] < hi).mean()),
                "fnmr_global_zscore": float((s_g_global[ok] < lo).mean()),
                "not_confirmed_global_zscore": float((s_g_global[ok] < hi).mean()),
                "fnmr_baseline_arcface": float((arc_g[m] < baseline_threshold).mean()),
            }

    # -- shipped fit: calibration + evaluation identities ------------------ #
    # The held-out numbers above are the honest estimate of generalisation.
    # The parameters that ship are then refitted on all non-cohort identities
    # so the tail quantiles rest on as many pairs as the data allows.
    final_idx = np.where(part != "cohort")[0]
    fin_z, fin_cos, fi, fj = pair_scores(normaliser, fused[final_idx])
    fin_groups = same_group_labels(final_idx, fi, fj)
    hard_final = fin_groups != "cross-group"
    fin_z, fin_cos, fin_groups, fi, fj = (
        fin_z[hard_final], fin_cos[hard_final], fin_groups[hard_final], fi[hard_final], fj[hard_final])
    mu0_final, sigma0_final = float(fin_cos.mean()), float(fin_cos.std())
    shipped = ScoreCalibrator.fit(
        cohort, fin_z, fin_groups, mu0=mu0_final, sigma0=sigma0_final, top_k=top_k,
        fmr_low=args.fmr_low, fmr_high=args.fmr_high,
        source=f"{args.dataset_name}: cohort AS-norm, worst-group anchors")
    arc_fin = data["arc"][final_idx]
    baseline_shipped = float(np.quantile(np.clip(arc_fin @ arc_fin.T, -1, 1)[fi, fj], 1.0 - args.baseline_fmr))

    # -- write assets ------------------------------------------------------ #
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / "cohort.npz", embeddings=cohort.astype(np.float16))
    params = json.loads(shipped.params.to_json())
    params["quality"] = {"norm_min": quality_config.norm_min, "norm_max": quality_config.norm_max}
    params["baseline_cosine_threshold"] = round(baseline_shipped, 4)
    params["notes"] = {
        **params.get("notes", {}),
        "dataset": args.dataset_name, "dataset_license": args.dataset_license,
        "cohort": cohort_note, "calibration_faces": int(len(final_idx)), "calibration_pairs": int(fin_z.size),
        "held_out_report": "calibration_report.json",
        "group_axis": "labels of the calibration dataset (offline only)",
    }
    (args.out / "calibration.json").write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
    report = {
        "dataset": args.dataset_name, "dataset_license": args.dataset_license,
        "what_this_measures": (
            "False-match rates between different people of the same labelled group, on identities "
            "held out from both the cohort and the threshold fit. One image per person: genuine "
            "pairs are not available here, so FNMR is not measured by this report."
        ),
        "faces": {"total": int(n), "cohort": int((part == "cohort").sum()),
                  "calibration": int(len(cal_idx)), "evaluation": int(len(ev_idx))},
        "cohort": cohort_note, "top_k": top_k,
        "quality_bounds": params["quality"], "quality_by_group": quality_by_group,
        "arcface_impostor_cosine_mean_by_group": impostor_means,
        "evaluation": evaluation,
        "genuine_pairs": genuine,
        "held_out_fit": {"z_low": calibrator.params.z_low, "z_high": calibrator.params.z_high,
                         "skipped_groups": calibrator.params.notes.get("skipped_groups")},
        "shipped_fit": {"z_low": shipped.params.z_low, "z_high": shipped.params.z_high,
                        "skipped_groups": shipped.params.notes.get("skipped_groups"),
                        "faces": int(len(final_idx)), "pairs": int(fin_z.size)},
    }
    (args.out / "calibration_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(json.dumps({"quality_by_group": quality_by_group, "impostor_means": impostor_means}, indent=1))
    for key in ("tau_low", "tau_high"):
        e = evaluation[key]
        print(f"{key} target {e['target_fmr']:g}: cohort gap {e['with_cohort']['gap']:.2e} ser {e['with_cohort']['ser']}"
              f" | global gap {e['global_zscore']['gap']:.2e} ser {e['global_zscore']['ser']}")
        print("   cohort ", {g: f"{v['fmr']:.1e}" for g, v in e["with_cohort"]["groups"].items()})
        print("   global ", {g: f"{v['fmr']:.1e}" for g, v in e["global_zscore"]["groups"].items()})
    if genuine:
        print("genuine", json.dumps(genuine["groups"], indent=1))
    print("held-out fit skipped groups:", calibrator.params.notes.get("skipped_groups"),
          "| shipped fit skipped groups:", shipped.params.notes.get("skipped_groups"))
    print("baseline", {g: f"{v['fmr']:.1e}" for g, v in baseline_eval.items()}, "thr", round(baseline_threshold, 3))
    print(f"wrote {args.out}/cohort.npz ({len(cohort)} x {cohort.shape[1]}), calibration.json, calibration_report.json")


if __name__ == "__main__":
    main()
