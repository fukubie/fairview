"""Stage 4: subgroup-equalised score calibration without demographic labels.

Why a plain threshold is unfair
-------------------------------
Face embeddings are not uniformly spread: for under-represented groups the
identities sit closer together, so impostor pairs inside those groups score
systematically higher. On the published RFW scores bundled with FaVFA the mean
impostor cosine is 0.197 for African pairs against 0.144 for Caucasian pairs,
and one global threshold tuned to an overall FMR of 1 % yields 2.5 % FMR for
the former and 0.3 % for the latter. No monotone global rescaling of the score
(Platt scaling included) can change that: it moves the threshold, not the
ordering.

What this module does instead
-----------------------------
1. **Adaptive symmetric cohort normalisation (AS-norm).** A fixed cohort of
   reference embeddings is compared against the probe and against the enrolled
   template. Each side's ``top_k`` *most similar* cohort scores estimate the
   local impostor distribution around that particular face::

       z = 1/2 * [ (c - mu_probe) / sigma_probe + (c - mu_template) / sigma_template ]

   A face living in a dense region of embedding space gets a higher ``mu`` and
   its raw scores are discounted accordingly. The correction is driven purely
   by geometry around the individual; no race, ethnicity or skin-tone label is
   predicted or consulted at run time.
2. **Logistic (Platt-form) map** ``s = sigmoid(a * z + b)`` onto ``[0, 1]``.
   ``a`` and ``b`` are solved so that two score anchors land on two impostor
   quantiles: ``s = anchor_low`` at the z giving ``fmr_low`` and
   ``s = anchor_high`` at the z giving ``fmr_high``. When demographic labels are
   available *offline, on the calibration set*, each quantile is taken from the
   worst-off group, so every group's FMR is at or below target at the anchors.

Without a cohort the calibrator degrades to a global z-score followed by the
same logistic map. That mode is monotone in the cosine and therefore cannot
improve subgroup parity; ``describe()`` reports which mode is active so the UI
and audit trail never overstate it.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


@dataclass(frozen=True)
class CalibrationParams:
    a: float  # logistic slope on z
    b: float  # logistic intercept
    mu0: float  # global impostor cosine mean (cohort-free fallback)
    sigma0: float  # global impostor cosine std (cohort-free fallback)
    top_k: int = 100
    sigma_floor: float = 0.02
    # Impostor false-match rates the two anchors were fitted to; None when the
    # parameters were not fitted on data (built-in fallback).
    fmr_low: float | None = 1e-2
    fmr_high: float | None = 1e-4
    anchor_low: float = 0.48
    anchor_high: float = 0.78
    z_low: float | None = None
    z_high: float | None = None
    source: str = "unspecified"
    notes: dict = field(default_factory=dict)

    def validate(self) -> "CalibrationParams":
        for name in ("a", "b", "mu0", "sigma0"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"calibration parameter {name} is not finite")
        if self.a <= 0:
            raise ValueError("calibration slope must be positive (score must grow with similarity)")
        if self.sigma0 <= 0 or self.sigma_floor <= 0 or self.top_k < 1:
            raise ValueError("sigma0, sigma_floor and top_k must be positive")
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_file(cls, path: Path) -> "CalibrationParams":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known).validate()


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class ScoreCalibrator:
    """Cohort-normalised, FMR-anchored mapping from cosine to ``s in [0, 1]``."""

    def __init__(self, params: CalibrationParams, cohort: np.ndarray | None = None) -> None:
        self.params = params.validate()
        if cohort is not None:
            cohort = np.asarray(cohort, dtype=np.float32)
            if cohort.ndim != 2 or cohort.shape[0] < 2:
                raise ValueError("cohort must be a (K, D) matrix with K >= 2")
            if not np.all(np.isfinite(cohort)):
                raise ValueError("cohort contains non-finite values")
            cohort = _normalise_rows(cohort)
        self.cohort = cohort

    # -- introspection ----------------------------------------------------- #

    @property
    def uses_cohort(self) -> bool:
        return self.cohort is not None

    @property
    def effective_top_k(self) -> int:
        return 0 if self.cohort is None else max(2, min(self.params.top_k, self.cohort.shape[0]))

    def describe(self) -> dict:
        return {
            "mode": "adaptive_cohort_snorm" if self.uses_cohort else "global_zscore",
            "equalises_subgroups": self.uses_cohort,
            "fitted_on_data": self.params.fmr_low is not None and self.params.fmr_high is not None,
            "cohort_size": 0 if self.cohort is None else int(self.cohort.shape[0]),
            "top_k": self.effective_top_k,
            "source": self.params.source,
            "fmr_at_tau_low_anchor": self.params.fmr_low,
            "fmr_at_tau_high_anchor": self.params.fmr_high,
            "anchor_low": self.params.anchor_low,
            "anchor_high": self.params.anchor_high,
        }

    # -- normalisation ----------------------------------------------------- #

    def cohort_stats(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-vector ``(mu, sigma)`` of its ``top_k`` nearest cohort scores.

        ``vectors`` is ``(N, D)`` with unit rows. Without a cohort the global
        impostor statistics are returned for every row.
        """
        vectors = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        count = vectors.shape[0]
        if self.cohort is None:
            return (
                np.full(count, self.params.mu0, dtype=np.float64),
                np.full(count, max(self.params.sigma0, self.params.sigma_floor), dtype=np.float64),
            )
        if vectors.shape[1] != self.cohort.shape[1]:
            raise ValueError(
                f"embedding dimension {vectors.shape[1]} does not match the cohort ({self.cohort.shape[1]})"
            )
        k = self.effective_top_k
        scores = vectors @ self.cohort.T  # (N, K)
        if k < scores.shape[1]:
            top = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:]
        else:
            top = scores
        top = top.astype(np.float64)
        mu = top.mean(axis=1)
        sigma = np.maximum(top.std(axis=1), self.params.sigma_floor)
        return mu, sigma

    def normalise(
        self,
        cosine: np.ndarray,
        probe_stats: tuple[np.ndarray, np.ndarray],
        template_stats: tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        """AS-norm of a ``(P, G)`` cosine matrix given per-row / per-column stats."""
        cosine = np.asarray(cosine, dtype=np.float64)
        mu_p, sd_p = (np.asarray(x, dtype=np.float64).reshape(-1, 1) for x in probe_stats)
        mu_t, sd_t = (np.asarray(x, dtype=np.float64).reshape(1, -1) for x in template_stats)
        return 0.5 * ((cosine - mu_p) / sd_p + (cosine - mu_t) / sd_t)

    def to_score(self, z: np.ndarray | float) -> np.ndarray:
        return sigmoid(self.params.a * np.asarray(z, dtype=np.float64) + self.params.b)

    def score_matrix(
        self,
        probes: np.ndarray,
        templates: np.ndarray,
        template_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cosine, calibrated)`` matrices of shape ``(P, G)``."""
        probes = np.atleast_2d(np.asarray(probes, dtype=np.float32))
        templates = np.atleast_2d(np.asarray(templates, dtype=np.float32))
        if templates.shape[0] == 0:
            empty = np.zeros((probes.shape[0], 0), dtype=np.float64)
            return empty, empty
        cosine = np.clip((probes @ templates.T).astype(np.float64), -1.0, 1.0)
        if template_stats is None:
            template_stats = self.cohort_stats(templates)
        z = self.normalise(cosine, self.cohort_stats(probes), template_stats)
        return cosine, self.to_score(z)

    # -- fitting ----------------------------------------------------------- #

    @staticmethod
    def solve_logistic(
        z_low: float, z_high: float, anchor_low: float = 0.48, anchor_high: float = 0.78
    ) -> tuple[float, float]:
        """Slope/intercept putting ``anchor_low`` at ``z_low`` and ``anchor_high`` at ``z_high``."""
        if not (0 < anchor_low < anchor_high < 1):
            raise ValueError("anchors must satisfy 0 < anchor_low < anchor_high < 1")
        if not (math.isfinite(z_low) and math.isfinite(z_high)) or z_high - z_low < 1e-6:
            raise ValueError("calibration quantiles are degenerate (z_high must exceed z_low)")
        a = (_logit(anchor_high) - _logit(anchor_low)) / (z_high - z_low)
        return a, _logit(anchor_low) - a * z_low

    @staticmethod
    def impostor_quantile(z_impostor: np.ndarray, fmr: float, min_events: int = 10) -> float:
        """z value exceeded by a fraction ``fmr`` of impostor scores.

        Refuses to read a tail quantile supported by fewer than ``min_events``
        observations; a threshold resting on a handful of pairs is noise.
        """
        z = np.asarray(z_impostor, dtype=np.float64)
        z = z[np.isfinite(z)]
        if z.size * fmr < min_events:
            raise ValueError(
                f"need at least {math.ceil(min_events / fmr)} impostor scores to estimate "
                f"FMR={fmr:g}; got {z.size}"
            )
        return float(np.quantile(z, 1.0 - fmr))

    @classmethod
    def fit(
        cls,
        cohort: np.ndarray | None,
        z_impostor: np.ndarray,
        groups: np.ndarray | None = None,
        *,
        mu0: float,
        sigma0: float,
        top_k: int = 100,
        sigma_floor: float = 0.02,
        fmr_low: float = 1e-2,
        fmr_high: float = 1e-4,
        anchor_low: float = 0.48,
        anchor_high: float = 0.78,
        source: str = "fit",
        notes: dict | None = None,
    ) -> "ScoreCalibrator":
        """Fit the logistic anchors on already-normalised impostor z-scores.

        ``groups`` (optional, offline labels of the calibration pairs) switches
        each quantile to the worst-off group. A group too small to support a
        given quantile is left out of that quantile only (the pooled value
        still applies to it) and is listed in ``notes['skipped_groups']``.
        """
        z_impostor = np.asarray(z_impostor, dtype=np.float64)
        notes = dict(notes or {})
        z_low = cls.impostor_quantile(z_impostor, fmr_low)
        z_high = cls.impostor_quantile(z_impostor, fmr_high)
        if groups is not None:
            groups = np.asarray(groups)
            lows, highs = [z_low], [z_high]
            skipped: dict[str, list[str]] = {"low": [], "high": []}
            per_group: dict[str, dict] = {}
            for label in np.unique(groups):
                member = z_impostor[groups == label]
                entry: dict[str, float | None] = {"z_low": None, "z_high": None, "pairs": int(member.size)}
                for key, fmr, bucket in (("z_low", fmr_low, lows), ("z_high", fmr_high, highs)):
                    try:
                        entry[key] = cls.impostor_quantile(member, fmr)
                        bucket.append(entry[key])
                    except ValueError:
                        skipped["low" if key == "z_low" else "high"].append(str(label))
                per_group[str(label)] = entry
            z_low, z_high = max(lows), max(highs)
            notes.update({"per_group_quantiles": per_group, "skipped_groups": skipped, "worst_group_anchoring": True})
        a, b = cls.solve_logistic(z_low, z_high, anchor_low, anchor_high)
        params = CalibrationParams(
            a=a, b=b, mu0=float(mu0), sigma0=float(sigma0), top_k=int(top_k),
            sigma_floor=float(sigma_floor), fmr_low=fmr_low, fmr_high=fmr_high,
            anchor_low=anchor_low, anchor_high=anchor_high, z_low=z_low, z_high=z_high,
            source=source, notes=notes,
        )
        return cls(params, cohort)

    # -- persistence ------------------------------------------------------- #

    @classmethod
    def load(cls, params_path: Path, cohort_path: Path | None = None) -> "ScoreCalibrator":
        params = CalibrationParams.from_file(params_path)
        cohort = None
        if cohort_path is not None and Path(cohort_path).exists():
            with np.load(cohort_path, allow_pickle=False) as archive:
                cohort = archive["embeddings"].astype(np.float32)
        return cls(params, cohort)
