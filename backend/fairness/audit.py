"""Offline fairness benchmark: subgroup FMR / FNMR parity and abstention.

Three things live here.

1. **Metric definitions** on plain NumPy (no fairlearn dependency):
   per-group False Match Rate and False Non-Match Rate, Equalized Odds
   difference, Skewed Error Ratio, Fairness Discrepancy Rate and per-group
   False Discovery Rate. A rate whose denominator is empty is ``NaN`` and is
   excluded from gaps. (fairlearn/scikit-learn silently report 0.0 there,
   which can manufacture or hide a disparity.)

2. **A synthetic embedding benchmark** across Fitzpatrick skin-type groups
   I-VI. It is a *simulation*: identities are random vectors whose within-group
   concentration spans the impostor-similarity range measured on RFW, and
   capture quality differs by group. It runs the production fusion, calibration
   and decision code on those vectors, so it verifies that the mechanisms do
   what they claim under a stated model. It is not evidence about how the
   shipped networks behave on real people; only an evaluation on consented,
   labelled face data can establish that.

3. **A real-score report** on the RFW verification protocol, using the
   published per-pair distances that ship with FaVFA when that folder is
   present. This shows the disparity a conventional single-threshold system
   exhibits on real data. It is score-level only (no embeddings are
   available), so the cohort normalisation cannot be replayed on it.

Definitions
-----------
* FMR_g  = P(system asserts a match | impostor pair, group g)
* FNMR_g = P(system asserts a non-match | genuine pair, group g)
* For the three-zone engine an abstention is neither; abstention rates are
  reported per group alongside, because pushing one group disproportionately
  into manual review is itself a disparity.
* Equalized Odds difference = max(max_g TPR_g - min_g TPR_g,
  max_g FPR_g - min_g FPR_g) for the binary event "match asserted"
  (fairlearn's ``equalized_odds_difference``, between-groups, worst case).
* Skewed Error Ratio (SER) = max_g err_g / min_g err_g (1 = parity).
* Fairness Discrepancy Rate (de Freitas Pereira & Marcel, 2021)
  = 1 - (alpha * max FMR gap + (1 - alpha) * max FNMR gap); 1 = parity.
* False Discovery Rate_g = false matches / all asserted matches in group g.
  It depends on the genuine:impostor mix of the benchmark, so compare it
  between engines, not against deployments with a different mix.
"""

from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import BASE_DIR, Thresholds
from backend.engine.calibrator import CalibrationParams, ScoreCalibrator
from backend.engine.decision import Candidate, Mode, Zone, decide
from backend.engine.embedder import fuse_embeddings, fusion_weight

FITZPATRICK_GROUPS = ("I", "II", "III", "IV", "V", "VI")

# Integer codes for vectorised zone arrays.
Z_MATCH, Z_NO_MATCH, Z_UNCERTAIN = 0, 1, 2


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #


def _rate(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator > 0 else float("nan")


def _json_number(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def max_gap(values: list[float]) -> float:
    """max - min over the defined (non-NaN) group rates; NaN if fewer than two."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    return max(finite) - min(finite) if len(finite) >= 2 else float("nan")


def skewed_error_ratio(values: list[float]) -> float:
    """max / min over defined group error rates.

    ``inf`` when the best group has zero errors and another does not (reported
    as ``null`` in JSON with the raw rates alongside); ``1.0`` when every group
    has zero errors; ``NaN`` with fewer than two defined groups.
    """
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    low, high = min(finite), max(finite)
    if high == 0:
        return 1.0
    return float("inf") if low == 0 else high / low


def fairness_discrepancy_rate(fmr_gap: float, fnmr_gap: float, alpha: float = 0.5) -> float:
    if not (math.isfinite(fmr_gap) and math.isfinite(fnmr_gap)):
        return float("nan")
    return 1.0 - (alpha * fmr_gap + (1.0 - alpha) * fnmr_gap)


def group_outcome_rates(
    zones: np.ndarray, genuine: np.ndarray, groups: np.ndarray, labels: list[str] | None = None
) -> dict[str, dict]:
    """Per-group outcome rates from zone codes, ground truth and group labels.

    ``groups`` holds arbitrary labels, or (faster) integer codes indexing
    ``labels``.
    """
    zones, genuine, groups = np.asarray(zones), np.asarray(genuine, dtype=bool), np.asarray(groups)
    if not (zones.shape == genuine.shape == groups.shape):
        raise ValueError("zones, genuine and groups must have the same shape")
    if labels is None:
        names, codes = np.unique(groups.astype(str), return_inverse=True)
        labels = [str(n) for n in names]
    else:
        codes = groups.astype(np.int64)
        if codes.size and (codes.min() < 0 or codes.max() >= len(labels)):
            raise ValueError("group codes fall outside the label list")
    if zones.size and (zones.min() < 0 or zones.max() > Z_UNCERTAIN):
        raise ValueError("unknown zone code")
    # One pass: joint histogram over (group, genuine?, zone).
    key = (codes.ravel() * 2 + genuine.ravel()) * 3 + zones.ravel().astype(np.int64)
    counts = np.bincount(key, minlength=len(labels) * 6).reshape(len(labels), 2, 3)
    report: dict[str, dict] = {}
    for index, label in enumerate(labels):
        imp, gen = counts[index, 0], counts[index, 1]
        n_imp, n_gen = int(imp.sum()), int(gen.sum())
        if n_imp + n_gen == 0:
            continue
        false_matches, true_matches = int(imp[Z_MATCH]), int(gen[Z_MATCH])
        false_non_matches = int(gen[Z_NO_MATCH])
        report[label] = {
            "n_genuine": n_gen,
            "n_impostor": n_imp,
            "false_matches": false_matches,
            "false_non_matches": false_non_matches,
            "fmr": _rate(false_matches, n_imp),
            "fnmr": _rate(false_non_matches, n_gen),
            "tpr": _rate(true_matches, n_gen),
            "genuine_abstention": _rate(int(gen[Z_UNCERTAIN]), n_gen),
            "impostor_abstention": _rate(int(imp[Z_UNCERTAIN]), n_imp),
            "false_discovery_rate": _rate(false_matches, false_matches + true_matches),
        }
    return report


def summarise(per_group: dict[str, dict], alpha: float = 0.5) -> dict:
    """Parity summary over a ``group_outcome_rates`` report."""
    fmr = [g["fmr"] for g in per_group.values()]
    fnmr = [g["fnmr"] for g in per_group.values()]
    tpr = [g["tpr"] for g in per_group.values()]
    fmr_gap, fnmr_gap, tpr_gap = max_gap(fmr), max_gap(fnmr), max_gap(tpr)
    eo_parts = [v for v in (tpr_gap, fmr_gap) if math.isfinite(v)]
    n_gen = sum(g["n_genuine"] for g in per_group.values())
    n_imp = sum(g["n_impostor"] for g in per_group.values())
    return {
        "overall_fmr": _rate(sum(g["false_matches"] for g in per_group.values()), n_imp),
        "overall_fnmr": _rate(sum(g["false_non_matches"] for g in per_group.values()), n_gen),
        "fmr_gap": fmr_gap,
        "fnmr_gap": fnmr_gap,
        "tpr_gap": tpr_gap,
        "equalized_odds_difference": max(eo_parts) if eo_parts else float("nan"),
        "ser_fmr": skewed_error_ratio(fmr),
        "ser_fnmr": skewed_error_ratio(fnmr),
        "fairness_discrepancy_rate": fairness_discrepancy_rate(fmr_gap, fnmr_gap, alpha),
        "false_discovery_rate_gap": max_gap([g["false_discovery_rate"] for g in per_group.values()]),
        "genuine_abstention_gap": max_gap([g["genuine_abstention"] for g in per_group.values()]),
        "impostor_abstention_gap": max_gap([g["impostor_abstention"] for g in per_group.values()]),
    }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        return _json_number(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# Vectorised decision rules (1:1 verification)
# --------------------------------------------------------------------------- #


def verify_zones(scores: np.ndarray, quality: np.ndarray, thresholds: Thresholds) -> np.ndarray:
    """Three-zone outcome for 1:1 comparisons; mirrors ``decision.decide``.

    ``scores`` is ``(P, G)``, ``quality`` is per probe ``(P,)``. Liveness is
    not simulated. ``tests/test_fairness.py`` asserts this stays equivalent to
    the scalar reference implementation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    low_quality = (np.asarray(quality, dtype=np.float64) < thresholds.q_min).reshape(-1, 1)
    zones = np.full(scores.shape, Z_UNCERTAIN, dtype=np.int8)
    zones[scores >= thresholds.tau_high] = Z_MATCH
    zones[scores < thresholds.tau_low] = Z_NO_MATCH
    zones[np.broadcast_to(low_quality, scores.shape)] = Z_UNCERTAIN
    zones[~np.isfinite(scores)] = Z_UNCERTAIN
    bad_quality = ~np.isfinite(np.asarray(quality, dtype=np.float64)).reshape(-1, 1)
    zones[np.broadcast_to(bad_quality, scores.shape)] = Z_UNCERTAIN  # NaN compares False above
    return zones


def baseline_zones(cosine: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(np.asarray(cosine) >= threshold, Z_MATCH, Z_NO_MATCH).astype(np.int8)


# --------------------------------------------------------------------------- #
# Synthetic Fitzpatrick benchmark
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of the simulated population. All are modelling assumptions."""

    seed: int = 7
    dim: int = 128  # per-backbone dimensionality of the simulated spaces
    identities: int = 320  # evaluation identities per group
    probes_per_identity: int = 3
    enrollment_samples: int = 3
    cohort_per_group: int = 200
    calibration_identities: int = 320
    # Shared similarity floor between any two faces, plus the extra
    # within-group similarity for types I..VI. The sums (0.13 .. 0.20) span the
    # mean impostor cosines measured per ethnicity on RFW (0.144 .. 0.197).
    shared_similarity: float = 0.10
    group_similarity: tuple[float, ...] = (0.03, 0.03, 0.04, 0.06, 0.08, 0.10)
    # Mean capture quality per group: cameras and auto-exposure serve darker
    # skin worse, which is the acquisition-side source of FNMR disparity.
    quality_mean: tuple[float, ...] = (0.74, 0.73, 0.71, 0.68, 0.64, 0.60)
    quality_concentration: float = 9.0
    quality_estimator_noise: float = 0.05
    # Embedding noise variance as a function of true quality q:
    # var = base + slope * (1 - q)^2. The AdaFace-like backbone is slightly
    # worse on clean captures and degrades more slowly on poor ones.
    arc_noise: tuple[float, float] = (0.55, 4.0)
    ada_noise: tuple[float, float] = (0.70, 2.2)
    twin_pairs_per_group: int = 60
    twin_similarity: tuple[float, float] = (0.45, 0.95)
    distractors: int = 40
    baseline_target_fmr: float = 1e-3


class _Population:
    """Generator of simulated identities and captures in two backbone spaces."""

    def __init__(self, config: SimulationConfig, rng: np.random.Generator) -> None:
        self.config, self.rng = config, rng
        d = config.dim
        self.shared = {name: self._unit(rng.standard_normal(d)) for name in ("ada", "arc")}
        self.group_dirs = {
            name: np.stack([self._unit(rng.standard_normal(d)) for _ in FITZPATRICK_GROUPS])
            for name in ("ada", "arc")
        }

    @staticmethod
    def _unit(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def identities(self, group: int, count: int) -> dict[str, np.ndarray]:
        beta, gamma = self.config.shared_similarity, self.config.group_similarity[group]
        out = {}
        for name in ("ada", "arc"):
            unique = self._unit(self.rng.standard_normal((count, self.config.dim)))
            out[name] = self._unit(
                math.sqrt(beta) * self.shared[name]
                + math.sqrt(gamma) * self.group_dirs[name][group]
                + math.sqrt(1.0 - beta - gamma) * unique
            )
        return out

    def lookalikes(self, identities: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        low, high = self.config.twin_similarity
        count = identities["arc"].shape[0]
        rho = self.rng.uniform(low, high, size=(count, 1))
        out = {}
        for name in ("ada", "arc"):
            other = self._unit(self.rng.standard_normal((count, self.config.dim)))
            out[name] = self._unit(rho * identities[name] + np.sqrt(1 - rho**2) * other)
        return out

    def quality(self, group: int, count: int, boost: float = 0.0) -> np.ndarray:
        mean = min(0.95, self.config.quality_mean[group] + boost)
        k = self.config.quality_concentration
        return self.rng.beta(mean * k, (1 - mean) * k, size=count)

    def observe_quality(self, true_quality: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.config.quality_estimator_noise, size=true_quality.shape)
        return np.clip(true_quality + noise, 0.0, 1.0)

    def capture(self, identities: dict[str, np.ndarray], quality: np.ndarray) -> dict[str, np.ndarray]:
        """One noisy sample per row; noise grows as quality falls."""
        out = {}
        for name, (base, slope) in (("ada", self.config.ada_noise), ("arc", self.config.arc_noise)):
            variance = base + slope * (1.0 - quality) ** 2
            noise = self.rng.standard_normal(identities[name].shape) / math.sqrt(self.config.dim)
            out[name] = self._unit(identities[name] + np.sqrt(variance)[:, None] * noise)
        return out

    def enroll(self, identities: dict[str, np.ndarray], group: int) -> tuple[np.ndarray, np.ndarray]:
        """Multi-sample templates. Returns ``(fused_templates, arc_templates)``."""
        count = identities["arc"].shape[0]
        fused_sum, arc_sum = 0.0, 0.0
        for _ in range(self.config.enrollment_samples):
            q = self.quality(group, count, boost=0.10)
            sample = self.capture(identities, q)
            weight = fusion_weight(self.observe_quality(q))
            fused_sum = fused_sum + fuse_embeddings(sample["ada"], sample["arc"], weight)
            arc_sum = arc_sum + sample["arc"]
        return self._unit(fused_sum).astype(np.float32), self._unit(arc_sum).astype(np.float32)

    def probe(self, identities: dict[str, np.ndarray], group: int):
        """Returns ``(fused_probes, arc_probes, observed_quality)``."""
        q = self.quality(group, identities["arc"].shape[0])
        sample = self.capture(identities, q)
        observed = self.observe_quality(q)
        fused = fuse_embeddings(sample["ada"], sample["arc"], fusion_weight(observed))
        return fused.astype(np.float32), sample["arc"].astype(np.float32), observed


class FairnessBenchmark:
    """Builds the simulated score sets once; re-evaluates thresholds instantly."""

    def __init__(self, config: SimulationConfig | None = None) -> None:
        self.config = config or SimulationConfig()
        started, cpu_started = time.perf_counter(), time.process_time()
        rng = np.random.default_rng(self.config.seed)
        pop = _Population(self.config, rng)
        cfg = self.config
        groups = range(len(FITZPATRICK_GROUPS))

        # 1. Reference cohort: disjoint identities, one capture each.
        cohort_parts = []
        for g in groups:
            fused, _, _ = pop.probe(pop.identities(g, cfg.cohort_per_group), g)
            cohort_parts.append(fused)
        cohort = np.concatenate(cohort_parts)

        # 2. Calibration split (disjoint identities, offline group labels).
        # Cohort statistics only; the logistic anchors are fitted below.
        normaliser = ScoreCalibrator(CalibrationParams(a=1.0, b=0.0, mu0=0.0, sigma0=1.0), cohort)
        cal_z, cal_groups, cal_arc_impostor, cal_fused_impostor = [], [], [], []
        for g in groups:
            ids = pop.identities(g, cfg.calibration_identities)
            templates, arc_templates = pop.enroll(ids, g)
            probes, arc_probes, _ = pop.probe(ids, g)
            cosine = probes @ templates.T
            z = normaliser.normalise(cosine, normaliser.cohort_stats(probes), normaliser.cohort_stats(templates))
            off_diagonal = ~np.eye(len(z), dtype=bool)
            cal_z.append(z[off_diagonal])
            cal_groups.append(np.full(int(off_diagonal.sum()), g))
            cal_fused_impostor.append(cosine[off_diagonal])
            cal_arc_impostor.append((arc_probes @ arc_templates.T)[off_diagonal])
        cal_z_all = np.concatenate(cal_z)
        fused_impostor = np.concatenate(cal_fused_impostor)
        self.calibrator = ScoreCalibrator.fit(
            cohort, cal_z_all, np.concatenate(cal_groups),
            mu0=float(fused_impostor.mean()), sigma0=float(fused_impostor.std()),
            source="synthetic Fitzpatrick calibration split",
        )
        # Same logistic anchors, but no cohort: isolates what AS-norm contributes.
        global_z = (fused_impostor - fused_impostor.mean()) / fused_impostor.std()
        self.global_calibrator = ScoreCalibrator.fit(
            None, global_z, None,
            mu0=float(fused_impostor.mean()), sigma0=float(fused_impostor.std()),
            source="synthetic calibration split, global z-score",
        )
        self.baseline_default_threshold = float(
            np.quantile(np.concatenate(cal_arc_impostor), 1.0 - cfg.baseline_target_fmr)
        )

        # 3. Evaluation split.
        self._scores, self._scores_global, self._arc, self._quality = [], [], [], []
        self._genuine, self._group = [], []
        template_stats_cache = {}
        self._eval_sets = {}
        for g in groups:
            ids = pop.identities(g, cfg.identities)
            templates, arc_templates = pop.enroll(ids, g)
            template_stats_cache[g] = self.calibrator.cohort_stats(templates)
            self._eval_sets[g] = (ids, templates, arc_templates)
            owner = np.arange(cfg.identities)
            for _ in range(cfg.probes_per_identity):
                probes, arc_probes, observed_q = pop.probe(ids, g)
                _, scores = self.calibrator.score_matrix(probes, templates, template_stats_cache[g])
                _, scores_global = self.global_calibrator.score_matrix(probes, templates)
                self._scores.append(scores.astype(np.float32))
                self._scores_global.append(scores_global.astype(np.float32))
                self._arc.append((arc_probes @ arc_templates.T).astype(np.float32))
                self._quality.append(observed_q.astype(np.float32))
                self._genuine.append(owner[:, None] == owner[None, :])
                self._group.append(np.full((cfg.identities, cfg.identities), g, dtype=np.int8))
        self._group_flat = np.concatenate([x.ravel() for x in self._group])
        self._genuine_flat = np.concatenate([x.ravel() for x in self._genuine])
        # Threshold-independent artefacts, computed once so slider moves are cheap.
        arc_flat = np.concatenate([c.ravel() for c in self._arc])
        self._arc_impostor_sorted = np.sort(arc_flat[~self._genuine_flat])
        self._score_grid = np.linspace(0.0, 1.0, 51)
        self._cosine_grid = np.linspace(-0.1, 0.9, 51)
        self._curve_cache = {
            "fair": {"threshold": self._score_grid.tolist(),
                     "groups": self._curves(self._scores, self._score_grid)},
            "baseline": {"threshold": self._cosine_grid.tolist(),
                         "groups": self._curves(self._arc, self._cosine_grid)},
        }

        # 4. Lookalike 1:N scenarios.
        self._lookalike = self._build_lookalikes(pop)
        self.build_seconds = time.perf_counter() - started
        self.build_cpu_seconds = time.process_time() - cpu_started

    # -- lookalikes -------------------------------------------------------- #

    def _build_lookalikes(self, pop: _Population) -> list[dict]:
        cfg = self.config
        cases = []
        for g, label in enumerate(FITZPATRICK_GROUPS):
            a_ids = pop.identities(g, cfg.twin_pairs_per_group)
            b_ids = pop.lookalikes(a_ids)
            distractor_ids = pop.identities(g, cfg.distractors)
            a_fused, a_arc = pop.enroll(a_ids, g)
            b_fused, b_arc = pop.enroll(b_ids, g)
            d_fused, d_arc = pop.enroll(distractor_ids, g)
            probes, arc_probes, quality = pop.probe(a_ids, g)
            for i in range(cfg.twin_pairs_per_group):
                names = ["A", "B"] + [f"D{j}" for j in range(cfg.distractors)]
                fused_gallery = np.concatenate([a_fused[i : i + 1], b_fused[i : i + 1], d_fused])
                arc_gallery = np.concatenate([a_arc[i : i + 1], b_arc[i : i + 1], d_arc])
                cosine, scores = self.calibrator.score_matrix(probes[i : i + 1], fused_gallery)
                cases.append({
                    "group": label,
                    "names": names,
                    "scores": scores[0],
                    "cosine": cosine[0],
                    "arc": (arc_probes[i : i + 1] @ arc_gallery.T)[0],
                    "quality": float(quality[i]),
                })
        return cases

    def _evaluate_lookalikes(self, thresholds: Thresholds, baseline_threshold: float) -> dict:
        def tally() -> dict:
            return {"correct_match": 0, "wrong_match": 0, "uncertain": 0, "no_match": 0}

        scenarios = {
            # The true subject and their lookalike are both enrolled.
            "both_enrolled": {"fair": tally(), "baseline": tally()},
            # Only the lookalike is enrolled; the probe belongs to nobody on file.
            "lookalike_only": {"fair": tally(), "baseline": tally()},
        }
        reasons: dict[str, int] = {}
        for case in self._lookalike:
            for scenario, drop in (("both_enrolled", None), ("lookalike_only", 0)):
                keep = [i for i in range(len(case["names"])) if i != drop]
                names = [case["names"][i] for i in keep]
                decision = decide(
                    Mode.IDENTIFY,
                    [Candidate(n, float(case["scores"][i]), float(case["cosine"][i])) for n, i in zip(names, keep)],
                    case["quality"], None, thresholds,
                )
                fair = scenarios[scenario]["fair"]
                if decision.zone is Zone.MATCH_CONFIRMED:
                    fair["correct_match" if decision.subject == "A" else "wrong_match"] += 1
                elif decision.zone is Zone.UNCERTAIN_REVIEW:
                    fair["uncertain"] += 1
                    if scenario == "both_enrolled":
                        reasons[decision.reason.value] = reasons.get(decision.reason.value, 0) + 1
                else:
                    fair["no_match"] += 1
                arc = case["arc"][keep]
                best = int(np.argmax(arc))
                base = scenarios[scenario]["baseline"]
                if arc[best] >= baseline_threshold:
                    base["correct_match" if names[best] == "A" else "wrong_match"] += 1
                else:
                    base["no_match"] += 1
        total = len(self._lookalike)
        for scenario in scenarios.values():
            for engine in scenario.values():
                engine["total"] = total
                engine["wrong_match_rate"] = _rate(engine["wrong_match"], total)
        scenarios["both_enrolled"]["fair"]["abstention_reasons"] = reasons
        return scenarios

    def simulate_lookalike(
        self,
        similarity: float,
        quality: float,
        both_enrolled: bool,
        thresholds: Thresholds,
        group: str = "IV",
        trials: int = 200,
        baseline_threshold: float | None = None,
        seed: int = 0,
    ) -> dict:
        """Interactive stress test: a probe of person A against a lookalike B.

        ``similarity`` is the cosine between the two identities (0.3 ~ vague
        resemblance, 0.9+ ~ identical twins); ``quality`` is the capture
        quality of the probe. With ``both_enrolled`` the gallery holds A and
        B; otherwise only B, i.e. the probe belongs to nobody on file.
        """
        if group not in FITZPATRICK_GROUPS:
            raise ValueError(f"group must be one of {FITZPATRICK_GROUPS}")
        if not (0.0 <= similarity < 1.0 and 0.0 <= quality <= 1.0):
            raise ValueError("similarity must be in [0, 1) and quality in [0, 1]")
        trials = int(min(max(trials, 1), 1000))
        baseline_threshold = self.baseline_default_threshold if baseline_threshold is None else baseline_threshold
        g = FITZPATRICK_GROUPS.index(group)
        pop = _Population(self.config, np.random.default_rng([self.config.seed, 1009, seed]))
        a_ids = pop.identities(g, trials)
        rho = float(similarity)
        b_ids = {}
        for name in ("ada", "arc"):
            other = pop._unit(pop.rng.standard_normal((trials, self.config.dim)))
            b_ids[name] = pop._unit(rho * a_ids[name] + math.sqrt(1 - rho**2) * other)
        a_fused, a_arc = pop.enroll(a_ids, g)
        b_fused, b_arc = pop.enroll(b_ids, g)
        true_q = np.full(trials, float(quality))
        sample = pop.capture(a_ids, true_q)
        observed = pop.observe_quality(true_q)
        probes = fuse_embeddings(sample["ada"], sample["arc"], fusion_weight(observed)).astype(np.float32)

        fair = {"MATCH_A": 0, "MATCH_B": 0, Zone.UNCERTAIN_REVIEW.value: 0, Zone.NO_MATCH.value: 0}
        base = {"MATCH_A": 0, "MATCH_B": 0, Zone.NO_MATCH.value: 0}
        reasons: dict[str, int] = {}
        examples = []
        for i in range(trials):
            gallery = [("B", b_fused[i], b_arc[i])]
            if both_enrolled:
                gallery.insert(0, ("A", a_fused[i], a_arc[i]))
            cosine, scores = self.calibrator.score_matrix(probes[i : i + 1], np.stack([t for _, t, _ in gallery]))
            decision = decide(
                Mode.IDENTIFY,
                [Candidate(n, float(scores[0, j]), float(cosine[0, j])) for j, (n, _, _) in enumerate(gallery)],
                float(observed[i]), None, thresholds,
            )
            if decision.zone is Zone.MATCH_CONFIRMED:
                fair[f"MATCH_{decision.subject}"] += 1
            else:
                fair[decision.zone.value] += 1
                if decision.zone is Zone.UNCERTAIN_REVIEW:
                    reasons[decision.reason.value] = reasons.get(decision.reason.value, 0) + 1
            arc = np.array([float(t @ sample["arc"][i]) for _, _, t in gallery])
            best = int(np.argmax(arc))
            if arc[best] >= baseline_threshold:
                base[f"MATCH_{gallery[best][0]}"] += 1
            else:
                base[Zone.NO_MATCH.value] += 1
            if len(examples) < 5:
                examples.append({"fair": decision.to_dict(),
                                 "baseline_cosine": {n: round(float(c), 4) for (n, _, _), c in zip(gallery, arc)}})
        return _jsonable({
            "kind": "synthetic_simulation",
            "group": group,
            "identity_similarity": rho,
            "probe_quality": float(quality),
            "both_enrolled": both_enrolled,
            "trials": trials,
            "truth": "probe is person A",
            "fair": {"outcomes": fair, "abstention_reasons": reasons,
                     "false_identification_rate": _rate(fair["MATCH_B"], trials)},
            "baseline": {"outcomes": base, "cosine_threshold": baseline_threshold,
                         "false_identification_rate": _rate(base["MATCH_B"], trials)},
            "examples": examples,
        })

    # -- evaluation -------------------------------------------------------- #

    def _engine_report(self, zones: list[np.ndarray]) -> dict:
        flat = np.concatenate([z.ravel() for z in zones])
        per_group = group_outcome_rates(
            flat, self._genuine_flat, self._group_flat, list(FITZPATRICK_GROUPS)
        )
        return {"groups": per_group, "summary": summarise(per_group)}

    def _curves(self, score_sets: list[np.ndarray], grid: np.ndarray) -> dict:
        """Per-group FMR/FNMR as the match threshold sweeps ``grid`` (no gating)."""
        curves = {}
        per_block = len(score_sets) // len(FITZPATRICK_GROUPS)
        for g, label in enumerate(FITZPATRICK_GROUPS):
            blocks = range(g * per_block, (g + 1) * per_block)
            scores = np.concatenate([score_sets[b].ravel() for b in blocks])
            genuine = np.concatenate([self._genuine[b].ravel() for b in blocks])
            gen = np.sort(scores[genuine])
            imp = np.sort(scores[~genuine])
            fmr = 1.0 - np.searchsorted(imp, grid, side="left") / imp.size
            fnmr = np.searchsorted(gen, grid, side="left") / gen.size
            curves[label] = {"fmr": fmr.tolist(), "fnmr": fnmr.tolist()}
        return curves

    def evaluate(self, thresholds: Thresholds | None = None, baseline_threshold: float | None = None) -> dict:
        thresholds = (thresholds or Thresholds()).validate()
        if baseline_threshold is None:
            baseline_threshold = self.baseline_default_threshold
        started = time.perf_counter()

        fair = self._engine_report(
            [verify_zones(s, q, thresholds) for s, q in zip(self._scores, self._quality)]
        )
        # Ablation: identical policy and anchors, cohort normalisation removed.
        no_cohort = self._engine_report(
            [verify_zones(s, q, thresholds) for s, q in zip(self._scores_global, self._quality)]
        )
        baseline = self._engine_report([baseline_zones(c, baseline_threshold) for c in self._arc])

        # Baseline re-tuned to the fair engine's pooled FMR, so the comparison
        # of gaps is not an artefact of sitting at a different operating point.
        arc_impostor = self._arc_impostor_sorted
        fair_fmr = fair["summary"]["overall_fmr"]
        matched = None
        if math.isfinite(fair_fmr) and fair_fmr * arc_impostor.size >= 10:
            rank = min(arc_impostor.size - 1, int(math.ceil((1.0 - fair_fmr) * arc_impostor.size)))
            matched_threshold = float(arc_impostor[rank])
            matched = self._engine_report([baseline_zones(c, matched_threshold) for c in self._arc])
            matched["cosine_threshold"] = matched_threshold

        result = {
            "kind": "synthetic_simulation",
            "disclaimer": (
                "Simulated embeddings under stated assumptions. Shows the behaviour of the "
                "fusion, calibration and decision code; it is not a measurement of the deployed "
                "networks on real people."
            ),
            "groups": list(FITZPATRICK_GROUPS),
            "group_axis": "Fitzpatrick skin type",
            "thresholds": thresholds.to_dict(),
            "baseline_cosine_threshold": baseline_threshold,
            "baseline_default_cosine_threshold": self.baseline_default_threshold,
            "fair": fair,
            "fair_without_cohort": no_cohort,
            "baseline": baseline,
            "baseline_matched_fmr": matched,
            "lookalikes": self._evaluate_lookalikes(thresholds, baseline_threshold),
            "curves": self._curve_cache,
            "calibration": self.calibrator.describe(),
            "simulation": {
                "seed": self.config.seed,
                "identities_per_group": self.config.identities,
                "probes_per_identity": self.config.probes_per_identity,
                "cohort_size": int(self.calibrator.cohort.shape[0]),
                "comparisons": int(self._genuine_flat.size),
                "protocol": "1:1 verification, all same-group pairs (hard impostors)",
                "within_group_similarity": [
                    round(self.config.shared_similarity + g, 3) for g in self.config.group_similarity
                ],
                "mean_quality": list(self.config.quality_mean),
            },
            "timing": {
                "build_seconds": round(self.build_seconds, 3),
                "evaluate_seconds": round(time.perf_counter() - started, 3),
            },
        }
        return _jsonable(result)


# --------------------------------------------------------------------------- #
# Real scores: RFW protocol
# --------------------------------------------------------------------------- #


def default_rfw_dir() -> Path:
    """Where to look for a FaVFA checkout with the published RFW scores.

    ``FAIRFACE_RFW_DIR`` if set, else ``FaVFA-main/`` inside the project
    (git-ignored), else ``FaVFA-main/`` next to the project folder. The data is
    CeCILL-licensed and is not shipped with Fairview.
    """
    configured = os.environ.get("FAIRFACE_RFW_DIR", "").strip()
    if configured:
        return Path(configured)
    inside = BASE_DIR / "FaVFA-main"
    return inside if inside.is_dir() else BASE_DIR.parent / "FaVFA-main"


def load_rfw_scores(root: Path | None = None) -> dict | None:
    """Load published RFW pair scores with their ethnicity labels, if present.

    Reads ``data/rfw.csv`` (pair metadata) and ``model_results/BUPT_RFW.csv``
    (L2 distance between unit embeddings) from a FaVFA checkout and joins them
    on the image pair. Result rows without metadata are dropped. Returns
    ``None`` when the files are not available.
    """
    root = Path(root) if root else default_rfw_dir()
    meta_path, score_path = root / "data" / "rfw.csv", root / "model_results" / "BUPT_RFW.csv"
    if not (meta_path.is_file() and score_path.is_file()):
        return None
    distances: dict[tuple[str, str], list[float]] = {}
    with open(score_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            distances.setdefault((row["img_1"], row["img_2"]), []).append(float(row["dist"]))
    cosine, genuine, groups = [], [], []
    with open(meta_path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            found = distances.get((row["img_1"], row["img_2"]))
            if not found or len(found) != 1:  # missing, or ambiguous duplicate key
                continue
            # Unit vectors: ||a - b||^2 = 2 - 2 cos.
            cosine.append(1.0 - found[0] ** 2 / 2.0)
            genuine.append(row["y_true"].strip() == "1")
            groups.append(row["ethnicity"].split(" x ")[0].strip())
    if not cosine:
        return None
    return {
        "cosine": np.asarray(cosine, dtype=np.float64),
        "genuine": np.asarray(genuine, dtype=bool),
        "groups": np.asarray(groups),
        "source": str(root),
    }


def rfw_report(data: dict, target_fmr: float = 1e-2, cosine_threshold: float | None = None) -> dict:
    """Per-ethnicity error rates of a single global threshold on real scores.

    Also reports the per-group thresholds that an oracle with demographic
    labels would need to equalise FMR, to show the size of the correction and
    that equalising FMR does not equalise FNMR.
    """
    cosine, genuine, groups = data["cosine"], data["genuine"], data["groups"]
    impostor = cosine[~genuine]
    if cosine_threshold is None:
        cosine_threshold = float(np.quantile(impostor, 1.0 - target_fmr))
    per_group = group_outcome_rates(baseline_zones(cosine, cosine_threshold), genuine, groups)
    oracle = {}
    for label in per_group:
        member = groups == label
        group_impostor = cosine[member & ~genuine]
        if group_impostor.size * target_fmr < 10:
            continue
        threshold = float(np.quantile(group_impostor, 1.0 - target_fmr))
        group_genuine = cosine[member & genuine]
        oracle[label] = {
            "cosine_threshold": threshold,
            "fmr": _rate(int((group_impostor >= threshold).sum()), group_impostor.size),
            "fnmr": _rate(int((group_genuine < threshold).sum()), group_genuine.size),
        }
    stats = {
        label: {
            "impostor_cosine_mean": float(cosine[(groups == label) & ~genuine].mean()),
            "genuine_cosine_mean": float(cosine[(groups == label) & genuine].mean()),
        }
        for label in per_group
    }
    min_impostors = min(g["n_impostor"] for g in per_group.values())
    return _jsonable({
        "kind": "real_scores",
        "dataset": "RFW verification pairs (published per-pair scores distributed with FaVFA)",
        "note": (
            "Real data, conventional single-threshold decision. Score-level only: the cohort "
            "normalisation needs embeddings and cannot be replayed on these scores."
        ),
        "group_axis": "ethnicity label of the pair",
        "pairs": int(cosine.size),
        "cosine_threshold": cosine_threshold,
        "target_fmr": target_fmr,
        "groups": per_group,
        "summary": summarise(per_group),
        "score_statistics": stats,
        "oracle_per_group_thresholds": oracle,
        "fmr_resolution": 1.0 / min_impostors if min_impostors else None,
    })
