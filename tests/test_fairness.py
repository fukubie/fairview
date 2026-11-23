"""Fairness metrics, subgroup FMR parity and abstention on lookalikes."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.config import Thresholds
from backend.engine.calibrator import CalibrationParams, ScoreCalibrator
from backend.engine.decision import Candidate, Mode, Zone, decide
from backend.fairness.audit import (
    FITZPATRICK_GROUPS,
    Z_MATCH,
    Z_NO_MATCH,
    Z_UNCERTAIN,
    FairnessBenchmark,
    baseline_zones,
    fairness_discrepancy_rate,
    group_outcome_rates,
    load_rfw_scores,
    max_gap,
    rfw_report,
    skewed_error_ratio,
    summarise,
    verify_zones,
)


@pytest.fixture(scope="module")
def benchmark() -> FairnessBenchmark:
    return FairnessBenchmark()


@pytest.fixture(scope="module")
def report(benchmark) -> dict:
    return benchmark.evaluate(Thresholds())


class TestMetricDefinitions:
    # Worked example cross-checked against fairlearn 0.16 (between-groups):
    # TPR A/B/C = 0.5 / 0.667 / 1.0, FPR A/C = 0.5 / 0.0, group B has no negatives.
    y_true = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    y_pred = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0])
    groups = np.array(list("AAAABBBCCCC"))

    def _report(self):
        zones = np.where(self.y_pred == 1, Z_MATCH, Z_NO_MATCH)
        return group_outcome_rates(zones, self.y_true, self.groups)

    def test_per_group_rates(self):
        r = self._report()
        assert r["A"]["tpr"] == pytest.approx(0.5) and r["A"]["fmr"] == pytest.approx(0.5)
        assert r["B"]["tpr"] == pytest.approx(2 / 3)
        assert r["C"]["tpr"] == pytest.approx(1.0) and r["C"]["fmr"] == pytest.approx(0.0)
        assert r["A"]["fnmr"] == pytest.approx(0.5)

    def test_empty_denominator_is_nan_not_zero(self):
        # fairlearn reports FPR = 0.0 for group B (no negatives); that would
        # make B look perfectly fair on a rate it has no evidence for.
        assert math.isnan(self._report()["B"]["fmr"])

    def test_equalized_odds_ignores_undefined_groups(self):
        summary = summarise(self._report())
        assert summary["tpr_gap"] == pytest.approx(0.5)
        assert summary["fmr_gap"] == pytest.approx(0.5)  # A vs C only
        assert summary["equalized_odds_difference"] == pytest.approx(0.5)

    def test_integer_codes_match_label_path(self):
        zones = np.where(self.y_pred == 1, Z_MATCH, Z_NO_MATCH)
        codes = np.array([0] * 4 + [1] * 3 + [2] * 4)
        coded, labelled = group_outcome_rates(zones, self.y_true, codes, ["A", "B", "C"]), self._report()
        assert coded.keys() == labelled.keys()
        for label in labelled:
            assert coded[label] == pytest.approx(labelled[label], nan_ok=True)

    def test_abstentions_are_neither_error(self):
        zones = np.array([Z_UNCERTAIN, Z_UNCERTAIN, Z_MATCH, Z_NO_MATCH])
        genuine = np.array([True, False, True, False])
        r = group_outcome_rates(zones, genuine, np.array(["g"] * 4))["g"]
        assert r["fmr"] == 0.0 and r["fnmr"] == 0.0
        assert r["genuine_abstention"] == 0.5 and r["impostor_abstention"] == 0.5

    def test_gap_and_ser_edge_cases(self):
        assert max_gap([0.1, float("nan"), 0.4]) == pytest.approx(0.3)
        assert math.isnan(max_gap([0.1]))
        assert skewed_error_ratio([0.01, 0.04]) == pytest.approx(4.0)
        assert skewed_error_ratio([0.0, 0.0]) == 1.0
        assert math.isinf(skewed_error_ratio([0.0, 0.02]))
        assert math.isnan(skewed_error_ratio([float("nan"), 0.02]))

    def test_fairness_discrepancy_rate(self):
        assert fairness_discrepancy_rate(0.0, 0.0) == 1.0
        assert fairness_discrepancy_rate(0.02, 0.10, alpha=0.5) == pytest.approx(0.94)
        assert math.isnan(fairness_discrepancy_rate(float("nan"), 0.1))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError):
            group_outcome_rates(np.zeros(3, dtype=int), np.zeros(2, dtype=bool), np.zeros(3))


class TestVectorisedPolicyMatchesReference:
    def test_verify_zones_equals_scalar_decide(self):
        rng = np.random.default_rng(3)
        thresholds = Thresholds()
        scores = rng.uniform(0, 1, size=(300, 4))
        # Include exact boundary values, which is where the two could diverge.
        scores[0] = [thresholds.tau_low, thresholds.tau_high, 0.0, 1.0]
        quality = rng.uniform(0, 1, size=300)
        quality[1] = thresholds.q_min
        # Non-finite inputs are exactly where a vectorised copy tends to drift.
        quality[2], quality[3] = float("nan"), float("inf")
        scores[4] = [float("nan"), float("inf"), -float("inf"), 0.9]
        zones = verify_zones(scores, quality, thresholds)
        code = {Zone.MATCH_CONFIRMED: Z_MATCH, Zone.NO_MATCH: Z_NO_MATCH, Zone.UNCERTAIN_REVIEW: Z_UNCERTAIN}
        for i in range(scores.shape[0]):
            for j in range(scores.shape[1]):
                expected = decide(
                    Mode.VERIFY, [Candidate("x", float(scores[i, j]), 0.0)], float(quality[i]), None, thresholds
                )
                assert zones[i, j] == code[expected.zone]

    def test_baseline_zones(self):
        assert baseline_zones(np.array([0.2, 0.35, 0.36]), 0.35).tolist() == [Z_NO_MATCH, Z_MATCH, Z_MATCH]


class TestCohortNormalisation:
    def test_denser_region_gets_discounted(self):
        """Same raw cosine, different neighbourhoods -> different calibrated score."""
        rng = np.random.default_rng(0)
        dim = 64
        centre = np.zeros(dim)
        centre[0] = 1.0
        tight = centre + 0.35 * rng.standard_normal((300, dim)) / math.sqrt(dim)  # crowded region
        sparse = rng.standard_normal((300, dim))  # everything else
        cohort = np.concatenate([tight, sparse])
        calibrator = ScoreCalibrator(
            CalibrationParams(a=1.0, b=0.0, mu0=0.0, sigma0=0.1, top_k=50), cohort
        )
        crowded_probe = centre + 0.35 * rng.standard_normal(dim) / math.sqrt(dim)
        crowded_probe /= np.linalg.norm(crowded_probe)
        isolated_probe = rng.standard_normal(dim)
        isolated_probe /= np.linalg.norm(isolated_probe)
        mu, _ = calibrator.cohort_stats(np.stack([crowded_probe, isolated_probe]))
        assert mu[0] > mu[1] + 0.3  # the crowded face sees much closer impostors

        def template_at(probe, cosine):
            other = rng.standard_normal(dim)
            other -= (other @ probe) * probe
            other /= np.linalg.norm(other)
            return cosine * probe + math.sqrt(1 - cosine**2) * other

        raw = 0.6
        _, s_crowded = calibrator.score_matrix(crowded_probe[None], template_at(crowded_probe, raw)[None])
        _, s_isolated = calibrator.score_matrix(isolated_probe[None], template_at(isolated_probe, raw)[None])
        assert s_crowded[0, 0] < s_isolated[0, 0]

    def test_without_cohort_is_monotone_global(self):
        calibrator = ScoreCalibrator(CalibrationParams(a=2.0, b=-1.0, mu0=0.1, sigma0=0.1))
        assert not calibrator.describe()["equalises_subgroups"]
        z = (np.array([0.1, 0.3, 0.5]) - 0.1) / 0.1
        assert np.allclose(calibrator.to_score(z), 1 / (1 + np.exp(-(2 * z - 1))))

    def test_anchor_solution(self):
        a, b = ScoreCalibrator.solve_logistic(2.0, 6.0, 0.48, 0.78)
        calibrator = ScoreCalibrator(CalibrationParams(a=a, b=b, mu0=0, sigma0=1))
        assert calibrator.to_score(2.0) == pytest.approx(0.48)
        assert calibrator.to_score(6.0) == pytest.approx(0.78)

    def test_refuses_unsupported_tail_quantile(self):
        with pytest.raises(ValueError, match="impostor scores"):
            ScoreCalibrator.impostor_quantile(np.random.default_rng(0).standard_normal(5000), 1e-4)

    def test_worst_group_anchoring_bounds_every_group(self):
        rng = np.random.default_rng(1)
        z = np.concatenate([rng.normal(0.0, 1, 200000), rng.normal(0.6, 1, 200000)])
        groups = np.repeat(["a", "b"], 200000)
        calibrator = ScoreCalibrator.fit(None, z, groups, mu0=0, sigma0=1)
        for label in ("a", "b"):
            fmr = float(np.mean(calibrator.to_score(z[groups == label]) >= 0.78))
            assert fmr <= 1e-4 * 1.5


class TestSubgroupParity:
    def test_runs_fast_enough_for_interactive_use(self, benchmark):
        # CPU time, not wall-clock: the budget is a property of the code and
        # must not fail because the machine happens to be busy.
        import time

        started = time.process_time()
        benchmark.evaluate(Thresholds())
        evaluate_cpu = time.process_time() - started
        assert benchmark.build_cpu_seconds + evaluate_cpu < 10.0

    def test_payload_is_strict_json(self, report):
        import json

        json.dumps(report, allow_nan=False)
        assert report["groups"] == list(FITZPATRICK_GROUPS)
        assert report["kind"] == "synthetic_simulation" and report["disclaimer"]

    def test_baseline_shows_the_disparity_being_fixed(self, report):
        # The simulated population reproduces the real phenomenon: one global
        # threshold gives the densest group several times the FMR of the sparsest.
        groups = report["baseline"]["groups"]
        assert groups["VI"]["fmr"] > 3 * groups["I"]["fmr"]
        assert report["baseline"]["summary"]["ser_fmr"] > 3

    def test_cohort_normalisation_narrows_fmr_gap(self, report):
        with_cohort = report["fair"]["summary"]
        without = report["fair_without_cohort"]["summary"]
        assert with_cohort["fmr_gap"] < 0.5 * without["fmr_gap"]
        assert with_cohort["ser_fmr"] < 0.5 * without["ser_fmr"]

    def test_fair_engine_beats_baseline_at_matched_fmr(self, report):
        matched = report["baseline_matched_fmr"]
        assert matched is not None
        assert matched["summary"]["overall_fmr"] == pytest.approx(report["fair"]["summary"]["overall_fmr"], rel=0.05)
        assert report["fair"]["summary"]["fmr_gap"] < matched["summary"]["fmr_gap"]
        assert report["fair"]["summary"]["ser_fmr"] < matched["summary"]["ser_fmr"]

    def test_every_group_meets_the_high_anchor_fmr_target(self, report):
        target = report["calibration"]["fmr_at_tau_high_anchor"]
        for label, group in report["fair"]["groups"].items():
            assert group["fmr"] <= 3 * target, label

    def test_abstention_cost_is_reported_per_group(self, report):
        # Parity on error rates is bought partly with manual-review load that
        # falls unevenly; the report must expose it rather than hide it.
        summary = report["fair"]["summary"]
        assert summary["genuine_abstention_gap"] > 0
        assert all("genuine_abstention" in g for g in report["fair"]["groups"].values())

    def test_thresholds_change_outcomes(self, benchmark):
        strict = benchmark.evaluate(Thresholds(tau_high=0.95, tau_low=0.48))
        loose = benchmark.evaluate(Thresholds(tau_high=0.60, tau_low=0.48))
        assert strict["fair"]["summary"]["overall_fmr"] < loose["fair"]["summary"]["overall_fmr"]


class TestLookalikeAbstention:
    def test_fair_engine_never_misidentifies_enrolled_lookalikes(self, report):
        fair = report["lookalikes"]["both_enrolled"]["fair"]
        assert fair["wrong_match"] == 0
        assert fair["abstention_reasons"].get("NARROW_CANDIDATE_MARGIN", 0) > 0
        assert fair["correct_match"] + fair["uncertain"] + fair["no_match"] == fair["total"]

    def test_baseline_misidentifies_some(self, report):
        assert report["lookalikes"]["both_enrolled"]["baseline"]["wrong_match"] > 0

    def test_identical_twins_trigger_narrow_margin(self, benchmark):
        result = benchmark.simulate_lookalike(0.92, 0.85, True, Thresholds(), trials=150)
        outcomes = result["fair"]["outcomes"]
        assert outcomes["MATCH_B"] == 0
        assert outcomes["UNCERTAIN_REVIEW"] >= 0.95 * 150
        assert result["fair"]["abstention_reasons"]["NARROW_CANDIDATE_MARGIN"] >= 0.95 * 150
        assert result["baseline"]["outcomes"]["MATCH_B"] > 0

    def test_distinct_people_still_match(self, benchmark):
        result = benchmark.simulate_lookalike(0.05, 0.85, True, Thresholds(), trials=150)
        assert result["fair"]["outcomes"]["MATCH_A"] >= 0.9 * 150
        assert result["fair"]["outcomes"]["MATCH_B"] == 0

    def test_degraded_probe_abstains_for_quality(self, benchmark):
        result = benchmark.simulate_lookalike(0.5, 0.15, True, Thresholds(), trials=100)
        assert result["fair"]["outcomes"]["UNCERTAIN_REVIEW"] == 100
        assert result["fair"]["abstention_reasons"] == {"LOW_IMAGE_QUALITY": 100}

    def test_open_set_limitation_is_not_hidden(self, benchmark):
        # When only the lookalike is enrolled there is no second candidate, so
        # the margin rule cannot fire. The report must show those errors.
        result = benchmark.simulate_lookalike(0.92, 0.85, False, Thresholds(), trials=100)
        assert result["fair"]["outcomes"]["MATCH_B"] > 0
        assert result["fair"]["false_identification_rate"] > 0

    def test_input_validation(self, benchmark):
        with pytest.raises(ValueError):
            benchmark.simulate_lookalike(1.0, 0.5, True, Thresholds())
        with pytest.raises(ValueError):
            benchmark.simulate_lookalike(0.5, 0.5, True, Thresholds(), group="VII")


@pytest.fixture(scope="module")
def rfw():
    data = load_rfw_scores()
    if data is None:
        pytest.skip("RFW score files are not available in this checkout")
    return data


class TestRealScoresRFW:

    def test_join_drops_unlabelled_pairs(self, rfw):
        assert rfw["cosine"].size == 23923
        assert set(np.unique(rfw["groups"])) == {"African", "Asian", "Caucasian", "Indian"}

    def test_single_threshold_disparity_on_real_data(self, rfw):
        report = rfw_report(rfw, target_fmr=1e-2)
        groups = report["groups"]
        assert report["summary"]["overall_fmr"] == pytest.approx(0.01, abs=2e-4)
        # Independently computed reference values for this score file.
        assert groups["African"]["fmr"] == pytest.approx(0.0248, abs=1e-3)
        assert groups["Caucasian"]["fmr"] == pytest.approx(0.0030, abs=5e-4)
        assert report["summary"]["ser_fmr"] > 7
        assert report["summary"]["equalized_odds_difference"] == pytest.approx(0.101, abs=5e-3)

    def test_oracle_thresholds_equalise_fmr_but_not_fnmr(self, rfw):
        oracle = rfw_report(rfw, target_fmr=1e-2)["oracle_per_group_thresholds"]
        assert max_gap([g["fmr"] for g in oracle.values()]) < 1e-3
        assert max_gap([g["fnmr"] for g in oracle.values()]) > 0.05

    def test_missing_files_return_none(self, tmp_path):
        assert load_rfw_scores(tmp_path) is None

    def test_rfw_dir_can_be_configured(self, tmp_path, monkeypatch):
        from backend.fairness.audit import default_rfw_dir

        monkeypatch.setenv("FAIRFACE_RFW_DIR", str(tmp_path))
        assert default_rfw_dir() == tmp_path
        monkeypatch.setenv("FAIRFACE_RFW_DIR", str(tmp_path / "absent"))
        assert load_rfw_scores() is None
