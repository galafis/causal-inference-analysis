"""
Tests for the Causal Inference Analysis module.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causal_methods import (
    PropensityScoreMatching, DifferenceInDifferences,
    InstrumentalVariables, RegressionDiscontinuity, SyntheticControl
)


class TestPropensityScoreMatching:
    def test_estimate_propensity(self):
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 2)
        treatment = (X[:, 0] + np.random.randn(n) * 0.5 > 0).astype(float)
        psm = PropensityScoreMatching()
        scores = psm.estimate_propensity(X, treatment)
        assert scores.shape == (n,)
        assert np.all(scores >= 0) and np.all(scores <= 1)

    def test_matching(self):
        np.random.seed(42)
        n = 50
        X = np.random.randn(n, 1)
        treatment = (X[:, 0] > 0).astype(float)
        psm = PropensityScoreMatching(n_neighbors=1)
        psm.estimate_propensity(X, treatment)
        matches = psm.match(treatment)
        assert len(matches) > 0

    def test_ate_estimation(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 2)
        treatment = (X[:, 0] > 0).astype(float)
        outcome = 2.0 * treatment + X[:, 0] + np.random.randn(n) * 0.5
        psm = PropensityScoreMatching()
        result = psm.estimate_ate(outcome, treatment, X)
        assert "ate" in result
        assert result["n_matches"] > 0
        # ATE should be approximately 2.0
        assert abs(result["ate"] - 2.0) < 2.0


class TestDifferenceInDifferences:
    def test_did_estimate(self):
        np.random.seed(42)
        n = 50
        treated_pre = np.random.randn(n) + 5
        treated_post = np.random.randn(n) + 8  # effect of 3
        control_pre = np.random.randn(n) + 5
        control_post = np.random.randn(n) + 6  # natural increase of 1

        did = DifferenceInDifferences()
        result = did.estimate(treated_pre, treated_post, control_pre, control_post)
        assert "did_estimate" in result
        # DiD should be approximately 2 (3-1)
        assert abs(result["did_estimate"] - 2.0) < 1.5

    def test_no_effect(self):
        np.random.seed(42)
        n = 100
        pre = np.random.randn(n)
        post = np.random.randn(n) + 1
        did = DifferenceInDifferences()
        result = did.estimate(pre, post, pre, post)
        assert abs(result["did_estimate"]) < 0.5


class TestInstrumentalVariables:
    def test_iv_estimate(self):
        np.random.seed(42)
        n = 500
        instrument = np.random.randn(n)
        treatment = 0.8 * instrument + np.random.randn(n) * 0.3
        outcome = 3.0 * treatment + np.random.randn(n) * 0.5

        iv = InstrumentalVariables()
        result = iv.estimate(outcome, treatment, instrument)
        assert "iv_estimate" in result
        assert result["first_stage_f"] > 5  # Strong instrument
        assert abs(result["iv_estimate"] - 3.0) < 1.5

    def test_with_covariates(self):
        np.random.seed(42)
        n = 300
        covariates = np.random.randn(n, 1)
        instrument = np.random.randn(n)
        treatment = 0.7 * instrument + covariates[:, 0] + np.random.randn(n) * 0.3
        outcome = 2.0 * treatment + covariates[:, 0] + np.random.randn(n) * 0.5

        iv = InstrumentalVariables()
        result = iv.estimate(outcome, treatment, instrument, covariates)
        assert "iv_estimate" in result


class TestRegressionDiscontinuity:
    def test_rd_estimate(self):
        np.random.seed(42)
        n = 400
        running = np.random.uniform(-5, 5, n)
        treatment = (running >= 0).astype(float)
        outcome = 1.0 + 0.5 * running + 3.0 * treatment + np.random.randn(n) * 0.5

        rd = RegressionDiscontinuity()
        result = rd.estimate(outcome, running, cutoff=0.0, bandwidth=2.0)
        assert "rd_estimate" in result
        assert result["n_left"] > 0
        assert result["n_right"] > 0
        assert abs(result["rd_estimate"] - 3.0) < 2.0

    def test_custom_bandwidth(self):
        np.random.seed(42)
        n = 200
        running = np.random.uniform(-3, 3, n)
        outcome = 1.0 + 2.0 * (running >= 0).astype(float) + np.random.randn(n) * 0.3

        rd = RegressionDiscontinuity()
        result = rd.estimate(outcome, running, cutoff=0.0, bandwidth=1.5)
        assert result["bandwidth"] == 1.5


class TestSyntheticControl:
    def test_synthetic_control(self):
        np.random.seed(42)
        T = 20
        pre = 10
        J = 5

        # Create treated unit with an effect of 5 starting at period 10
        treated = np.random.randn(T) + 10
        treated[pre:] += 5  # treatment effect

        # Control units (no treatment)
        controls = np.random.randn(T, J) + 10

        sc = SyntheticControl()
        result = sc.estimate(treated, controls, pre_periods=pre)

        assert "ate" in result
        assert "weights" in result
        assert len(result["weights"]) == J
        assert result["n_post_periods"] == T - pre
        # ATE should be approximately 5
        assert result["ate"] > 0

    def test_weights_sum(self):
        np.random.seed(42)
        treated = np.random.randn(10) + 5
        controls = np.random.randn(10, 3) + 5
        sc = SyntheticControl()
        result = sc.estimate(treated, controls, pre_periods=5)
        assert abs(sum(result["weights"]) - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
