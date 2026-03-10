"""
Causal Inference Methods Module
Propensity score matching, difference-in-differences, instrumental variables,
regression discontinuity, and synthetic control.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class PropensityScoreMatching:
    """Propensity score matching for causal effect estimation."""

    def __init__(self, n_neighbors: int = 1):
        self.n_neighbors = n_neighbors
        self.propensity_scores = None

    def estimate_propensity(self, X: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        """
        Estimate propensity scores using logistic regression (simple gradient descent).
        """
        n, d = X.shape
        # Add intercept
        X_int = np.column_stack([np.ones(n), X])
        beta = np.zeros(d + 1)

        lr = 0.01
        for _ in range(1000):
            z = X_int @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
            grad = X_int.T @ (p - treatment) / n
            beta -= lr * grad

        self.propensity_scores = 1.0 / (1.0 + np.exp(-np.clip(X_int @ beta, -500, 500)))
        return self.propensity_scores

    def match(self, treatment: np.ndarray) -> List[Tuple[int, int]]:
        """Match treated units to control units based on propensity scores."""
        if self.propensity_scores is None:
            raise RuntimeError("Must estimate propensity scores first.")

        treated_idx = np.where(treatment == 1)[0]
        control_idx = np.where(treatment == 0)[0]

        matches = []
        for t_idx in treated_idx:
            t_score = self.propensity_scores[t_idx]
            distances = np.abs(self.propensity_scores[control_idx] - t_score)
            nearest = control_idx[np.argsort(distances)[:self.n_neighbors]]
            for c_idx in nearest:
                matches.append((t_idx, c_idx))

        return matches

    def estimate_ate(self, outcome: np.ndarray, treatment: np.ndarray,
                     X: np.ndarray) -> Dict:
        """Estimate Average Treatment Effect using matching."""
        self.estimate_propensity(X, treatment)
        matches = self.match(treatment)

        if not matches:
            return {"ate": 0.0, "n_matches": 0}

        effects = []
        for t_idx, c_idx in matches:
            effects.append(outcome[t_idx] - outcome[c_idx])

        ate = float(np.mean(effects))
        se = float(np.std(effects) / np.sqrt(len(effects))) if len(effects) > 1 else 0.0

        return {
            "ate": round(ate, 6),
            "se": round(se, 6),
            "n_matches": len(matches),
            "ci_lower": round(ate - 1.96 * se, 6),
            "ci_upper": round(ate + 1.96 * se, 6),
        }


class DifferenceInDifferences:
    """Difference-in-Differences estimator."""

    def estimate(self, outcome_treated_pre: np.ndarray, outcome_treated_post: np.ndarray,
                 outcome_control_pre: np.ndarray, outcome_control_post: np.ndarray) -> Dict:
        """
        Estimate treatment effect using DiD.

        Args:
            outcome_treated_pre: Outcomes for treated group before treatment.
            outcome_treated_post: Outcomes for treated group after treatment.
            outcome_control_pre: Outcomes for control group before treatment.
            outcome_control_post: Outcomes for control group after treatment.
        """
        mean_t_pre = np.mean(outcome_treated_pre)
        mean_t_post = np.mean(outcome_treated_post)
        mean_c_pre = np.mean(outcome_control_pre)
        mean_c_post = np.mean(outcome_control_post)

        did = (mean_t_post - mean_t_pre) - (mean_c_post - mean_c_pre)

        # Standard error using pooled variance
        n_t = len(outcome_treated_pre)
        n_c = len(outcome_control_pre)
        var_t = (np.var(outcome_treated_post - outcome_treated_pre[:len(outcome_treated_post)]) / n_t
                 if n_t > 0 else 0)
        var_c = (np.var(outcome_control_post - outcome_control_pre[:len(outcome_control_post)]) / n_c
                 if n_c > 0 else 0)
        se = float(np.sqrt(var_t + var_c))

        return {
            "did_estimate": round(float(did), 6),
            "se": round(se, 6),
            "treated_change": round(float(mean_t_post - mean_t_pre), 6),
            "control_change": round(float(mean_c_post - mean_c_pre), 6),
            "ci_lower": round(float(did - 1.96 * se), 6),
            "ci_upper": round(float(did + 1.96 * se), 6),
        }


class InstrumentalVariables:
    """Two-stage least squares (2SLS) instrumental variable estimator."""

    def estimate(self, outcome: np.ndarray, treatment: np.ndarray,
                 instrument: np.ndarray,
                 covariates: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate causal effect using instrumental variables.

        Args:
            outcome: Outcome variable (Y).
            treatment: Endogenous treatment variable (D).
            instrument: Instrumental variable (Z).
            covariates: Optional control variables.
        """
        n = len(outcome)

        # Build design matrix
        if covariates is not None:
            Z = np.column_stack([np.ones(n), instrument, covariates])
        else:
            Z = np.column_stack([np.ones(n), instrument])

        # Stage 1: Regress treatment on instrument
        beta_first = np.linalg.lstsq(Z, treatment, rcond=None)[0]
        treatment_hat = Z @ beta_first

        # Stage 2: Regress outcome on predicted treatment
        if covariates is not None:
            X_second = np.column_stack([np.ones(n), treatment_hat, covariates])
        else:
            X_second = np.column_stack([np.ones(n), treatment_hat])

        beta_second = np.linalg.lstsq(X_second, outcome, rcond=None)[0]
        iv_estimate = float(beta_second[1])

        # Residuals and SE
        residuals = outcome - X_second @ beta_second
        mse = float(np.sum(residuals ** 2) / (n - len(beta_second)))
        xtx_inv = np.linalg.inv(X_second.T @ X_second)
        se = float(np.sqrt(mse * xtx_inv[1, 1]))

        # First-stage F-statistic
        ss_total = np.sum((treatment - np.mean(treatment)) ** 2)
        ss_resid = np.sum((treatment - treatment_hat) ** 2)
        f_stat = float(((ss_total - ss_resid) / 1) / (ss_resid / (n - Z.shape[1])))

        return {
            "iv_estimate": round(iv_estimate, 6),
            "se": round(se, 6),
            "first_stage_f": round(f_stat, 4),
            "ci_lower": round(iv_estimate - 1.96 * se, 6),
            "ci_upper": round(iv_estimate + 1.96 * se, 6),
        }


class RegressionDiscontinuity:
    """Regression Discontinuity Design estimator."""

    def estimate(self, outcome: np.ndarray, running_var: np.ndarray,
                 cutoff: float, bandwidth: Optional[float] = None) -> Dict:
        """
        Estimate treatment effect at the cutoff.

        Args:
            outcome: Outcome variable.
            running_var: Running/forcing variable.
            cutoff: Cutoff point for treatment assignment.
            bandwidth: Window around cutoff (default: IQR/2).
        """
        if bandwidth is None:
            iqr = np.percentile(running_var, 75) - np.percentile(running_var, 25)
            bandwidth = iqr / 2

        # Select observations within bandwidth
        mask = np.abs(running_var - cutoff) <= bandwidth
        y_local = outcome[mask]
        x_local = running_var[mask]

        # Separate left and right of cutoff
        left_mask = x_local < cutoff
        right_mask = x_local >= cutoff

        if np.sum(left_mask) < 2 or np.sum(right_mask) < 2:
            return {"rd_estimate": 0.0, "se": 0.0, "n_left": 0, "n_right": 0}

        # Local linear regression on each side
        def local_linear(x, y):
            n = len(x)
            X = np.column_stack([np.ones(n), x - cutoff])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return beta

        beta_left = local_linear(x_local[left_mask], y_local[left_mask])
        beta_right = local_linear(x_local[right_mask], y_local[right_mask])

        # RD estimate = difference in intercepts
        rd_estimate = float(beta_right[0] - beta_left[0])

        # Standard errors
        resid_left = y_local[left_mask] - np.column_stack(
            [np.ones(np.sum(left_mask)), x_local[left_mask] - cutoff]
        ) @ beta_left
        resid_right = y_local[right_mask] - np.column_stack(
            [np.ones(np.sum(right_mask)), x_local[right_mask] - cutoff]
        ) @ beta_right

        var_left = np.var(resid_left) / np.sum(left_mask)
        var_right = np.var(resid_right) / np.sum(right_mask)
        se = float(np.sqrt(var_left + var_right))

        return {
            "rd_estimate": round(rd_estimate, 6),
            "se": round(se, 6),
            "bandwidth": round(bandwidth, 4),
            "n_left": int(np.sum(left_mask)),
            "n_right": int(np.sum(right_mask)),
            "ci_lower": round(rd_estimate - 1.96 * se, 6),
            "ci_upper": round(rd_estimate + 1.96 * se, 6),
        }


class SyntheticControl:
    """Synthetic Control Method for comparative case studies."""

    def estimate(self, treated_outcome: np.ndarray,
                 control_outcomes: np.ndarray,
                 pre_periods: int) -> Dict:
        """
        Estimate treatment effect using synthetic control.

        Args:
            treated_outcome: Outcome series for treated unit (T,).
            control_outcomes: Outcome matrix for control units (T, J).
            pre_periods: Number of pre-treatment periods.
        """
        T, J = control_outcomes.shape

        # Find weights by minimizing pre-treatment distance
        y_pre = treated_outcome[:pre_periods]
        X_pre = control_outcomes[:pre_periods, :]

        # Constrained optimization: non-negative weights summing to 1
        # Using iterative reweighting
        weights = np.ones(J) / J

        for _ in range(200):
            synthetic_pre = X_pre @ weights
            errors = y_pre - synthetic_pre

            # Update weights proportional to correlation
            for j in range(J):
                corr = np.dot(errors, X_pre[:, j])
                weights[j] *= np.exp(0.01 * corr)

            # Normalize
            weights = np.maximum(weights, 0)
            w_sum = weights.sum()
            if w_sum > 0:
                weights /= w_sum

        # Construct synthetic control
        synthetic = control_outcomes @ weights

        # Treatment effect = treated - synthetic for post-treatment periods
        effects = treated_outcome[pre_periods:] - synthetic[pre_periods:]
        ate = float(np.mean(effects)) if len(effects) > 0 else 0.0

        # Pre-treatment fit
        pre_rmse = float(np.sqrt(np.mean((treated_outcome[:pre_periods] - synthetic[:pre_periods]) ** 2)))

        return {
            "ate": round(ate, 6),
            "weights": weights.tolist(),
            "synthetic_series": synthetic.tolist(),
            "effects": effects.tolist(),
            "pre_treatment_rmse": round(pre_rmse, 6),
            "n_post_periods": len(effects),
        }
