"""Genie-style synthetic multi-event likelihoods for adaptive-prior development.

Recipe (after gwkokab's 'genie', per RO): for each synthetic BNS event, use a
Fisher-like width to set the measurement covariance, draw an offset from it to
displace the observed peak from the truth, and use the quadratic peak as the
log-likelihood.  The per-EOS evidence of one event is then

    lnZ_e^(i) = -(LambdaTilde_e(m1_i, m2_i) - x_obs_i)^2 / (2 sigma_i^2)

(up to constants), and **multiple events multiply**: lnZ_e = sum_i lnZ_e^(i).
This gives a zero-cost stand-in for the per-event CIP evidences -- the multi-event
scaling of the exact ("painful") mode is n_events x n_EOS full CIP runs per
generation, so synthetic likelihoods are how the adaptive loop is developed and
its convergence characterized before burning core-hours.

The observable is the leading tidal combination
LambdaTilde(m1, m2, Lambda1, Lambda2) (Favata/Wade convention).
"""
from __future__ import annotations

import numpy as np

from ..priors.gp_pca import sequence_lambda_of_m


def lambda_tilde(m1, m2, lam1, lam2):
    """Leading-order effective tidal deformability (dimensionless)."""
    mt = m1 + m2
    return (16.0 / 13.0) * ((m1 + 12.0 * m2) * m1 ** 4 * lam1
                            + (m2 + 12.0 * m1) * m2 ** 4 * lam2) / mt ** 5


def draw_bns_events(rng, n_events, truth_lambda_of_m, m_mean=1.35, m_sigma=0.06,
                    sigma_lamtilde=80.0, sigma_scatter=0.3):
    """Draw a synthetic BNS event set with genie-style observed peaks.

    Masses ~ N(m_mean, m_sigma) (sorted so m1 >= m2); each event's measurement
    width sigma_i = sigma_lamtilde * exp(N(0, sigma_scatter)) (an SNR-like
    spread); observed peak = truth + N(0, sigma_i) offset.  Returns a list of
    dicts (m1, m2, sigma, x_obs, x_true).
    """
    events = []
    for _ in range(n_events):
        m = np.sort(rng.normal(m_mean, m_sigma, size=2))[::-1]
        lam1 = truth_lambda_of_m(m[0])
        lam2 = truth_lambda_of_m(m[1])
        if not (np.isfinite(lam1) and np.isfinite(lam2)):
            continue
        x_true = lambda_tilde(m[0], m[1], lam1, lam2)
        sig = sigma_lamtilde * np.exp(rng.normal(0.0, sigma_scatter))
        events.append(dict(m1=float(m[0]), m2=float(m[1]), sigma=float(sig),
                           x_true=float(x_true),
                           x_obs=float(x_true + rng.normal(0.0, sig))))
    return events


class SyntheticTidalEvents:
    """``evidence_fn(prior, coeffs) -> lnZ`` for a fixed synthetic event set.

    Per EOS draw e: evaluate Lambda(m1_i), Lambda(m2_i) from the reconstructed
    sequence, form LambdaTilde, and sum the quadratic-peak log-likelihoods over
    events.  Draws whose stable branch does not cover an event's masses get a
    heavy penalty (they cannot produce the observed system).
    """

    def __init__(self, events, off_branch_penalty=-50.0):
        self.events = list(events)
        self.off_branch_penalty = float(off_branch_penalty)

    def __call__(self, prior, coeffs):
        out = np.zeros(len(coeffs))
        for k, c in enumerate(coeffs):
            tot = 0.0
            for ev in self.events:
                l1 = sequence_lambda_of_m(prior, c, ev["m1"])
                l2 = sequence_lambda_of_m(prior, c, ev["m2"])
                if not (np.isfinite(l1) and np.isfinite(l2)):
                    tot += self.off_branch_penalty
                    continue
                x = lambda_tilde(ev["m1"], ev["m2"], l1, l2)
                tot += -0.5 * ((x - ev["x_obs"]) / ev["sigma"]) ** 2
            out[k] = tot
        return out
