import numpy as np
from scipy.stats import skewnorm

from nmbackend.inference.cooling_population import (
    CoolingSource, IsoPopulation, age_weights, log_like_grid, posterior_weights,
    source_mass_kernels, star_mass_posterior, thermal_loglike, _trapz)

LOG_T = np.linspace(2, 6.5, 181)
ETA = np.array([0.0, 0.3])          # toy envelope offsets in log L


def toy_grid(m_du, mmax):
    """logL[M, g, eta, t]: slow track, 1 dex fainter above m_du; g=1 halves the drop."""
    M = np.sort(np.r_[np.linspace(1.0, mmax, 12), m_du - 1e-3, m_du + 1e-3])
    M = M[(M >= 1.0) & (M <= mmax)]
    slow = 33.6 - 0.35 * (LOG_T - 3)
    drop = np.array([1.0, 0.5])[None, :, None, None] * (M > m_du)[:, None, None, None]
    return M, slow[None, None, None, :] + ETA[None, None, :, None] - drop


def test_kernel_matches_monte_carlo():
    rng = np.random.default_rng(1)
    src = CoolingSource("s", dict(form="box_log10t", lo=3.8, hi=4.2),
                        dict(form="split_normal_log10L", mode=33.1, sigma_lo=0.2, sigma_hi=0.2),
                        sigma_age_edge=0.05)
    M, L = toy_grid(1.5, 2.1)
    m_fine = np.linspace(1.0, 2.1, 801)
    S = source_mass_kernels(L, M, LOG_T, [src], m_fine)[0]          # [g, m]
    # brute force at one mass below onset, gap 0: t ~ p_birth * box, eta uniform
    lt = rng.uniform(3.6, 4.4, 400000)
    from scipy.special import erf
    box = 0.5 * (erf((lt - 3.8) / (np.sqrt(2) * 0.05)) - erf((lt - 4.2) / (np.sqrt(2) * 0.05)))
    # box_log10t is a density in log10 t: uniform draws in lt need no t factor
    e = rng.integers(0, 2, lt.size)
    model = 33.6 - 0.35 * (lt - 3) + ETA[e]
    like = np.exp(thermal_loglike(model, src))
    mc = np.sum(like * box) / np.sum(box)
    j = np.searchsorted(m_fine, 1.3)
    assert abs(S[0, j] / mc - 1) < 0.01


def test_age_weights_conventions():
    th = dict(form="upper_limit", log10_L_max=33.0)
    inside = (LOG_T > 4.05) & (LOG_T < 4.95)
    w = age_weights(LOG_T, CoolingSource("s", dict(form="uniform_t", lo=1e4, hi=1e5), th,
                                         sigma_age_edge=1e-3))
    assert 6 < w[inside][-1] / w[inside][0] < 10.5          # uniform in t: grows as t per log bin
    w = age_weights(LOG_T, CoolingSource("s", dict(form="loguniform_t", lo=1e4, hi=1e5), th,
                                         sigma_age_edge=1e-3))
    assert abs(w[inside][-1] / w[inside][0] - 1) < 1e-6


def _synthetic_catalog(rng, m_du, pop_true, n=30):
    M_true = skewnorm.rvs(pop_true[2], loc=pop_true[0], scale=pop_true[1], size=n, random_state=rng)
    M_true = np.clip(M_true, 1.01, 2.05)
    srcs = []
    for m in M_true:
        lt = rng.uniform(3.3, 5.7)
        e = rng.integers(0, 2)
        L = 33.6 - 0.35 * (lt - 3) + ETA[e] - 1.0 * (m > m_du) + rng.normal(0, 0.1)
        srcs.append(CoolingSource("x", dict(form="box_log10t", lo=lt - 0.1, hi=lt + 0.1),
                                  dict(form="split_normal_log10L", mode=L, sigma_lo=0.15,
                                       sigma_hi=0.15)))
    return srcs


def test_recovers_onset_mass_and_population():
    rng = np.random.default_rng(7)
    m_du_true, mmax = 1.55, 2.1
    srcs = _synthetic_catalog(rng, m_du_true, (1.35, 0.25, 2.0))
    draws = np.linspace(1.15, 2.05, 19)                  # EOS draws labelled by M_DU
    m_fine = np.linspace(1.0, 2.2, 481)
    S = np.stack([source_mass_kernels(*toy_grid(d, mmax)[::-1], LOG_T, srcs, m_fine)
                  for d in draws])                        # [e, i, g, m]
    k = 300
    pop = IsoPopulation(mu=rng.uniform(1.1, 1.7, k), sigma=rng.uniform(0.05, 0.4, k),
                        alpha=rng.uniform(-3, 5, k), f_ecsn=np.zeros(k))
    lnL = log_like_grid(S, np.full(len(draws), mmax), pop, m_fine)
    w, lz, ess = posterior_weights(lnL)
    p_e = w.sum(axis=(0, 2))
    assert abs(draws[np.argmax(p_e)] - m_du_true) < 0.1
    # gap model 0 (full drop) is the truth
    assert w.sum(axis=(0, 1))[0] > 0.7
    mu_post = np.sum(w.sum(axis=(1, 2)) * pop.mu)
    assert 1.2 < mu_post < 1.6
    post = star_mass_posterior(w, S[:, 0], np.full(len(draws), mmax), pop, m_fine)
    assert abs(_trapz(post, m_fine) - 1) < 1e-3


def test_split_at_onset_blocks_interpolation():
    from nmbackend.inference.cooling_population import split_at_onset, _interp_mass
    M = np.array([1.0, 1.2, 1.4, 1.6])
    L = np.array([33.0, 33.0, 30.0, 30.0])[:, None]
    nd = np.array([0, 0, 1, 2])
    M2, L2 = split_at_onset(M, nd, L)
    v = _interp_mass(L2, M2, np.array([1.25, 1.29, 1.31, 1.35]))[:, 0]
    assert np.allclose(v, [33, 33, 30, 30])
    M3, L3 = split_at_onset(M, np.zeros(4), L)
    assert M3 is not None and len(M3) == 4


def test_mixture_weights_respected():
    """Regression: component normalization must not cancel the ECSN share."""
    from nmbackend.inference.cooling_population import mixture_density
    m = np.linspace(0.95, 2.7, 2001)
    for fe in (0.0, 0.07, 0.5):
        pop = IsoPopulation(mu=np.array([1.5]), sigma=np.array([0.2]), alpha=np.array([0.0]),
                            f_ecsn=np.array([fe]))
        p = mixture_density(pop, m, 2.2)[0]
        assert abs(_trapz(p, m) - 1) < 2e-3
        spike = (m > 1.2) & (m < 1.3)
        base = mixture_density(IsoPopulation(np.array([1.5]), np.array([0.2]), np.array([0.0]),
                                             np.array([0.0])), m, 2.2)[0]
        assert abs(_trapz(p[spike] - (1 - fe) * base[spike], m[spike]) - fe) < 5e-3
    # likelihood of a star measured at 1.25 must grow with the ECSN share
    S = np.exp(-0.5 * ((m - 1.25) / 0.02) ** 2)[None, None, None, :]
    lo = log_like_grid(S, np.array([2.2]), IsoPopulation(np.array([1.6]), np.array([0.15]),
                                                         np.array([0.]), np.array([0.05])), m)
    hi = log_like_grid(S, np.array([2.2]), IsoPopulation(np.array([1.6]), np.array([0.15]),
                                                         np.array([0.]), np.array([0.5])), m)
    assert hi[0, 0, 0] > lo[0, 0, 0] + 1.0
