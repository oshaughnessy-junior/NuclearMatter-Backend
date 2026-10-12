"""Hierarchical cooling likelihood over EOS draws, gap models and a mass population.

For cooling source i with thermal datum d_i and age datum a_i,

    L_i(e, g, Lam) = int dM p(M | Lam, Mmax_e) int deta p(eta) int dt p(a_i | t) p_birth(t)
                     p(d_i | log10 L_gamma^inf(t; M, e, g, eta))

on a precomputed curve grid logL[M, g, eta, t] per EOS draw e (paper 7,
analyses/inference/grid).  Each star's mass, envelope and age are marginalized;
the population enters only through p(M | Lam).  Selection is a constant (clean
subsample, nmb-papers paper7 SELECTION-SCOPE option O3) unless supplied.

Pipeline:
  source_mass_kernels  -> S_i[e, g, m]  on a common fine mass grid (zero above Mmax_e)
  population_density   -> F[k, m]       unnormalized p(M | Lam_k); Z[k, e] by truncation
  log_like_grid        -> lnL[k, e, g] = sum_i ln(F @ S_i) - N ln Z
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.special import erf, log_ndtr, logsumexp
from scipy.stats import norm

__all__ = ["CoolingSource", "age_weights", "thermal_loglike", "source_mass_kernels",
           "IsoPopulation", "population_density", "log_like_grid", "posterior_weights",
           "star_mass_posterior"]

M_ECSN, SIG_ECSN = 1.25, 0.01
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class CoolingSource:
    """One source; ages and luminosities in log10 (yr, erg/s).

    age: dict with "form" one of
      split_normal_log10t (mode_log10, sigma_lo, sigma_hi), loguniform_t (lo, hi),
      uniform_t (lo, hi), box_log10t (lo, hi in log10); combined with a uniform
      birth rate (weight proportional to t) and soft edges of width sigma_age_edge dex.
    thermal: dict with "form" one of
      split_normal_log10L (mode, sigma_lo, sigma_hi), uniform_log10L (lo, hi; soft edges),
      upper_limit (log10_L_max; P(L < max) with width sigma_lim).
    sigma_sys (dex) is added in quadrature to every thermal width (distance, atmosphere).
    """
    name: str
    age: dict
    thermal: dict
    sigma_sys: float = 0.0
    sigma_lim: float = 0.1
    sigma_age_edge: float = 0.03
    meta: dict = field(default_factory=dict)

    @property
    def kind(self):
        return "upper_limit" if self.thermal["form"] == "upper_limit" else "detection"

    @classmethod
    def from_record(cls, r, **kw):
        """From a nmb-papers paper7 analyses/inference/data/sources.json record."""
        return cls(r["name"], dict(r["age"]["likelihood"]), dict(r["thermal"]["likelihood"]),
                   meta=r, **kw)


def _soft_box(x, lo, hi, s):
    return 0.5 * (erf((x - lo) / (np.sqrt(2) * s)) - erf((x - hi) / (np.sqrt(2) * s)))


def _split_normal_ln(x, mode, slo, shi):
    s = np.where(x < mode, slo, shi)
    return -0.5 * ((x - mode) / s) ** 2 - np.log(np.sqrt(2 * np.pi) * 0.5 * (slo + shi))


def age_weights(log_t, src: CoolingSource) -> np.ndarray:
    """Normalized age weights per log10 t node.

    The tabulated age is read as the age posterior under a uniform birth rate,
    i.e. p(a | t) p_birth(t) dt is the stated density: split normal or uniform in
    log10 t, or uniform in t ("uniform_t", historical ages).
    """
    a, f, s = src.age, src.age["form"], src.sigma_age_edge
    if f == "split_normal_log10t":
        pa = np.exp(_split_normal_ln(log_t, a["mode_log10"], np.hypot(a["sigma_lo"], s),
                                     np.hypot(a["sigma_hi"], s)))
    elif f in ("loguniform_t", "uniform_t"):
        pa = _soft_box(log_t, np.log10(a["lo"]), np.log10(a["hi"]), s)
        if f == "uniform_t":
            pa = pa * 10.0 ** log_t
    elif f == "box_log10t":
        pa = _soft_box(log_t, a["lo"], a["hi"], s)
    else:
        raise ValueError(f)
    w = pa * np.gradient(log_t)
    return w / w.sum()


def thermal_loglike(logL, src: CoolingSource, sigma_model=0.0):
    """ln p(d_i | model log10 L) elementwise; sigma_model broadcasts against logL."""
    th, f = src.thermal, src.thermal["form"]
    extra = np.square(src.sigma_sys) + np.square(sigma_model)
    if f == "split_normal_log10L":
        slo, shi = np.sqrt(th["sigma_lo"] ** 2 + extra), np.sqrt(th["sigma_hi"] ** 2 + extra)
        return _split_normal_ln(logL, th["mode"], slo, shi)
    if f == "uniform_log10L":
        s = np.sqrt(0.03 ** 2 + extra)
        with np.errstate(divide="ignore"):
            return np.log(np.maximum(_soft_box(logL, th["lo"], th["hi"], s), 1e-300)) \
                - np.log(th["hi"] - th["lo"])
    if f == "upper_limit":
        s = np.sqrt(src.sigma_lim ** 2 + extra)
        return log_ndtr((th["log10_L_max"] - logL) / s)
    raise ValueError(f)


def _interp_mass(logL, M_knots, m_fine):
    """Linear interpolation of logL[M, ...] onto m_fine (values above the last knot: nan)."""
    idx = np.clip(np.searchsorted(M_knots, m_fine) - 1, 0, len(M_knots) - 2)
    w = (m_fine - M_knots[idx]) / (M_knots[idx + 1] - M_knots[idx])
    out = (1 - w)[(...,) + (None,) * (logL.ndim - 1)] * logL[idx] \
        + w[(...,) + (None,) * (logL.ndim - 1)] * logL[idx + 1]
    out[(m_fine < M_knots[0]) | (m_fine > M_knots[-1])] = np.nan
    return out


def source_mass_kernels(logL, M_knots, log_t, sources: Sequence[CoolingSource], m_fine,
                        eta_weights=None, sigma_model=None, valid=None):
    """S[i, g, m] = sum_eta w_eta sum_t w_i(t) p(d_i | logL) for one EOS draw.

    logL: [nM, nG, nEta, nT]; M_knots ascending (top knot ~ Mmax of the draw);
    sigma_model: None, scalar, array over log_t, or array shaped like logL.  Mass points outside the knots get 0.
    valid: optional bool [nM, nG, nEta]; invalid curves are excluded from the eta average
    (renormalized) and a mass point with no valid curve gets 0.
    """
    nM, nG, nE, nT = logL.shape
    ew = np.full(nE, 1.0 / nE) if eta_weights is None else np.asarray(eta_weights, float)
    finite = np.isfinite(logL).all(axis=-1)                    # [M, G, E]
    ok = _interp_mass(finite.astype(float), M_knots, m_fine) > 0.999
    if valid is not None:
        ok &= _interp_mass(valid.astype(float), M_knots, m_fine) > 0.999
    ok &= np.isfinite(m_fine)[:, None, None]
    wE = ew[None, None, :] * ok                               # [m, G, E]
    norm_e = wE.sum(-1)
    L0 = np.where(np.isfinite(logL), logL, 0.0)
    full_sm = sigma_model is not None and np.ndim(sigma_model) == 4
    out = np.zeros((len(sources), len(m_fine), nG))
    for i, src in enumerate(sources):
        aw = age_weights(log_t, src)
        cols = np.nonzero(aw > 1e-5 * aw.max())[0]           # age support (weight > 1e-5 peak)
        lf = np.nan_to_num(_interp_mass(L0[..., cols], M_knots, m_fine))   # [m, G, E, t]
        if sigma_model is None:
            sm = 0.0
        elif full_sm:
            sm = np.nan_to_num(_interp_mass(np.asarray(sigma_model, float)[..., cols],
                                            M_knots, m_fine))
        else:
            sm = np.broadcast_to(sigma_model, log_t.shape)[cols]
        pe = np.exp(thermal_loglike(lf, src, sm)) @ aw[cols]  # [m, G, E]
        out[i] = np.where(norm_e > 0, (pe * wE).sum(-1) / np.where(norm_e > 0, norm_e, 1), 0.0)
    return out.transpose(0, 2, 1)                             # [i, G, m]


@dataclass
class IsoPopulation:
    """Isolated-NS birth masses: truncated skew-normal plus an ECSN Gaussian (W3 form)."""
    mu: np.ndarray
    sigma: np.ndarray
    alpha: np.ndarray
    f_ecsn: np.ndarray
    M_min: float = 1.0

    @property
    def n(self):
        return len(np.atleast_1d(self.mu))


def population_density(pop: IsoPopulation, m_fine):
    """Untruncated densities F[k, m] and their CDF along m_fine (for truncation)."""
    mu, sg, al, fe = (np.atleast_1d(np.asarray(x, float))[:, None]
                      for x in (pop.mu, pop.sigma, pop.alpha, pop.f_ecsn))
    m = m_fine[None, :]
    z = (m - mu) / sg
    f_sn = np.exp(-0.5 * z * z) / (sg * np.sqrt(2 * np.pi)) * (1 + erf(al * z / np.sqrt(2)))
    # CDF by trapezoid on m_fine from M_min (scipy's skewnorm.cdf is far too slow here);
    # same quadrature as the likelihood, so truncation normalizations are consistent
    fa = f_sn * (m >= pop.M_min)
    F_sn = np.concatenate([np.zeros((fa.shape[0], 1)),
                           np.cumsum(0.5 * (fa[:, 1:] + fa[:, :-1]) * np.diff(m_fine), axis=1)], 1)
    f_ec = norm.pdf(m, M_ECSN, SIG_ECSN)
    F_ec = norm.cdf(m, M_ECSN, SIG_ECSN) - norm.cdf(pop.M_min, M_ECSN, SIG_ECSN)
    above = m >= pop.M_min
    return ((1 - fe) * f_sn * above, fe * f_ec * above), ((1 - fe) * F_sn, fe * F_ec)


def _trunc_norm(cdfs, m_fine, Mmax):
    """Per-component normalizations on [M_min, Mmax_e] -> arrays [k, e]."""
    return [np.stack([np.interp(Mmax, m_fine, row) for row in C]) for C in cdfs]


def log_like_grid(S, Mmax, pop: IsoPopulation, m_fine, chunk=64):
    """lnL[k, e, g] for kernels S[e, i, g, m] (all draws) and population samples k.

    Each component (skew-normal, ECSN) is renormalized on [M_min, Mmax_e] separately,
    as in nmb-papers W3 popmodel.p_mass, so f_ecsn stays the ECSN share.
    """
    S = np.asarray(S)
    nE, nI, nG, nm = S.shape
    (f_sn, f_ec), (F_sn, F_ec) = population_density(pop, m_fine)
    Zsn, Zec = _trunc_norm((F_sn, F_ec), m_fine, np.asarray(Mmax))   # [k, e]
    dm = np.gradient(m_fine)
    out = np.empty((pop.n, nE, nG))
    for e0 in range(0, nE, chunk):
        sl = slice(e0, min(nE, e0 + chunk))
        Se = S[sl] * dm                                                 # [e, i, g, m]
        flat = Se.reshape(-1, nm).T                                     # [m, e*i*g]
        a = (f_sn @ flat).reshape((pop.n,) + Se.shape[:3])
        b = (f_ec @ flat).reshape((pop.n,) + Se.shape[:3])
        zs, ze = Zsn[:, sl, None, None], Zec[:, sl, None, None]
        lik = np.where(zs > 0, a / np.where(zs > 0, zs, 1), 0.0) \
            + np.where(ze > 1e-300, b / np.where(ze > 1e-300, ze, 1), 0.0)
        with np.errstate(divide="ignore"):
            out[:, sl, :] = np.log(lik).sum(axis=2)
    return out


def posterior_weights(lnL, ln_prior_e=None, ln_prior_g=None, ln_prior_k=None):
    """Normalized weights over the (k, e, g) product of sample/draw/gap grids.

    Population samples k are draws from the population (hyper)prior, so ln_prior_k is
    only for extra factors (e.g. a mass-proxy likelihood); EOS draws e are prior draws
    reweighted by ln_prior_e (e.g. the macroscopic likelihood).
    """
    lw = np.array(lnL, float)
    if ln_prior_k is not None:
        lw = lw + np.asarray(ln_prior_k)[:, None, None]
    if ln_prior_e is not None:
        lw = lw + np.asarray(ln_prior_e)[None, :, None]
    if ln_prior_g is not None:
        lw = lw + np.asarray(ln_prior_g)[None, None, :]
    lw = np.where(np.isfinite(lw), lw, -np.inf)
    lz = logsumexp(lw)
    w = np.exp(lw - lz)
    return w, lz, 1.0 / np.sum(w ** 2)


def star_mass_posterior(w, S_i, Mmax, pop: IsoPopulation, m_fine):
    """Posterior density of source i's mass on m_fine, averaged over weights w[k, e, g].

    S_i: [e, g, m] kernel of that source.
    """
    (f_sn, f_ec), (F_sn, F_ec) = population_density(pop, m_fine)
    Zsn, Zec = _trunc_norm((F_sn, F_ec), m_fine, np.asarray(Mmax))
    out = np.zeros_like(m_fine)
    for k in np.nonzero(w.sum(axis=(1, 2)) > 1e-12 * w.sum())[0]:
        prior = (f_sn[k][None, :] / np.where(Zsn[k] > 0, Zsn[k], np.inf)[:, None]
                 + f_ec[k][None, :] / np.where(Zec[k] > 1e-300, Zec[k], np.inf)[:, None])
        prior = prior * (m_fine[None, :] <= np.asarray(Mmax)[:, None])   # [e, m]
        post = prior[:, None, :] * S_i                                    # [e, g, m]
        z = _trapz(post, m_fine, axis=-1)
        post = np.where(z[..., None] > 0, post / np.where(z > 0, z, 1)[..., None], 0.0)
        out += np.einsum("eg,egm->m", w[k], post)
    return out / w.sum()
