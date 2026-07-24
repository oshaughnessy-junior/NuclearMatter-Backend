"""Postmerger observable hooks (pure numpy).

Quasi-universal-fit likelihood factors for postmerger observables, usable
over the same EOS draw pools as the GW/NICER evidence hooks.  Design note
(paper-4 "structural" result): a fit whose regressor is an inspiral-scale
static (D-tilde, Lambda-tilde, R_1.6) can only re-sharpen information the
inspiral already carries; genuine high-density reach enters through fit
RESIDUALS correlating with high-density structure, or through
threshold-type observables (M_thresh / prompt collapse) keyed to TOV-scale
properties.  Both channel types belong here; a forecast built only on
statics-regressed f_peak fits is circular.

Implemented:
- ``dtilde``           binary intrinsic tidal response, 2607.15588 Eq. (18)
- ``FPeakFit``         M_tot f_pk = fit(D-tilde) families, Eqs. (19)-(21)
- ``STROUD2026_EXP``   verified Tab. II exponential coefficients
- ``fpeak_lnlike``     Gaussian-in-ln-f likelihood with fit systematic
- ``mthresh_lnlike``   NOT IMPLEMENTED YET (coefficients must be verified
                       against the source PDFs before entry; do not guess)
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["MSUN_S", "GMSUN_KM", "dtilde", "FPeakFit", "STROUD2026_EXP",
           "fpeak_lnlike", "MthreshFit", "BAUSWEIN2013", "KOEPPEL2019_LIN",
           "mthresh_lnlike"]

#: G Msun / c^3 in seconds.
MSUN_S = 4.925490947e-6


def dtilde(m1, m2, D1, D2):
    """Binary intrinsic tidal response, 2607.15588 Eq. (18).

    Same mass weighting as Lambda-tilde; equal masses give
    dtilde(m, m, D, D) = D exactly.
    """
    m1, m2 = np.asarray(m1, float), np.asarray(m2, float)
    D1, D2 = np.asarray(D1, float), np.asarray(D2, float)
    mt5 = (m1 + m2) ** 5
    return (16.0 / 13.0) * ((m1 + 12.0 * m2) * m1**4 * D1
                            + (m2 + 12.0 * m1) * m2**4 * D2) / mt5


@dataclass(frozen=True)
class FPeakFit:
    """A quasi-universal fit M_tot f_pk = form(D-tilde).

    ``sigma_lnf`` is the EOS-systematic scatter (in ln f) to be added in
    quadrature to any measurement error; treat it as a model parameter of
    the likelihood, not a constant of nature.
    """
    name: str
    form: str            # "linear" | "exp" | "power"
    m: float
    b: float
    sigma_lnf: float
    reference: str

    def mfpk(self, Dt):
        Dt = np.asarray(Dt, dtype=float)
        if self.form == "linear":
            return self.m * Dt + self.b
        if self.form == "exp":
            return self.b * np.exp(-self.m * Dt)
        if self.form == "power":
            return self.b * Dt**self.m
        raise ValueError(self.form)

    def fpk_hz(self, mtot_msun, Dt):
        """Peak frequency in Hz for total mass ``mtot_msun``."""
        return self.mfpk(Dt) / (np.asarray(mtot_msun, float) * MSUN_S)


#: 2607.15588 Tab. II exponential fit — coefficients verified against the
#: source PDF 2026-07-24 (m = 3.38e-3 +/- 1.85e-4, b = 5.51e-2 +/- 1.10e-3,
#: R^2 = 0.811).  sigma_lnf = 0.05 is a PLACEHOLDER pending a residual
#: refit; the fit was calibrated on ~1.3-1.4 Msun near-equal-mass binaries.
STROUD2026_EXP = FPeakFit(
    name="stroud2026-dtilde-exp", form="exp",
    m=3.38e-3, b=5.51e-2, sigma_lnf=0.05,
    reference="Stroud, Radice & Reddy, arXiv:2607.15588, Tab. II",
)


def fpeak_lnlike(f_obs_hz, sigma_obs_hz, f_pred_hz, sigma_lnf):
    """ln L for an observed peak frequency, Gaussian in ln f.

    Combines the relative measurement error and the fit's EOS-systematic
    width in quadrature.  Broadcasts over draw arrays in ``f_pred_hz``.
    """
    f_obs = float(f_obs_hz)
    sig_ln = np.hypot(float(sigma_obs_hz) / f_obs, float(sigma_lnf))
    z = (np.log(np.asarray(f_pred_hz, float)) - np.log(f_obs)) / sig_ln
    return -0.5 * z**2 - np.log(sig_ln) - 0.5 * np.log(2.0 * np.pi)


#: G Msun / c^2 in km.
GMSUN_KM = 1.4766


@dataclass(frozen=True)
class MthreshFit:
    """Linear prompt-collapse threshold fit M_thresh = k * M_max with
    k = slope * C + intercept, where C is a compactness built from
    TOV-scale statics.  ``sigma_k`` is the fit residual scale in k.
    """
    name: str
    slope: float
    intercept: float
    sigma_k: float
    compactness: str          # "C*_1.6" (G Mmax / c^2 R_1.6) or "C_TOV"
    reference: str

    def k(self, C):
        return self.slope * np.asarray(C, float) + self.intercept

    def mthresh(self, mmax_msun, radius_km):
        """Threshold total mass [Msun]; ``radius_km`` is R_1.6 or R_TOV
        per ``compactness``."""
        C = GMSUN_KM * np.asarray(mmax_msun, float) / np.asarray(radius_km, float)
        return self.k(C) * np.asarray(mmax_msun, float)


#: Bauswein, Baumgarte & Janka, PRL 111, 131101 (2013), verified against
#: the arXiv:1307.5191 PDF 2026-07-24: k = -3.606 C*_1.6 + 2.380 with
#: C*_1.6 = G Mmax / (c^2 R_1.6); maximum residual 0.025 in k.  Anchor
#: check against their table: TMA (Mmax 2.02, R_1.6 13.73) -> 3.23 vs
#: simulated 3.25; IUF (1.95, 12.57) -> 3.03 vs 3.05.
BAUSWEIN2013 = MthreshFit(
    name="bauswein2013-C16", slope=-3.606, intercept=2.380, sigma_k=0.025,
    compactness="C*_1.6",
    reference="Bauswein, Baumgarte & Janka, PRL 111, 131101 (2013)",
)

#: Koeppel, Bovard & Rezzolla, ApJL 872, L16 (2019) linear variant:
#: k = -3.38 C_TOV + 2.43.  CAUTION: the PDF text layer drops the minus
#: sign; the sign is fixed here by their own lowest-order quote
#: Mth/M_TOV ~ 1.415 at C_TOV ~ 0.30 and by physical monotonicity.
#: Their nonlinear fit (their Eq. 4) is preferred at extreme compactness
#: and is NOT yet implemented.  sigma_k from their Delta Mth ~ 0.05 Msun
#: at Mmax ~ 2 -> ~0.025.
KOEPPEL2019_LIN = MthreshFit(
    name="koeppel2019-linear-CTOV", slope=-3.38, intercept=2.43, sigma_k=0.025,
    compactness="C_TOV",
    reference="Koeppel, Bovard & Rezzolla, ApJL 872, L16 (2019)",
)


def _norm_cdf(x):
    from math import erf
    x = np.asarray(x, float)
    return 0.5 * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))


def mthresh_lnlike(prompt_collapse, mtot_msun, sigma_mtot_msun,
                   mmax_msun, radius_km, fit=BAUSWEIN2013):
    """ln L of observing prompt collapse (or not) at total mass ``mtot``.

    This is the postmerger channel that carries TOV-scale information
    (paper-4 structural argument): M_thresh is keyed to (Mmax, R) of the
    maximum-mass configuration, which inspiral leaves undetermined.

    Parameters broadcast over draw arrays in ``mmax_msun``/``radius_km``.
    ``prompt_collapse`` True: L = P(Mtot > M_thresh); False: 1 - that.
    The threshold uncertainty combines the fit residual (sigma_k * Mmax)
    and the measurement error on Mtot in quadrature.
    """
    mth = fit.mthresh(mmax_msun, radius_km)
    sigma = np.hypot(fit.sigma_k * np.asarray(mmax_msun, float),
                     float(sigma_mtot_msun))
    z = (float(mtot_msun) - mth) / sigma
    p = _norm_cdf(z)
    p = np.clip(p if prompt_collapse else 1.0 - p, 1e-300, 1.0)
    return np.log(p)
