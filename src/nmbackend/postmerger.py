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

__all__ = ["MSUN_S", "dtilde", "FPeakFit", "STROUD2026_EXP", "fpeak_lnlike",
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


def mthresh_lnlike(*args, **kwargs):
    """Prompt-collapse threshold-mass likelihood — not yet implemented.

    This is the channel that carries TOV-scale information (the paper-4
    structural argument).  Implementation requires verifying a published
    M_thresh fit (e.g. the Bauswein et al. threshold-mass relations)
    against the source PDFs; coefficients MUST NOT be entered from memory
    or LLM extraction without an anchor check.
    """
    raise NotImplementedError(
        "verify a published M_thresh fit against its PDF first; see docstring")
