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
- ``dtilde``              binary intrinsic tidal response, 2607.15588 Eq. (18)
- ``FPeakFit``            M_tot f_pk = fit(D-tilde) families, Eqs. (19)-(21)
- ``STROUD2026_EXP``      verified Tab. II exponential coefficients
- ``fpeak_lnlike``        Gaussian-in-ln-f likelihood with fit systematic
- ``MthreshFit``          linear prompt-collapse threshold k = slope C + b
- ``BAUSWEIN2013``        verified linear fit, C*_1.6 compactness
- ``KOEPPEL2019_LIN``     verified linear fit, C_TOV compactness
- ``MthreshNonlinFit``    nonlinear threshold k = a - b/(1 - c C)
- ``KOEPPEL2019_NONLIN``  verified Koeppel Eq. (4) nonlinear fit
- ``mthresh_lnlike``      prompt-collapse (non)detection likelihood

House rule: every literature coefficient above is verified against the
source PDF (never memory / web extraction) and pinned by a numeric anchor
test that the source paper itself quotes.  PDF text extraction can drop
minus signs -- treat every sign as suspect until an anchor confirms it.
"""
from dataclasses import dataclass

import numpy as np

__all__ = ["MSUN_S", "GMSUN_KM", "dtilde", "FPeakFit", "STROUD2026_EXP",
           "fpeak_lnlike", "MthreshFit", "BAUSWEIN2013", "KOEPPEL2019_LIN",
           "MthreshNonlinFit", "KOEPPEL2019_NONLIN", "mthresh_lnlike"]

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
#: R^2 = 0.811, from Tab. II on PDF page 22).
#:
#: sigma_lnf residual refit (2026-07-24).  Exact (D-tilde, M f_pk) pairs
#: CANNOT be reconstructed from the PDF: the only tables are Tab. I (runs:
#: Model, M_NS, E_GW, J_GW, f_pk, t_BH -- PDF page 18) and Tab. II (fit
#: coefficients); D-tilde is never tabulated, and the CoRe points that
#: also enter the fit appear only as markers in Fig. 11.  Fallback used:
#: from Tab. I take the N=8 non-collapsing equal-mass runs with a quoted
#: f_pk (EOS-II/III/IV/V), form M f_pk = (G M_tot/c^3) f_pk with
#: M_tot = 2 M_NS; these span M f_pk in [0.0289, 0.0405].  With R^2 = 0.811
#: the residual scatter is sqrt(1 - R^2) * std(ln M f_pk) = 0.051 (and
#: sqrt(1-R^2)*std(M f_pk)/mean = 0.050), hence sigma_lnf = 0.051.
#: CAVEAT: this is a LOWER BOUND -- it uses only the tabulated simulation
#: subset, whose M f_pk spread (0.029-0.040) is narrower than the full
#: fitted set.  The full Fig. 11 range (M f_pk ~ 0.02-0.06, read off the
#: axis) with R^2 = 0.811 implies scatter as large as ~0.10-0.13 if the
#: CoRe points dominate the variance; treat sigma_lnf as uncertain at the
#: factor-of-2 level and prefer the larger value where conservatism on the
#: EOS systematic matters.  Fit calibrated on ~1.3-1.4 Msun near-equal-mass
#: binaries.
STROUD2026_EXP = FPeakFit(
    name="stroud2026-dtilde-exp", form="exp",
    m=3.38e-3, b=5.51e-2, sigma_lnf=0.051,
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


@dataclass(frozen=True)
class MthreshNonlinFit:
    """Nonlinear prompt-collapse threshold fit, Koeppel-Bovard-Rezzolla
    (2019) Eq. (4):

        k(C) = Mth / M_TOV = a - b / (1 - c C),   a = 2 b / (2 - c),

    where C is a TOV-scale compactness (C_TOV = G Mmax / c^2 R_TOV).  The
    tie a = 2b/(2-c) is imposed by the authors so the black-hole limit
    Mth / M_TOV -> 0 holds at C -> 1/2 (a maximally compact configuration
    is already collapsing); ``a`` is therefore a derived property, not a
    free field, and cannot drift out of sync with (b, c).  ``sigma_k`` is
    the fit residual scale in k.  Same call surface as ``MthreshFit`` so it
    drops straight into ``mthresh_lnlike``.
    """
    name: str
    b: float
    c: float
    sigma_k: float
    compactness: str          # "C_TOV"
    reference: str

    @property
    def a(self):
        return 2.0 * self.b / (2.0 - self.c)

    def k(self, C):
        C = np.asarray(C, float)
        return self.a - self.b / (1.0 - self.c * C)

    def mthresh(self, mmax_msun, radius_km):
        """Threshold total mass [Msun]; ``radius_km`` is R_TOV."""
        C = GMSUN_KM * np.asarray(mmax_msun, float) / np.asarray(radius_km, float)
        return self.k(C) * np.asarray(mmax_msun, float)


#: Koeppel, Bovard & Rezzolla, ApJL 872, L16 (2019) nonlinear threshold fit
#: (preferred over the linear variant at extreme compactness).  Their
#: Eq. (4), verified against arXiv:1901.09977 PDF (pages 3-4) 2026-07-24:
#:     Mth / M_TOV = a - b/(1 - c C_TOV),   a = 2 b/(2 - c),
#: with b = 1.01, c = 1.34 (=> a = 3.0606).  The a = 2b/(2-c) tie is the
#: authors' imposed black-hole limit Mth/M_TOV -> 0 at C_TOV -> 1/2.
#: Anchors from the paper: k(0.288) = 1.416 reproduces their lowest-order
#: quote Mth/M_TOV ~ 1.415 (their Eq. 3, near C_TOV ~ 0.30); k(0.5) = 0
#: exactly (BH limit); k is monotonically decreasing on C_TOV in
#: [0.24, 0.35]; it stays within their stated "8% at most" of the linear
#: variant over that range (7.1% at C_TOV = 0.35) and within ~3% near
#: C_TOV ~ 0.30.  sigma_k as for the linear variant (their Delta Mth ~
#: 0.05 Msun at Mmax ~ 2 -> ~0.025).
KOEPPEL2019_NONLIN = MthreshNonlinFit(
    name="koeppel2019-nonlinear-CTOV", b=1.01, c=1.34, sigma_k=0.025,
    compactness="C_TOV",
    reference="Koeppel, Bovard & Rezzolla, ApJL 872, L16 (2019), Eq. (4)",
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
