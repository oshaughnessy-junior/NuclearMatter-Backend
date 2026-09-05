"""Differentiable ejecta-conditions surrogate (the bridge "mmapipe" segment).

Maps binary parameters + EOS-derived stellar quantities to merger-ejecta
conditions -- dynamical ejecta mass and velocity, remnant disk mass, and an
electron-fraction / lanthanide-proxy summary -- via published NR fitting
formulae, JAX end-to-end, so that composed with the metamodel + TOV machinery
one gets d(ejecta)/d(NEP) gradients (FRIB-SCOPING section 5: the
FRIB -> E_sym -> EOS -> ejecta -> nucleosynthesis chain).

Every quantity is returned as a :class:`UQ` (value, sigma_fit, sigma_formula)
-- never a bare number.  ``sigma_fit`` encodes the calibration papers' own
residual scatter; ``sigma_formula`` is the spread across independent fit
families evaluated on the same inputs.  These two systematics are what the
bridge's honest error bars are made of; downstream consumers (n2n trajectory
gradients, p5-crossover composition claims) must carry both.

Extensibility: fit families live in registries (``MEJ_FAMILIES`` etc.) of
:class:`FitFamily` entries -- a predictor plus a sigma model plus provenance
-- and ``ejecta_conditions`` accepts registry overrides.  The input struct
:class:`EjectaInputs` already carries aligned spins (chi1, chi2); every
family implemented today IGNORES spin (none of the published calibrations
include it), but a spin-aware fit is a registry entry away, with no API
change.  The same route serves future replacements for these fits, which are
known to be crude (factor-~2-level residuals; see each family's sigma note).

Implemented fit families (all coefficients verified against the source PDFs;
house one-strike rule as in ``postmerger.py``: PDF-verified, sign-suspect
until anchored):

- ``radice2018``   Radice et al. 2018, ApJ 869, 130 (arXiv:1809.11161).
                   M_ej: Eq. (18) [Dietrich & Ujevic form, refit], coefficients
                   Eqs. (20)-(21); v_ej: Eqs. (22)-(23); M_disk: Eq. (25)
                   (tanh in Lambda-tilde).  Calibrated on their fiducial
                   microphysical-EOS simulations without neutrino re-absorption,
                   q close to 1.
- ``kf2020``       Kruger & Foucart 2020, PRD 101, 103002 (arXiv:2002.07728).
                   M_dyn: Eq. (6) (compactness-based, no baryon masses needed);
                   M_disk: Eq. (4) (compactness of the lighter star).
                   Calibration: C_lighter in [0.135, 0.219], mass ratio
                   (lighter/heavier) in [0.775, 1].  No velocity fit.
- ``nedora2021``   Nedora et al. 2022, CQG 39, 015008 (arXiv:2011.11110).
                   Second-order polynomials P_2^2(q, Lambda-tilde), their
                   Eq. (6), in the calibration their conclusion recommends
                   (M0RefSet + M0/M1Set: "datasets with the most advanced
                   physics input", i.e. neutrino emission AND absorption).
                   Provides M_ej, v_ej, <Ye>, and M_disk.  q = m_heavy/m_light
                   >= 1; combined calibration range roughly q <= 1.3 (M0/M1Set)
                   to 1.8 (M0RefSet), Lambda-tilde in [340, 1440].

Electron-fraction component model (coarse, every number cited):

- tidal (cold) component   Ye ~ 0.10:  Radice et al. 2018 Sec. 3.1: the tidal
  tail is "partially reprocessed by weak reactions to slightly higher values
  of Ye ~ 0.1"; pre-reprocessing values are lower.  Carried as 0.10 +/- 0.05.
- shocked/polar component  Ye ~ 0.30:  same source: shocked material is
  "reprocessed to Ye > 0.25", with a neutrino-wind component "with Ye up to
  0.4".  Encoded as 0.30 +/- 0.08 (covers 0.25-0.4 at ~1 sigma).
- mass-weighted <Ye> comes from the Nedora et al. P_2^2 fit; the tidal mass
  fraction is then INFERRED by inverting the two-component mixture, so the
  component split inherits its (q, Lambda-tilde) dependence from a published
  fit rather than from an invented one.
- lanthanide proxy: smooth step at Ye_crit = 0.26 +/- 0.04, from Lippuner &
  Roberts 2015, ApJ 815, 82 (arXiv:1508.03133), abstract: "the ejecta is
  lanthanide-free for Ye >~ 0.22-0.30, depending on s and tau".

Differentiability contract: all outputs support jax.grad / jax.jacfwd with
respect to (m1, m2) and the EOS inputs (Lambda_i, R_i).  Clamps use
jnp.maximum / jnp.clip, so gradients are exactly zero in clamped (physically
saturated) regimes rather than NaN.  ``ejecta_from_family`` composes with
``tov.family.Family`` interpolators; ``tests/test_ejecta.py`` checks
d M_ej / d L_sym through the metamodel + TOV stack against finite differences.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import jax.numpy as jnp

from .constants import GMSUN_KM

__all__ = [
    "UQ", "EjectaInputs", "EjectaConditions", "FitFamily",
    "lambda_tilde", "baryonic_mass",
    "radice2018_mej", "radice2018_vej", "radice2018_mdisk",
    "kf2020_mdyn", "kf2020_mdisk",
    "nedora2021_mej", "nedora2021_vej", "nedora2021_ye", "nedora2021_mdisk",
    "MEJ_FAMILIES", "VEJ_FAMILIES", "MDISK_FAMILIES",
    "prompt_collapse_disk_factor", "lanthanide_weight",
    "ejecta_conditions", "ejecta_from_family",
    "YE_TIDAL", "YE_SHOCKED", "YE_LANTHANIDE_CRIT",
]


class UQ(NamedTuple):
    """A value with its two systematic error components.

    ``sigma_fit``      residual scatter of the underlying calibration(s), as
                       published (or explicitly encoded from published
                       statements -- see each fit's sigma note).
    ``sigma_formula``  spread (population std) across the independent fit
                       families evaluated at the same inputs; zero when only
                       one family exists for the quantity.
    """
    value: jnp.ndarray
    sigma_fit: jnp.ndarray
    sigma_formula: jnp.ndarray

    @property
    def sigma(self):
        """Total 1-sigma: fit scatter and inter-formula spread in quadrature."""
        return jnp.sqrt(self.sigma_fit ** 2 + self.sigma_formula ** 2)


class EjectaInputs(NamedTuple):
    """Binary + EOS-derived inputs to the fit families.

    Masses in Msun, radii in km, Lambda dimensionless, chi = aligned
    dimensionless spins (accepted for forward compatibility; all currently
    published calibrations are non-spinning and ignore them).  Derived
    quantities (compactness, Lambda-tilde, heavy/light ordering) are provided
    as methods so every family sees one convention.
    """
    m1: jnp.ndarray
    m2: jnp.ndarray
    lambda1: jnp.ndarray
    lambda2: jnp.ndarray
    r1_km: jnp.ndarray
    r2_km: jnp.ndarray
    chi1: jnp.ndarray = 0.0
    chi2: jnp.ndarray = 0.0

    @property
    def c1(self):
        return GMSUN_KM * self.m1 / self.r1_km

    @property
    def c2(self):
        return GMSUN_KM * self.m2 / self.r2_km

    @property
    def lam_tilde(self):
        return lambda_tilde(self.m1, self.m2, self.lambda1, self.lambda2)

    @property
    def q_heavy(self):
        """Mass ratio m_heavy/m_light >= 1 (Nedora et al. convention)."""
        return (jnp.maximum(self.m1, self.m2) / jnp.minimum(self.m1, self.m2))

    def light(self):
        """(mass, compactness) of the lighter star, differentiably selected."""
        swap = self.m2 > self.m1
        m = jnp.where(swap, self.m1, self.m2)
        c = jnp.where(swap, self.c1, self.c2)
        return m, c


@dataclass(frozen=True)
class FitFamily:
    """One published fitting formula: predictor, sigma model, provenance.

    ``predict(x)`` maps :class:`EjectaInputs` to the quantity; ``sigma(v, x)``
    maps (predicted value, inputs) to the family's own 1-sigma residual
    scatter.  ``cite`` is the arXiv id; ``notes`` records calibration range
    and caveats.  Registering a new (e.g. spin-dependent) family means adding
    one of these to the relevant registry -- nothing else changes.
    """
    name: str
    cite: str
    predict: Callable
    sigma: Callable
    notes: str = ""


class EjectaConditions(NamedTuple):
    """Ejecta-conditions summary.  Masses in Msun, velocities in c, Ye pure."""
    m_dyn: UQ            # dynamical ejecta mass
    v_dyn: UQ            # mass-averaged asymptotic dynamical-ejecta velocity
    m_disk: UQ           # remnant accretion-disk mass
    ye_dyn: UQ           # mass-averaged electron fraction of dynamical ejecta
    ye_tidal: UQ         # characteristic Ye, tidal component (literature const)
    ye_shocked: UQ       # characteristic Ye, shocked component (literature const)
    f_tidal: UQ          # tidal mass fraction inferred from the Ye mixture
    x_lanthanide: UQ     # lanthanide-rich mass-fraction proxy of the dyn. ejecta
    m_dyn_by: dict       # per-family central values (Msun)
    v_dyn_by: dict       # per-family central values (c)
    m_disk_by: dict      # per-family central values (Msun)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

def lambda_tilde(m1, m2, lambda1, lambda2):
    """Reduced (effective) tidal deformability Lambda-tilde.

    Standard mass weighting (Favata 2014 / Wade et al. 2014); same weights as
    ``postmerger.dtilde``, so equal masses give lambda_tilde(m,m,L,L) = L.
    """
    mt5 = (m1 + m2) ** 5
    return (16.0 / 13.0) * ((m1 + 12.0 * m2) * m1 ** 4 * lambda1
                            + (m2 + 12.0 * m1) * m2 ** 4 * lambda2) / mt5


def baryonic_mass(m, C):
    """Baryonic mass from gravitational mass and compactness.

    Kruger & Foucart 2020 Eq. (7): M_b = M (1 + 0.6 C / (1 - 0.5 C)); the
    approximation they themselves use to evaluate baryon-mass-based ejecta
    fits when only (M, R) are known.
    """
    return m * (1.0 + 0.6 * C / (1.0 - 0.5 * C))


# ----------------------------------------------------------------------------
# Radice et al. 2018 (arXiv:1809.11161)
# ----------------------------------------------------------------------------

# Eq. (20)-(21), verified from the PDF.
_R18_MEJ = dict(alpha=-0.657, beta=4.254, gamma=-32.61, delta=5.205, n=-0.773)
# Eq. (23), verified.
_R18_VEJ = dict(alpha=-0.287, beta=0.494, gamma=-3.000)
# Eq. (25), verified; floor 1e-3 Msun.
_R18_MDISK = dict(alpha=0.084, beta=0.127, gamma=567.1, delta=405.14)


def radice2018_mej(x: EjectaInputs):
    """Dynamical ejecta mass [Msun], Radice et al. 2018 Eq. (18).

    M_ej/(1e-3 Msun) = [alpha (M_b/M_a)^(1/3) (1-2C_a)/C_a
                        + beta (M_b/M_a)^n + gamma (1 - M_a/M_a*)] M_a*
                       + (a <-> b) + delta,
    with M* the baryonic masses (formed here via the KF2020 Eq. (7)
    approximation).  Negative predictions are clamped to zero.
    """
    p = _R18_MEJ
    mb1 = baryonic_mass(x.m1, x.c1)
    mb2 = baryonic_mass(x.m2, x.c2)

    def one(ma, mb, ca, mstar_a):
        return (p["alpha"] * (mb / ma) ** (1.0 / 3.0) * (1.0 - 2.0 * ca) / ca
                + p["beta"] * (mb / ma) ** p["n"]
                + p["gamma"] * (1.0 - ma / mstar_a)) * mstar_a

    fit = one(x.m1, x.m2, x.c1, mb1) + one(x.m2, x.m1, x.c2, mb2) + p["delta"]
    return 1.0e-3 * jnp.maximum(fit, 0.0)


def radice2018_vej(x: EjectaInputs):
    """Mass-averaged asymptotic ejecta velocity [c], Radice et al. 2018 Eq. (22).

    v = [alpha (M_a/M_b)(1 + gamma C_a)] + (a <-> b) + beta.
    """
    p = _R18_VEJ
    v = (p["alpha"] * (x.m1 / x.m2) * (1.0 + p["gamma"] * x.c1)
         + p["alpha"] * (x.m2 / x.m1) * (1.0 + p["gamma"] * x.c2) + p["beta"])
    return jnp.maximum(v, 0.0)


def radice2018_mdisk(x: EjectaInputs):
    """Remnant disk mass [Msun], Radice et al. 2018 Eq. (25).

    M_disk = max(1e-3, alpha + beta * tanh((Lambda-tilde - gamma)/delta)).
    Calibrated on comparable-mass binaries; they warn that large mass
    asymmetry and prompt collapse are outside the fit.
    """
    p = _R18_MDISK
    m = (p["alpha"]
         + p["beta"] * jnp.tanh((x.lam_tilde - p["gamma"]) / p["delta"]))
    return jnp.maximum(m, 1.0e-3)


# ----------------------------------------------------------------------------
# Kruger & Foucart 2020 (arXiv:2002.07728)
# ----------------------------------------------------------------------------

# Eq. (6), verified.
_KF20_MDYN = dict(a=-9.3335, b=114.17, c=-337.56, n=1.5465)
# Eq. (4), verified; floor 5e-4 * M_lighter.
_KF20_MDISK = dict(a=-8.1324, c=1.4820, d=1.7784)


def kf2020_mdyn(x: EjectaInputs):
    """Dynamical ejecta mass [Msun], Kruger & Foucart 2020 Eq. (6).

    M_dyn/(1e-3 Msun) = [a/C_1 + b (M_2/M_1)^n + c C_1] M_1 + (1 <-> 2),
    negative values -> 0.  Vanishes for both very compact and very extended
    stars (physical prompt-collapse / no-shock limits, their Sec. III).
    """
    p = _KF20_MDYN

    def one(ma, mb, ca):
        return (p["a"] / ca + p["b"] * (mb / ma) ** p["n"] + p["c"] * ca) * ma

    fit = one(x.m1, x.m2, x.c1) + one(x.m2, x.m1, x.c2)
    return 1.0e-3 * jnp.maximum(fit, 0.0)


def kf2020_mdisk(x: EjectaInputs):
    """Remnant disk mass [Msun], Kruger & Foucart 2020 Eq. (4).

    M_disk = M_light * max(5e-4, (a C_light + c)^d), with the compactness of
    the LIGHTER star (their M1).  Anchor (their Table I): C1 = 0.167,
    M1 = 1.5 -> M_disk = 0.037 Msun.
    """
    p = _KF20_MDISK
    m_light, c_light = x.light()
    base = jnp.maximum(p["a"] * c_light + p["c"], 0.0)
    return m_light * jnp.maximum(base ** p["d"], 5.0e-4)


# ----------------------------------------------------------------------------
# Nedora et al. 2021 (arXiv:2011.11110)
# ----------------------------------------------------------------------------

# P_2^2(q, Lt) = b0 + b1 q + b2 Lt + b3 q^2 + b4 q Lt + b5 Lt^2, their Eq. (6),
# q = m_heavy/m_light >= 1.  Rows "& M0/M1Set" of Table IV (ejecta) and
# Table VI (disk): the calibration recommended in their conclusion.  Verified
# from the PDF.
_N21_LOG10_MEJ = (-1.32, -3.82e-1, -4.47e-3, -3.39e-1, 3.21e-3, 4.31e-7)
_N21_VEJ = (5.94e-1, -1.48e-1, -8.62e-4, -5.02e-2, 3.25e-4, 3.16e-7)
_N21_YE = (2.55e-1, 3.83e-2, 2.36e-4, -6.66e-2, -1.92e-4, -1.86e-8)
_N21_MDISK = (-1.85, 2.59, 7.07e-4, -7.33e-1, -8.08e-4, 2.75e-7)


def _p22(coef, q, lt):
    b0, b1, b2, b3, b4, b5 = coef
    return b0 + b1 * q + b2 * lt + b3 * q ** 2 + b4 * q * lt + b5 * lt ** 2


def nedora2021_mej(x: EjectaInputs):
    """Dynamical ejecta mass [Msun] = 10**P_2^2(q, Lt), Nedora et al. Table IV
    (the fitted quantity is log10(M_ej/Msun))."""
    return 10.0 ** _p22(_N21_LOG10_MEJ, x.q_heavy, x.lam_tilde)


def nedora2021_vej(x: EjectaInputs):
    """Mean asymptotic ejecta velocity [c] = P_2^2(q, Lt), Nedora et al."""
    return jnp.maximum(_p22(_N21_VEJ, x.q_heavy, x.lam_tilde), 0.0)


def nedora2021_ye(x: EjectaInputs):
    """Mass-averaged dynamical-ejecta Ye = P_2^2(q, Lt), Nedora et al.

    Anchor: over the restricted common range q in (1, 1.2), Lt in (350, 850)
    their M0RefSet + M0/M1Set mean is <Ye> = 0.20 +/- 0.04 (Sec. III C).
    Clipped to [0.01, 0.5] outside the calibrated region.
    """
    return jnp.clip(_p22(_N21_YE, x.q_heavy, x.lam_tilde), 0.01, 0.5)


def nedora2021_mdisk(x: EjectaInputs):
    """Remnant disk mass [Msun] = P_2^2(q, Lt), Nedora et al. Table VI.

    The disk fit is performed as log10(data) vs log10(P_2^2) with a -3.0
    floor, i.e. the polynomial gives the mass directly, floored at 1e-3 Msun.
    """
    return jnp.maximum(_p22(_N21_MDISK, x.q_heavy, x.lam_tilde), 1.0e-3)


# ----------------------------------------------------------------------------
# family registries (extension point: add spin-aware / future fits here)
# ----------------------------------------------------------------------------
# sigma models -- each an explicit encoding of a published statement:
#  * radice2018 M_ej: their Eq. (17) numerical floor is 0.5*M + 5e-5, but
#    Sec. 4.1 reports fit-vs-data "differences ... as large as ~200%"; we read
#    200% as a ~2-sigma excursion and carry sigma = 1.0*M + 5e-5 Msun.
#  * kf2020 M_dyn: sigma = 0.004 Msun, their Gaussian fit to the residuals.
#  * nedora2021 M_ej: conclusion quotes accuracy "to a factor of ~2", 68%
#    fractional range (-0.8, +0.2); encoded as sigma = 0.5*M.
#  * radice2018 v_ej: fiducial numerical uncertainty 0.02 c, residuals "as
#    large as 0.1 c"; 0.1 c read as ~2 sigma -> sigma = 0.05 c.
#  * nedora2021 v_ej: "typically to ~50%", 68% range (-0.4, +0.2) -> 0.3*v.
#  * disk masses: Radice Eq. (24) / KF2020 Eq. (3) both use
#    Delta M_disk = 0.5*M + 5e-4; Nedora 68% range (-0.4, +0.2) -> 0.3*M + 5e-4.

MEJ_FAMILIES = {
    "radice2018": FitFamily(
        "radice2018", "1809.11161", radice2018_mej,
        lambda v, x: 1.0 * v + 5.0e-5,
        "DU-form refit; microphysical EOS, no nu-absorption, q ~ 1"),
    "kf2020": FitFamily(
        "kf2020", "2002.07728", kf2020_mdyn,
        lambda v, x: jnp.full_like(jnp.asarray(v, float), 4.0e-3),
        "compactness form; calibrated incl. asymmetric + near-threshold runs"),
    "nedora2021": FitFamily(
        "nedora2021", "2011.11110", nedora2021_mej,
        lambda v, x: 0.5 * v,
        "P22(q, Lt), M0RefSet+M0/M1Set (recommended row)"),
}

VEJ_FAMILIES = {
    "radice2018": FitFamily(
        "radice2018", "1809.11161", radice2018_vej,
        lambda v, x: jnp.full_like(jnp.asarray(v, float), 0.05), ""),
    "nedora2021": FitFamily(
        "nedora2021", "2011.11110", nedora2021_vej,
        lambda v, x: 0.3 * v, ""),
}

MDISK_FAMILIES = {
    "radice2018": FitFamily(
        "radice2018", "1809.11161", radice2018_mdisk,
        lambda v, x: 0.5 * v + 5.0e-4, "tanh in Lambda-tilde"),
    "kf2020": FitFamily(
        "kf2020", "2002.07728", kf2020_mdisk,
        lambda v, x: 0.5 * v + 5.0e-4, "lighter-star compactness"),
    "nedora2021": FitFamily(
        "nedora2021", "2011.11110", nedora2021_mdisk,
        lambda v, x: 0.3 * v + 5.0e-4, "P22(q, Lt)"),
}


# ----------------------------------------------------------------------------
# electron fraction, lanthanide proxy, prompt collapse
# ----------------------------------------------------------------------------

# Component Ye constants: see module docstring for the exact quotes.
YE_TIDAL = UQ(jnp.asarray(0.10), jnp.asarray(0.05), jnp.asarray(0.0))
YE_SHOCKED = UQ(jnp.asarray(0.30), jnp.asarray(0.08), jnp.asarray(0.0))
# Lippuner & Roberts 2015: lanthanide-free for Ye >~ 0.22-0.30 (s, tau dep.).
YE_LANTHANIDE_CRIT = UQ(jnp.asarray(0.26), jnp.asarray(0.04), jnp.asarray(0.0))
# Smoothing width of the lanthanide step in Ye: OUR regularization choice
# (comparable to the s,tau-induced blurring of the threshold), not published.
_LAN_WIDTH = 0.03


def lanthanide_weight(ye, ye_crit=None):
    """Smooth lanthanide-rich indicator in [0, 1]: sigmoid((Ye_c - Ye)/w)."""
    yc = YE_LANTHANIDE_CRIT.value if ye_crit is None else ye_crit
    return 1.0 / (1.0 + jnp.exp((ye - yc) / _LAN_WIDTH))


def prompt_collapse_disk_factor(m_tot, m_thresh, width=0.06):
    """Smooth disk-suppression factor in [0, 1] for prompt collapse.

    Kruger & Foucart 2020 (quoting the Coughlin et al. total-mass disk fit)
    state that a binary with "M_tot >= 0.95 M_thr will not result in a
    significant accretion disk"; the tanh width of that fit is d = 0.05957 in
    M_tot/M_thr.  Encoded as a sigmoid centered at 0.95 M_thr with that
    width.  Coarse gate, applied only when a threshold mass is supplied; its
    location uncertainty is at least the M_thresh fit systematics
    (see ``postmerger.MthreshFit``).
    """
    return 1.0 / (1.0 + jnp.exp(-(0.95 - m_tot / m_thresh) / width))


# ----------------------------------------------------------------------------
# combined surrogate
# ----------------------------------------------------------------------------

def _combine(families: dict, x: EjectaInputs):
    """Evaluate a family registry: mean value, RMS sigma_fit, population
    spread across families.  Returns (UQ, {name: value})."""
    by = {name: fam.predict(x) for name, fam in families.items()}
    vals = jnp.stack(list(by.values()))
    sigs = jnp.stack([jnp.broadcast_to(families[n].sigma(by[n], x),
                                       jnp.shape(vals[0])) for n in by])
    return UQ(jnp.mean(vals, axis=0),
              jnp.sqrt(jnp.mean(sigs ** 2, axis=0)),
              jnp.std(vals, axis=0)), by


def ejecta_conditions(m1, m2, lambda1, lambda2, r1_km, r2_km,
                      chi1=0.0, chi2=0.0,
                      m_thresh: Optional[float] = None,
                      mej_families: Optional[dict] = None,
                      vej_families: Optional[dict] = None,
                      mdisk_families: Optional[dict] = None,
                      ) -> EjectaConditions:
    """Ejecta conditions + systematics from binary and EOS-derived inputs.

    Parameters: gravitational masses [Msun], dimensionless tidal
    deformabilities Lambda_i, radii [km]; optionally aligned spins (carried,
    unused by current families), a threshold mass [Msun] to gate the disk for
    prompt collapse (e.g. from the ``postmerger`` M_thresh fits), and registry
    overrides to swap or extend the fit families per quantity.

    Inputs may be scalars or broadcastable jnp arrays; the result is a pytree
    of :class:`UQ` entries, differentiable end-to-end.
    """
    x = EjectaInputs(
        m1=jnp.asarray(m1, float), m2=jnp.asarray(m2, float),
        lambda1=jnp.asarray(lambda1, float), lambda2=jnp.asarray(lambda2, float),
        r1_km=jnp.asarray(r1_km, float), r2_km=jnp.asarray(r2_km, float),
        chi1=jnp.asarray(chi1, float), chi2=jnp.asarray(chi2, float))

    m_dyn, mej_by = _combine(mej_families or MEJ_FAMILIES, x)
    v_dyn, vej_by = _combine(vej_families or VEJ_FAMILIES, x)

    mdisk_reg = mdisk_families or MDISK_FAMILIES
    if m_thresh is None:
        m_disk, mdisk_by = _combine(mdisk_reg, x)
    else:
        gate = prompt_collapse_disk_factor(x.m1 + x.m2, m_thresh)
        gated = {
            n: FitFamily(f.name, f.cite,
                         (lambda xx, _f=f: gate * _f.predict(xx)),
                         (lambda v, xx, _f=f: gate * _f.sigma(v, xx)),
                         f.notes)
            for n, f in mdisk_reg.items()}
        m_disk, mdisk_by = _combine(gated, x)

    # --- electron fraction & lanthanide proxy ---
    ye = nedora2021_ye(x)
    # sigma_fit = 0.03: Nedora's fiducial Delta Ye = 0.01 scaled by the
    # recommended calibration's chi2_nu = 9.6 (sqrt(9.6)*0.01 ~= 0.03); their
    # all-datasets accuracy quote is ~0.1, treated as an extreme bound.
    ye_dyn = UQ(ye, jnp.broadcast_to(jnp.asarray(0.03), jnp.shape(ye)),
                jnp.zeros(jnp.shape(ye)))

    # Tidal mass fraction from inverting the two-component Ye mixture:
    # <Ye> = f * Ye_tidal + (1 - f) * Ye_shocked.  Clip to [0, 1]; the clip's
    # zero gradient outside the branch is the physically saturated case.
    span = YE_SHOCKED.value - YE_TIDAL.value
    f = jnp.clip((YE_SHOCKED.value - ye) / span, 0.0, 1.0)
    df = jnp.sqrt((ye_dyn.sigma_fit / span) ** 2
                  + (f * YE_TIDAL.sigma_fit / span) ** 2
                  + ((1.0 - f) * YE_SHOCKED.sigma_fit / span) ** 2)
    f_tidal = UQ(f, df, jnp.zeros(jnp.shape(f)))

    # Lanthanide-rich mass-fraction proxy: component-weighted smooth step.
    def xlan(yc):
        return (f * lanthanide_weight(YE_TIDAL.value, yc)
                + (1.0 - f) * lanthanide_weight(YE_SHOCKED.value, yc))
    x0 = xlan(YE_LANTHANIDE_CRIT.value)
    # half-range over the published Ye_crit band as the threshold systematic,
    # plus the mixture-fraction uncertainty
    dx_crit = 0.5 * jnp.abs(
        xlan(YE_LANTHANIDE_CRIT.value + YE_LANTHANIDE_CRIT.sigma_fit)
        - xlan(YE_LANTHANIDE_CRIT.value - YE_LANTHANIDE_CRIT.sigma_fit))
    dx_f = jnp.abs(lanthanide_weight(YE_TIDAL.value)
                   - lanthanide_weight(YE_SHOCKED.value)) * f_tidal.sigma_fit
    x_lan = UQ(x0, jnp.sqrt(dx_crit ** 2 + dx_f ** 2), jnp.zeros(jnp.shape(x0)))

    return EjectaConditions(
        m_dyn=m_dyn, v_dyn=v_dyn, m_disk=m_disk, ye_dyn=ye_dyn,
        ye_tidal=YE_TIDAL, ye_shocked=YE_SHOCKED, f_tidal=f_tidal,
        x_lanthanide=x_lan,
        m_dyn_by=mej_by, v_dyn_by=vej_by, m_disk_by=mdisk_by,
    )


def ejecta_from_family(family, m1, m2, m_thresh=None, **kw) -> EjectaConditions:
    """Ejecta conditions from a TOV ``Family`` (the bridge composition point).

    Pulls Lambda(m) and R(m) from the family's stable-branch interpolators, so
    jax.jacfwd of any output with respect to the nuclear empirical parameters
    flows through metamodel -> TOV -> here (see tests/test_ejecta.py).
    Extra keyword arguments pass through to :func:`ejecta_conditions`.
    """
    from .tov.family import lambda_of_mass, radius_of_mass
    return ejecta_conditions(
        m1, m2,
        lambda_of_mass(family, m1), lambda_of_mass(family, m2),
        radius_of_mass(family, m1), radius_of_mass(family, m2),
        m_thresh=m_thresh, **kw)
