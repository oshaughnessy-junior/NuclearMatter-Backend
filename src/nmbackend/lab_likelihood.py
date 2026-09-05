"""Density-tagged laboratory (FRIB/HIC) likelihoods for the metamodel EOS (MB-N2).

Two encodings of laboratory symmetry-energy constraints, per the FRIB scoping
decision (nmb-papers ``scoping/FRIB-SCOPING.md`` §2, §3.2 proposal MB-N2):

1. **Density-tagged pressure likelihood** (recommended primary encoding, after
   Huth et al. arXiv:2107.06229 and Tsang et al. arXiv:2310.11588): each
   constraint k carries a *sensitive density* n_k and a Gaussian measurement of
   the symmetry pressure there,

       lnL(EOS) = sum_k  ln N( P_sym(n_k); mu_k, sigma_k^2 + sigma_model^2 )

   where sigma_model is the **transport-model systematic floor** — the
   across-code spread of HIC extractions (e.g. SpiRIT's seven-code spread).
   sigma_model may be a fixed value *or a prior*, in which case it is
   **marginalized numerically** (`SigmaModelPrior`).  The hierarchical
   treatment of the lab systematic is this module's novel feature and the
   honest response to the PREX/CREX episode: never encode an observable as a
   fixed-width likelihood when the extraction map itself is uncertain.

2. **Correlated (E_sym, L_sym) Gaussian** (`CorrelatedSymGaussian`): the cheap
   parameter-space alternative, with a full 2x2 covariance (off-diagonal
   E_sym–L_sym correlation), matching the metamodel NEP coordinates directly.
   Pitfall to remember: this form imports whatever model map was used to
   extract (S0, L) from data.

P_sym(n) is evaluated from the metamodel symmetry-energy expansion
(``eos/metamodel.py``: S(n) = E_sym + L_sym x + K_sym x^2/2 + Q_sym x^3/6,
x = (n - n_sat)/(3 n_sat)) via the thermodynamic derivative

    P_sym(n) = n^2 dS/dn        [MeV/fm^3],

computed with ``jax.grad`` of the *same* ``_esat_S`` the EOS table uses, so the
likelihood and the EOS can never use different expansions.  At saturation this
reduces to the textbook P_sym(n_sat) = n_sat L_sym / 3.  Everything here is
differentiable w.r.t. the NEPs (fixed *and* marginalized forms), so these
likelihoods drop into the Fisher / AD-Jacobian machinery unchanged.

**Beta-equilibrium caveat (MB-N5) — read before tagging densities above
~2 n_sat.**  In the full bridge EOS (``priors/bridge.py``), everything above
the matching density (default ``n_match_fac * n_sat`` = 2 n_sat) is a
nonparametric *beta-equilibrated* sector: it carries no isospin decomposition,
so a symmetric/asymmetric-matter quantity like P_sym is simply not defined
there for a bridge draw.  A density-tagged likelihood with n_k above the
matching density therefore applies to the **metamodel sector only** (the
analytic expansion extrapolated past its match point), *not* to the bridge —
it constrains bridge draws only indirectly, through the sub-2 n_sat anchor and
continuity.  Quantifying how much of a P_sym(2 n0) likelihood survives that
beta-eq projection is exactly the open "isospin map" question (MB-N5); this
module warns when you cross the line but does not resolve it.

Scenario input: `load_scenario_ladder()` reads the packaged **copy** of the
FRIB scenario ladder (``data/frib_scenario_ladder.yaml``).  The source of
truth lives in nmb-papers ``scoping/frib_scenario_ladder.yaml``; the copy
carries a provenance header, and every number in it is an *assumption* — no
published paper commits to "sigma_L = +/-X by year Y" for FRIB.

Downstream consumers: the p5-crossover Tier-B forecast (claim C3 encoding
comparison) and the paper-1 EM-likelihood robustness variant in nmb-papers.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from importlib import resources
from typing import Optional, Sequence, Union

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .eos.metamodel import DEFAULT_PARAMS, _esat_S

jax.config.update("jax_enable_x64", True)

_LOG_2PI = float(np.log(2.0 * np.pi))

__all__ = [
    "psym",
    "DensityTaggedConstraint",
    "SigmaModelPrior",
    "DensityTaggedPressureLikelihood",
    "CorrelatedSymGaussian",
    "load_scenario_ladder",
    "scenario_by_name",
    "gaussian_from_scenario",
    "density_tagged_from_scenario",
]


# ----------------------------------------------------------------------------
# P_sym(n) from the metamodel expansion
# ----------------------------------------------------------------------------

def _merged(params: Optional[dict]) -> dict:
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    return p


def psym(n, params: Optional[dict] = None):
    """Symmetry pressure P_sym(n) = n^2 dS/dn [MeV/fm^3] of the metamodel.

    ``n`` is baryon density in fm^-3 (scalar or array); ``params`` overrides
    ``eos.metamodel.DEFAULT_PARAMS`` (E_sym, L_sym, K_sym, Q_sym, n_sat, ...).
    Differentiable w.r.t. the NEPs.  dS/dn is taken with ``jax.grad`` of the
    same symmetry-energy expansion the EOS table is built from, so this cannot
    drift from the EOS.  At n = n_sat it equals n_sat * L_sym / 3.
    """
    p = _merged(params)
    n_arr = jnp.atleast_1d(jnp.asarray(n, dtype=jnp.float64))
    dS_dn = jax.vmap(jax.grad(lambda nn: _esat_S(nn, p)[1]))(n_arr)
    out = n_arr**2 * dS_dn
    return out[0] if jnp.ndim(jnp.asarray(n)) == 0 else out


# ----------------------------------------------------------------------------
# density-tagged pressure likelihood
# ----------------------------------------------------------------------------

@dataclass(frozen=True)
class DensityTaggedConstraint:
    """One laboratory constraint: P_sym at its sensitive density.

    Density-sensitivity assignment after Tsang et al. arXiv:2310.11588: each
    experiment constrains the symmetry pressure at one tagged density n_k.

    Attributes
    ----------
    n : float        sensitive density [fm^-3]
    mu : float       measured P_sym(n) central value [MeV/fm^3]
    sigma : float    measurement (statistical+experimental) width [MeV/fm^3];
                     the transport-model floor sigma_model is NOT included
                     here — it is a shared hyperparameter of the likelihood.
    name : str       label (e.g. "SpiRIT pions", "FRIB400 P(2n0)")
    source : str     provenance note (arXiv id / scenario ladder entry)
    """
    n: float
    mu: float
    sigma: float
    name: str = ""
    source: str = ""

    def __post_init__(self):
        if not (self.n > 0.0):
            raise ValueError(f"constraint {self.name!r}: density must be > 0")
        if not (self.sigma > 0.0):
            raise ValueError(f"constraint {self.name!r}: sigma must be > 0")


@dataclass(frozen=True)
class SigmaModelPrior:
    """Numerical prior over the transport-model systematic floor sigma_model.

    Represented as quadrature nodes ``grid`` (sigma values, MeV/fm^3) with
    normalized ``log_weights`` (prior pdf x trapezoid weight, normalized so
    logsumexp(log_weights) = 0).  Marginalization is then a logsumexp — exact
    for a delta prior, trapezoid-accurate otherwise, and differentiable
    w.r.t. the EOS parameters.
    """
    grid: tuple = field(repr=False)
    log_weights: tuple = field(repr=False)
    label: str = ""

    @staticmethod
    def _normalize(grid, logw, label):
        grid = np.asarray(grid, dtype=float)
        logw = np.asarray(logw, dtype=float)
        if grid.ndim != 1 or grid.shape != logw.shape or grid.size < 1:
            raise ValueError("grid and log_weights must be equal-length 1D")
        if np.any(grid < 0.0):
            raise ValueError("sigma_model support must be >= 0")
        from scipy.special import logsumexp as _lse
        logw = logw - _lse(logw)
        return SigmaModelPrior(tuple(grid), tuple(logw), label)

    @classmethod
    def fixed(cls, value: float) -> "SigmaModelPrior":
        """Delta prior — marginalization reduces exactly to the fixed form."""
        return cls._normalize([value], [0.0], f"fixed({value})")

    @classmethod
    def uniform(cls, lo: float, hi: float, n: int = 129) -> "SigmaModelPrior":
        if not (0.0 <= lo < hi):
            raise ValueError("need 0 <= lo < hi")
        g = np.linspace(lo, hi, n)
        w = np.full(n, 1.0)
        w[0] *= 0.5
        w[-1] *= 0.5                     # trapezoid; flat pdf
        return cls._normalize(g, np.log(w), f"uniform[{lo},{hi}]")

    @classmethod
    def half_normal(cls, scale: float, n: int = 129,
                    n_scales: float = 5.0) -> "SigmaModelPrior":
        """Half-normal on sigma_model >= 0 (truncated at n_scales*scale)."""
        if not scale > 0.0:
            raise ValueError("scale must be > 0")
        g = np.linspace(0.0, n_scales * scale, n)
        logpdf = -0.5 * (g / scale) ** 2
        w = np.full(n, 1.0)
        w[0] *= 0.5
        w[-1] *= 0.5
        return cls._normalize(g, logpdf + np.log(w), f"half_normal({scale})")

    @classmethod
    def from_logpdf(cls, logpdf, lo: float, hi: float,
                    n: int = 129, label: str = "custom") -> "SigmaModelPrior":
        """Arbitrary (unnormalized) logpdf callable on [lo, hi]."""
        g = np.linspace(lo, hi, n)
        lp = np.array([float(logpdf(s)) for s in g])
        w = np.full(n, 1.0)
        w[0] *= 0.5
        w[-1] *= 0.5
        return cls._normalize(g, lp + np.log(w), label)


def _ln_normal(x, mu, var):
    return -0.5 * (_LOG_2PI + jnp.log(var) + (x - mu) ** 2 / var)


class DensityTaggedPressureLikelihood:
    """lnL(EOS) = sum_k ln N(P_sym(n_k); mu_k, sigma_k^2 + sigma_model^2).

    ``sigma_model`` (the transport-model systematic floor, shared across
    constraints) is either a fixed float or a `SigmaModelPrior`; with a prior
    it is marginalized numerically:

        lnL_marg = ln  integral  dsigma p(sigma) prod_k N(...; sigma_k^2 + sigma^2)

    Both forms include the Gaussian normalization (mandatory for the
    marginalization to be meaningful) and are differentiable w.r.t. the NEP
    dict passed to ``__call__`` / ``ln_likelihood``.

    Beta-equilibrium caveat (MB-N5): constraints tagged above the metamodel
    matching density (``n_match_fac * n_sat``, default 2 n_sat) apply to the
    **metamodel sector only**, not to the beta-equilibrated nonparametric
    bridge — see the module docstring.  Construction warns when a tag crosses
    that line (silence with ``warn_above_match=False`` once acknowledged).
    """

    def __init__(self,
                 constraints: Sequence[DensityTaggedConstraint],
                 sigma_model: Union[float, SigmaModelPrior] = 0.0,
                 warn_above_match: bool = True,
                 eos_params: Optional[dict] = None):
        if len(constraints) == 0:
            raise ValueError("need at least one constraint")
        self.constraints = tuple(constraints)
        self.sigma_model = sigma_model
        if isinstance(sigma_model, (int, float)) and sigma_model < 0.0:
            raise ValueError("sigma_model must be >= 0")
        self._n = jnp.asarray([c.n for c in constraints], dtype=jnp.float64)
        self._mu = jnp.asarray([c.mu for c in constraints], dtype=jnp.float64)
        self._sig = jnp.asarray([c.sigma for c in constraints], dtype=jnp.float64)

        p = _merged(eos_params)
        n_match = p["n_match_fac"] * p["n_sat"]
        above = [c for c in constraints if c.n > n_match]
        if above and warn_above_match:
            names = ", ".join(c.name or f"n={c.n:g}" for c in above)
            warnings.warn(
                f"density tag(s) above the matching density n_match = "
                f"{n_match:g} fm^-3 ({names}): above n_match the bridge EOS is "
                "beta-equilibrated-only, so this likelihood applies to the "
                "metamodel sector, not the bridge (MB-N5; see the "
                "nmbackend.lab_likelihood module docstring).",
                UserWarning, stacklevel=2)

    def psym_tagged(self, params: Optional[dict] = None):
        """P_sym at the tagged densities [MeV/fm^3], shape (n_constraints,)."""
        return psym(self._n, params)

    def ln_likelihood(self, params: Optional[dict] = None,
                      sigma_model: Optional[float] = None):
        """Fixed-sigma_model log-likelihood (normalized Gaussian sum)."""
        if sigma_model is None:
            if isinstance(self.sigma_model, SigmaModelPrior):
                raise TypeError("sigma_model is a prior; use "
                                "ln_likelihood_marginalized (or __call__)")
            sigma_model = self.sigma_model
        ps = self.psym_tagged(params)
        var = self._sig**2 + jnp.asarray(sigma_model, dtype=jnp.float64) ** 2
        return jnp.sum(_ln_normal(ps, self._mu, var))

    def ln_likelihood_marginalized(self, params: Optional[dict] = None,
                                   prior: Optional[SigmaModelPrior] = None):
        """sigma_model numerically marginalized under ``prior``.

        Defaults to the prior given at construction.  Implemented as
        logsumexp(log_weights + lnL(sigma_i)) over the prior's quadrature
        grid — differentiable w.r.t. ``params``.
        """
        if prior is None:
            if not isinstance(self.sigma_model, SigmaModelPrior):
                raise TypeError("no SigmaModelPrior given at construction; "
                                "pass prior= explicitly")
            prior = self.sigma_model
        ps = self.psym_tagged(params)
        grid = jnp.asarray(prior.grid, dtype=jnp.float64)
        logw = jnp.asarray(prior.log_weights, dtype=jnp.float64)
        var = self._sig[None, :] ** 2 + grid[:, None] ** 2   # (n_sigma, n_k)
        lnL = jnp.sum(_ln_normal(ps[None, :], self._mu[None, :], var), axis=1)
        return logsumexp(logw + lnL)

    def __call__(self, params: Optional[dict] = None):
        """Dispatch on the sigma_model given at construction."""
        if isinstance(self.sigma_model, SigmaModelPrior):
            return self.ln_likelihood_marginalized(params)
        return self.ln_likelihood(params)


# ----------------------------------------------------------------------------
# correlated (E_sym, L_sym) Gaussian — the cheap alternative encoding
# ----------------------------------------------------------------------------

class CorrelatedSymGaussian:
    """Full-covariance 2x2 Gaussian likelihood on (E_sym, L_sym) [MeV].

    The cheap parameter-space encoding (Biswas / Margueron-line style).  Keep
    in mind that it imports the model map used to extract (S0, L) from data —
    the PREX-vs-CREX discrepancy is the same skin under different maps; the
    density-tagged form above is the more honest encoding.
    """

    def __init__(self, mean, cov):
        mean = np.asarray(mean, dtype=float)
        cov = np.asarray(cov, dtype=float)
        if mean.shape != (2,) or cov.shape != (2, 2):
            raise ValueError("mean must be (2,), cov (2,2): [E_sym, L_sym]")
        if not np.allclose(cov, cov.T):
            raise ValueError("cov must be symmetric")
        if not (cov[0, 0] > 0 and np.linalg.det(cov) > 0):
            raise ValueError("cov must be positive definite")
        self.mean = jnp.asarray(mean)
        self.cov = jnp.asarray(cov)
        self._inv = jnp.asarray(np.linalg.inv(cov))
        self._logdet = float(np.log(np.linalg.det(cov)))

    @classmethod
    def from_sigmas(cls, mean_E: float, mean_L: float,
                    sigma_E: float, sigma_L: float,
                    rho: float = 0.0) -> "CorrelatedSymGaussian":
        """Build from marginal widths and correlation coefficient rho."""
        if not (-1.0 < rho < 1.0):
            raise ValueError("need -1 < rho < 1")
        off = rho * sigma_E * sigma_L
        return cls([mean_E, mean_L],
                   [[sigma_E**2, off], [off, sigma_L**2]])

    def ln_likelihood(self, params: Optional[dict] = None):
        p = _merged(params)
        x = jnp.asarray([p["E_sym"], p["L_sym"]], dtype=jnp.float64)
        r = x - self.mean
        return -0.5 * (2.0 * _LOG_2PI + self._logdet + r @ self._inv @ r)

    __call__ = ln_likelihood


# ----------------------------------------------------------------------------
# scenario-ladder input (packaged copy; provenance in the file header)
# ----------------------------------------------------------------------------

def load_scenario_ladder(path: Optional[str] = None) -> dict:
    """Load the FRIB scenario ladder (packaged copy by default).

    The packaged file ``nmbackend/data/frib_scenario_ladder.yaml`` is a COPY of
    the source of truth in nmb-papers ``scoping/frib_scenario_ladder.yaml``,
    carried with a provenance header per the FRIB-SCOPING §8 decision (no live
    cross-repo dependency).  Every number in it is an assumption.
    """
    try:
        import yaml
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("load_scenario_ladder requires pyyaml "
                          "(pip install pyyaml)") from exc
    if path is not None:
        with open(path) as f:
            return yaml.safe_load(f)
    ref = resources.files("nmbackend") / "data" / "frib_scenario_ladder.yaml"
    return yaml.safe_load(ref.read_text())


def scenario_by_name(ladder: dict, name: str) -> dict:
    """Pick one scenario entry by its ``name`` (or ``tag``)."""
    for sc in ladder["scenarios"]:
        if sc["name"] == name or sc.get("tag") == name:
            return sc
    known = [sc["name"] for sc in ladder["scenarios"]]
    raise KeyError(f"scenario {name!r} not in ladder; known: {known}")


def gaussian_from_scenario(scenario: dict, rho: float = 0.0,
                           mean_E: Optional[float] = None,
                           mean_L: Optional[float] = None,
                           ) -> Optional[CorrelatedSymGaussian]:
    """Correlated (E_sym, L_sym) Gaussian from a ladder scenario.

    Widths come from the scenario's ``lab_sigma`` (E_sym and L_sym keys; other
    keys such as K_sym follow the ad_symmetry PRIOR_SD replacement semantics
    and are not encoded here — for the +FRIB400+Ksym scenario the honest
    encoding of the curvature information is `density_tagged_from_scenario`).
    Means default to the metamodel fiducials (forecast scenarios are centered
    on the fiducial model).  The ladder carries no correlation, so rho
    defaults to 0; pass rho to explore correlated variants.  Returns None for
    a scenario with no lab constraint (e.g. GW-only).
    """
    sig = scenario.get("lab_sigma") or {}
    if "E_sym" not in sig or "L_sym" not in sig:
        if not sig:
            return None
        raise ValueError(f"scenario {scenario.get('name')!r} lab_sigma must "
                         "carry both E_sym and L_sym for the Gaussian form")
    return CorrelatedSymGaussian.from_sigmas(
        mean_E if mean_E is not None else DEFAULT_PARAMS["E_sym"],
        mean_L if mean_L is not None else DEFAULT_PARAMS["L_sym"],
        sig["E_sym"], sig["L_sym"], rho=rho)


def density_tagged_from_scenario(scenario: dict,
                                 eos_params: Optional[dict] = None,
                                 sigma_model: Union[float, SigmaModelPrior] = 0.0,
                                 warn_above_match: bool = True,
                                 ) -> Optional[DensityTaggedPressureLikelihood]:
    """Density-tagged P_sym(2 n0) likelihood from a ladder scenario.

    Uses the scenario's ``psym2n0_frac`` (fractional P_sym(2 n_sat) width,
    e.g. 0.30 for FRIB400's "P(2n0) to ~30%"): the constraint is centered on
    the fiducial metamodel's P_sym(2 n_sat) with sigma = frac * |mu| — a
    forecast injection, not a measurement.  Returns None if the scenario
    carries no ``psym2n0_frac``.

    Note: 2 n_sat sits exactly at the default matching density, so the MB-N5
    beta-equilibrium caveat applies at the boundary; the constructed
    likelihood tags n = 2 n_sat (not above), so no warning fires by default.
    """
    frac = scenario.get("psym2n0_frac")
    if frac is None:
        return None
    p = _merged(eos_params)
    n_tag = 2.0 * p["n_sat"]
    mu = float(psym(n_tag, p))
    c = DensityTaggedConstraint(
        n=n_tag, mu=mu, sigma=frac * abs(mu),
        name=f"{scenario.get('name', '?')} P_sym(2n0)",
        source="frib_scenario_ladder.yaml (ASSUMED forecast value)")
    return DensityTaggedPressureLikelihood(
        [c], sigma_model=sigma_model, warn_above_match=warn_above_match,
        eos_params=eos_params)
