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

   Information content: P_sym = n^2 dS/dn carries **no E_sym information**
   (E_sym is the constant term of S(n) and dies under d/dn) — dlnL/dE_sym is
   identically zero.  Replacing the (E_sym, L_sym) Gaussian below with this
   encoding therefore *discards* any sigma(E_sym) constraint a scenario
   carries; combine both forms if both pieces of information are wanted.

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
likelihoods drop into the Fisher / AD-Jacobian machinery unchanged.  Unknown
parameter keys are rejected (a typo'd key would otherwise silently return the
fiducial likelihood with zero gradient).

**Beta-equilibrium caveat (MB-N5) — read before tagging densities at or above
~2 n_sat.**  In the full bridge EOS (``priors/bridge.py``), everything from
the bridge matching density upward (``n_L_fac * n_sat``, default 2 n_sat) is a
nonparametric *beta-equilibrated* sector: it carries no isospin decomposition,
so a symmetric/asymmetric-matter quantity like P_sym is simply not defined
there for a bridge draw.  A density-tagged likelihood with n_k at or above the
matching density therefore applies to the **metamodel sector only** (the
analytic expansion extrapolated past its match point), *not* to the bridge —
it constrains bridge draws only indirectly, through the sub-2 n_sat anchor and
continuity.  Quantifying how much of a P_sym(2 n0) likelihood survives that
beta-eq projection is exactly the open "isospin map" question (MB-N5); this
module warns when you cross the line but does not resolve it.  The default
warning threshold is the metamodel's own ``n_match_fac * n_sat``; if you draw
bridges with a non-default ``n_L_fac``, pass the true boundary explicitly as
``n_beta_eq`` — the two knobs are independent and only coincide at defaults.

Scenario input: `load_scenario_ladder()` reads the packaged **copy** of the
FRIB scenario ladder (``data/frib_scenario_ladder.yaml``).  The source of
truth lives in nmb-papers ``scoping/frib_scenario_ladder.yaml``; the copy
carries a provenance header, and every number in it is an *assumption* — no
published paper commits to "sigma_L = +/-X by year Y" for FRIB.

Downstream consumers: the p5-crossover Tier-B forecast (claim C3 encoding
comparison) and the paper-1 EM-likelihood robustness variant in nmb-papers.

Motivated by: Huth et al. Nature 606 (2022) 276 (arXiv:2107.06229) and
Tsang et al. (arXiv:2310.11588) — pressure-at-density likelihood encodings
re-implemented from the papers, no code reused; (S0, L) Gaussian priors after
Biswas et al. / the Margueron metamodel line.  See ATTRIBUTION.md.
"""
from __future__ import annotations

import math
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

def _merged(params: Optional[dict], base: Optional[dict] = None) -> dict:
    """Merge ``params`` over ``base`` (default: metamodel DEFAULT_PARAMS).

    Unknown keys raise: a typo'd NEP key ("Lsym") would otherwise be silently
    ignored, returning the fiducial likelihood with an exactly-zero gradient —
    the easiest way to produce a wrong-but-plausible robustness number.
    """
    p = dict(DEFAULT_PARAMS if base is None else base)
    if params:
        unknown = sorted(set(params) - set(DEFAULT_PARAMS))
        if unknown:
            raise ValueError(
                f"unknown EOS parameter key(s) {unknown}; known keys: "
                f"{sorted(DEFAULT_PARAMS)}")
        p.update(params)
    return p


def _psym_of(n_arr, p: dict):
    """P_sym on a 1D array for a fully merged parameter dict."""
    dS_dn = jax.vmap(jax.grad(lambda nn: _esat_S(nn, p)[1]))(n_arr)
    return n_arr**2 * dS_dn


def psym(n, params: Optional[dict] = None):
    """Symmetry pressure P_sym(n) = n^2 dS/dn [MeV/fm^3] of the metamodel.

    ``n`` is baryon density in fm^-3 (scalar or array of any rank; the output
    has the input's shape); ``params`` overrides
    ``eos.metamodel.DEFAULT_PARAMS`` (E_sym, L_sym, K_sym, Q_sym, n_sat, ...) —
    unknown keys raise.  Differentiable w.r.t. the NEPs.  dS/dn is taken with
    ``jax.grad`` of the same symmetry-energy expansion the EOS table is built
    from, so this cannot drift from the EOS.  At n = n_sat it equals
    n_sat * L_sym / 3.
    """
    p = _merged(params)
    n_arr = jnp.asarray(n, dtype=jnp.float64)
    return _psym_of(n_arr.ravel(), p).reshape(n_arr.shape)


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
    name : str       label (e.g. "SpiRIT pions", "FRIB400 P_sym(2n0)")
    source : str     provenance note (arXiv id / scenario ladder entry)
    """
    n: float
    mu: float
    sigma: float
    name: str = ""
    source: str = ""

    def __post_init__(self):
        for attr in ("n", "mu", "sigma"):
            if not math.isfinite(getattr(self, attr)):
                raise ValueError(f"constraint {self.name!r}: {attr} must be finite")
        if not (self.n > 0.0):
            raise ValueError(f"constraint {self.name!r}: density must be > 0")
        # sigma**2 must survive squaring in float64 (sigma ~ 1e-200 would give
        # var = 0 and a NaN likelihood downstream)
        if not (self.sigma > 0.0 and self.sigma**2 > 0.0):
            raise ValueError(f"constraint {self.name!r}: sigma must be > 0 "
                             "(and sigma**2 must not underflow)")


@dataclass(frozen=True)
class SigmaModelPrior:
    """Numerical prior over the transport-model systematic floor sigma_model.

    Represented as quadrature nodes ``grid`` (sigma values, MeV/fm^3) with
    ``log_weights`` (prior pdf x quadrature weight).  Weights are validated and
    normalized (so logsumexp(log_weights) = 0) on construction — including
    direct ``SigmaModelPrior(grid, log_weights)`` construction, so a
    hand-built prior cannot silently offset the evidence.  Marginalization is
    then a logsumexp — exact for a delta prior, trapezoid-accurate otherwise
    (default n=129 is accurate to ~3e-6 in lnL in the worst realistic case),
    and differentiable w.r.t. the EOS parameters.
    """
    grid: tuple = field(repr=False)
    log_weights: tuple = field(repr=False)
    label: str = ""

    def __post_init__(self):
        grid = np.asarray(self.grid, dtype=float)
        logw = np.asarray(self.log_weights, dtype=float)
        if grid.ndim != 1 or grid.shape != logw.shape or grid.size < 1:
            raise ValueError("grid and log_weights must be equal-length 1D")
        if not np.all(np.isfinite(grid)) or np.any(grid < 0.0):
            raise ValueError("sigma_model support must be finite and >= 0")
        if np.any(np.isnan(logw)):
            raise ValueError("log_weights must not contain NaN")
        m = np.max(logw)
        if not np.isfinite(m):
            raise ValueError("log_weights must contain at least one finite entry")
        logw = logw - (m + np.log(np.sum(np.exp(logw - m))))
        object.__setattr__(self, "grid", tuple(grid))
        object.__setattr__(self, "log_weights", tuple(logw))

    @classmethod
    def _trapezoid(cls, grid, logpdf_vals, label) -> "SigmaModelPrior":
        """Trapezoid rule on a uniform grid; the single quadrature site."""
        w = np.full(len(grid), 1.0)
        w[0] *= 0.5
        w[-1] *= 0.5
        return cls(tuple(grid), tuple(np.asarray(logpdf_vals) + np.log(w)), label)

    @classmethod
    def fixed(cls, value: float) -> "SigmaModelPrior":
        """Delta prior — marginalization reduces exactly to the fixed form."""
        return cls((value,), (0.0,), f"fixed({value})")

    @classmethod
    def uniform(cls, lo: float, hi: float, n: int = 129) -> "SigmaModelPrior":
        if not (0.0 <= lo < hi):
            raise ValueError("need 0 <= lo < hi")
        if n < 2:
            raise ValueError("need n >= 2 quadrature nodes (n=1 would silently "
                             "collapse to a delta at lo; use .fixed() for a delta)")
        g = np.linspace(lo, hi, n)
        return cls._trapezoid(g, np.zeros(n), f"uniform[{lo},{hi}]")

    @classmethod
    def half_normal(cls, scale: float, n: int = 129,
                    n_scales: float = 5.0) -> "SigmaModelPrior":
        """Half-normal on sigma_model >= 0 (truncated at n_scales*scale)."""
        if not scale > 0.0:
            raise ValueError("scale must be > 0")
        if n < 2:
            raise ValueError("need n >= 2 quadrature nodes (use .fixed() for a delta)")
        g = np.linspace(0.0, n_scales * scale, n)
        return cls._trapezoid(g, -0.5 * (g / scale) ** 2, f"half_normal({scale})")

    @classmethod
    def from_logpdf(cls, logpdf, lo: float, hi: float,
                    n: int = 129, label: str = "custom") -> "SigmaModelPrior":
        """Arbitrary (unnormalized) logpdf callable on [lo, hi]."""
        if not lo < hi:
            raise ValueError("need lo < hi")
        if n < 2:
            raise ValueError("need n >= 2 quadrature nodes (use .fixed() for a delta)")
        g = np.linspace(lo, hi, n)
        lp = np.array([float(logpdf(s)) for s in g])
        return cls._trapezoid(g, lp, label)


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
    dict passed to ``__call__`` / ``ln_likelihood`` (unknown keys raise).

    ``eos_params`` is the **fiducial** parameter set: it becomes the base the
    per-call ``params`` are merged over, so a likelihood built at a non-default
    fiducial evaluates at that fiducial by default.

    Beta-equilibrium caveat (MB-N5): constraints tagged at or above the bridge
    matching density apply to the **metamodel sector only**, not to the
    beta-equilibrated nonparametric bridge — see the module docstring.
    Construction warns when a tag reaches that boundary (silence with
    ``warn_above_match=False`` once acknowledged).  The default threshold is
    the metamodel's ``n_match_fac * n_sat``; when drawing bridges with a
    non-default ``n_L_fac``, pass the true boundary as ``n_beta_eq`` — the two
    knobs are independent and only coincide at defaults.
    """

    def __init__(self,
                 constraints: Sequence[DensityTaggedConstraint],
                 sigma_model: Union[float, SigmaModelPrior] = 0.0,
                 warn_above_match: bool = True,
                 eos_params: Optional[dict] = None,
                 n_beta_eq: Optional[float] = None):
        constraints = tuple(constraints)
        if len(constraints) == 0:
            raise ValueError("need at least one constraint")
        self.constraints = constraints
        if isinstance(sigma_model, SigmaModelPrior):
            self.sigma_model = sigma_model
            self._prior_grid = jnp.asarray(sigma_model.grid, dtype=jnp.float64)
            self._prior_logw = jnp.asarray(sigma_model.log_weights,
                                           dtype=jnp.float64)
        else:
            sm = float(sigma_model)      # rejects SigmaModelPrior-like misuse,
            if not (math.isfinite(sm) and sm >= 0.0):   # catches np/jnp scalars
                raise ValueError("sigma_model must be finite and >= 0")
            self.sigma_model = sm
            self._prior_grid = self._prior_logw = None
        self._n = jnp.asarray([c.n for c in constraints], dtype=jnp.float64)
        self._mu = jnp.asarray([c.mu for c in constraints], dtype=jnp.float64)
        self._sig = jnp.asarray([c.sigma for c in constraints], dtype=jnp.float64)

        self.eos_params = _merged(eos_params)
        if n_beta_eq is None:
            n_beta_eq = self.eos_params["n_match_fac"] * self.eos_params["n_sat"]
        self.n_beta_eq = float(n_beta_eq)
        at_or_above = [c for c in constraints if c.n >= self.n_beta_eq]
        if at_or_above and warn_above_match:
            names = ", ".join(c.name or f"n={c.n:g}" for c in at_or_above)
            warnings.warn(
                f"density tag(s) at or above the matching density n_beta_eq = "
                f"{self.n_beta_eq:g} fm^-3 ({names}): from there up the bridge "
                "EOS is beta-equilibrated-only, so this likelihood applies to "
                "the metamodel sector, not the bridge (MB-N5; see the "
                "nmbackend.lab_likelihood module docstring).  If your bridge "
                "uses a non-default n_L_fac, pass the true boundary as "
                "n_beta_eq=.  Silence with warn_above_match=False.",
                UserWarning, stacklevel=2)

    def psym_tagged(self, params: Optional[dict] = None):
        """P_sym at the tagged densities [MeV/fm^3], shape (n_constraints,).

        ``params`` are merged over the construction-time fiducial
        ``eos_params`` (unknown keys raise).
        """
        return _psym_of(self._n, _merged(params, base=self.eos_params))

    def ln_likelihood(self, params: Optional[dict] = None,
                      sigma_model: Optional[float] = None):
        """Fixed-sigma_model log-likelihood (normalized Gaussian sum)."""
        if isinstance(sigma_model, SigmaModelPrior):
            raise TypeError("got a SigmaModelPrior; use "
                            "ln_likelihood_marginalized(prior=...) to marginalize")
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
            grid, logw = self._prior_grid, self._prior_logw
        elif isinstance(prior, SigmaModelPrior):
            grid = jnp.asarray(prior.grid, dtype=jnp.float64)
            logw = jnp.asarray(prior.log_weights, dtype=jnp.float64)
        else:
            raise TypeError(f"prior must be a SigmaModelPrior, got "
                            f"{type(prior).__name__}; for a fixed sigma_model "
                            "use ln_likelihood(sigma_model=...)")
        ps = self.psym_tagged(params)
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
        if not (np.all(np.isfinite(mean)) and np.all(np.isfinite(cov))):
            raise ValueError("mean and cov must be finite")
        if not np.allclose(cov, cov.T):
            raise ValueError("cov must be symmetric")
        sign, logdet = np.linalg.slogdet(cov)
        if not (sign > 0 and np.all(np.diag(cov) > 0)):
            raise ValueError("cov must be positive definite")
        self.mean = jnp.asarray(mean)
        self.cov = jnp.asarray(cov)
        self._inv = jnp.asarray(np.linalg.inv(cov))
        self._logdet = float(logdet)

    @classmethod
    def from_sigmas(cls, mean_E: float, mean_L: float,
                    sigma_E: float, sigma_L: float,
                    rho: float = 0.0) -> "CorrelatedSymGaussian":
        """Build from marginal widths and correlation coefficient rho."""
        if not (sigma_E > 0.0 and sigma_L > 0.0):
            # a negative sigma would survive squaring and silently flip the
            # sign of the built correlation
            raise ValueError("sigma_E and sigma_L must be > 0")
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
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    ref = resources.files("nmbackend") / "data" / "frib_scenario_ladder.yaml"
    return yaml.safe_load(ref.read_text(encoding="utf-8"))


def scenario_by_name(ladder: dict, name: str) -> dict:
    """Pick one scenario entry by its ``name`` (or ``tag``)."""
    for sc in ladder["scenarios"]:
        if sc["name"] == name or sc.get("tag") == name:
            return sc
    known = [f"{sc['name']} (tag {sc.get('tag')})" for sc in ladder["scenarios"]]
    raise KeyError(f"scenario {name!r} not in ladder; known: {known}")


def gaussian_from_scenario(scenario: dict, rho: float = 0.0,
                           mean_E: Optional[float] = None,
                           mean_L: Optional[float] = None,
                           eos_params: Optional[dict] = None,
                           ) -> Optional[CorrelatedSymGaussian]:
    """Correlated (E_sym, L_sym) Gaussian from a ladder scenario.

    Widths come from the scenario's ``lab_sigma`` (E_sym and L_sym keys).  Any
    other lab_sigma key (e.g. K_sym in +FRIB400+Ksym) **cannot be encoded in
    the 2x2 form and is dropped with a UserWarning** — in this encoding the
    +FRIB400 and +FRIB400+Ksym rungs are identical, so an encoding-comparison
    sweep must not read that as "K_sym information buys nothing"; use
    `density_tagged_from_scenario` for the curvature information.

    Means default to the fiducial model (``eos_params`` merged over the
    metamodel defaults — pass the same fiducial you give the density-tagged
    builder so the claim-C3 encoding comparison is centered consistently);
    ``mean_E``/``mean_L`` override individually.  The ladder carries no
    correlation, so rho defaults to 0; pass rho to explore correlated
    variants.  Returns None for a scenario with no lab constraint (GW-only).
    """
    sig = scenario.get("lab_sigma") or {}
    if not sig:
        return None
    if "E_sym" not in sig or "L_sym" not in sig:
        raise ValueError(f"scenario {scenario.get('name')!r} lab_sigma must "
                         "carry both E_sym and L_sym for the Gaussian form")
    dropped = sorted(set(sig) - {"E_sym", "L_sym"})
    if dropped:
        warnings.warn(
            f"scenario {scenario.get('name')!r}: lab_sigma key(s) {dropped} "
            "cannot be encoded in the 2x2 (E_sym, L_sym) Gaussian and are "
            "DROPPED — this encoding is then blind to that information "
            "(use density_tagged_from_scenario for the curvature content).",
            UserWarning, stacklevel=2)
    p = _merged(eos_params)
    return CorrelatedSymGaussian.from_sigmas(
        mean_E if mean_E is not None else p["E_sym"],
        mean_L if mean_L is not None else p["L_sym"],
        sig["E_sym"], sig["L_sym"], rho=rho)


def density_tagged_from_scenario(scenario: dict,
                                 eos_params: Optional[dict] = None,
                                 sigma_model: Union[float, SigmaModelPrior] = 0.0,
                                 warn_above_match: bool = True,
                                 n_beta_eq: Optional[float] = None,
                                 ) -> Optional[DensityTaggedPressureLikelihood]:
    """Density-tagged P_sym(2 n0) likelihood from a ladder scenario.

    Uses the scenario's ``psym2n0_frac`` (fractional width, e.g. 0.30 for
    FRIB400's "P(2n0) to ~30%"): the constraint is centered on the fiducial
    model's P_sym(2 n_sat) with sigma = frac * mu — a forecast injection, not
    a measurement.  Returns None if the scenario carries no ``psym2n0_frac``.

    Interpretation note: the ladder's "P(2n0)" is applied here to the
    **symmetry pressure** P_sym = n^2 dS/dn, not to the neutron-matter
    pressure P_PNM = n^2 (de_sat/dn + dS/dn); the two differ at 2 n_sat by
    the isoscalar term (they coincide only at n_sat).  This is an encoding
    choice carried with the ladder copy's provenance header.

    Usage: **construct once at the fiducial, then evaluate with per-draw
    parameters** (``like({"L_sym": ...})``).  The fiducial ``eos_params`` is
    stored, so evaluation defaults to it and the fiducial sits at the peak.
    Do NOT rebuild with each draw's NEPs as ``eos_params`` — that re-centers
    the forecast on every draw, producing a pseudo-likelihood that
    systematically rewards small-|P_sym| draws.

    Raises if the fiducial P_sym(2 n_sat) is not positive: a fractional-width
    forecast is ill-defined there (with default K_sym = -100 this happens for
    L_sym below about 33 MeV — near the zero-crossing the width collapses
    toward a delta, which would silently over-tighten the posterior).

    MB-N5: the 2 n_sat tag sits exactly at the default beta-equilibrium
    boundary, so a UserWarning fires — the caveat genuinely applies to this
    constraint.  Silence with ``warn_above_match=False`` once acknowledged.
    """
    frac = scenario.get("psym2n0_frac")
    if frac is None:
        return None
    frac = float(frac)
    if not (0.0 < frac):
        raise ValueError(f"psym2n0_frac must be > 0, got {frac}")
    p = _merged(eos_params)
    n_tag = 2.0 * p["n_sat"]
    mu = float(psym(n_tag, p))
    if not mu > 0.0:
        raise ValueError(
            f"fiducial P_sym(2 n_sat) = {mu:g} MeV/fm^3 is not positive for "
            "these eos_params, so a fractional-width forecast (sigma = "
            f"frac * mu) is ill-defined; choose a fiducial with positive "
            "P_sym(2 n_sat) or build the DensityTaggedConstraint directly "
            "with an absolute sigma.")
    c = DensityTaggedConstraint(
        n=n_tag, mu=mu, sigma=frac * mu,
        name=f"{scenario.get('name', '?')} P_sym(2n0)",
        source="frib_scenario_ladder.yaml (ASSUMED forecast value)")
    return DensityTaggedPressureLikelihood(
        [c], sigma_model=sigma_model, warn_above_match=warn_above_match,
        eos_params=eos_params, n_beta_eq=n_beta_eq)
