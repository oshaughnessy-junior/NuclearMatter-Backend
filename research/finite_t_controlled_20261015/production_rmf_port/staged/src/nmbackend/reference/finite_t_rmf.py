"""Scalar CPU reference for the exact rounded GM1 uniform nuclear Hartree model.

Public T/fields/mu in MeV; n,s,Cv in fm^-3; e,P,f in MeV fm^-3. Thermal
antibaryons retained, vacuum sea/leptons/photons/weak/beta/AD/phase coexistence absent.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numbers
import numpy as np
from scipy.optimize import brentq, least_squares
from ..constants import HBARC
from ._rmf_moments import moments, projected_number_scalar
from ._rmf_response import constraint_jacobian, response

__all__ = ["SolveControls", "UniformRMFState", "state_density", "diagnose_density"]
_M = 939./HBARC
_CS, _CO, _CR, _B, _C = 11.785, 7.148, 4.410, .002947, -.001070


def _real(name, x):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, numbers.Real):
        raise ValueError(name+" must be a finite non-Boolean scalar")
    try:
        x = float(x)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name+" is not a represented finite float64 scalar") from exc
    if not math.isfinite(x):
        raise ValueError(name+" must be finite")
    return x


@dataclass(frozen=True)
class SolveControls:
    rtol: float = 2e-11
    tail: float = 60.

    def __post_init__(self):
        r = _real("rtol", self.rtol)
        t = _real("tail", self.tail)
        if not 0 < r < 1e-4 or not 30 <= t <= 100:
            raise ValueError("Require 0<rtol<1e-4, 30<=tail<=100")


@dataclass(frozen=True)
class UniformRMFState:
    temperature: float
    baryon_density: float
    proton_fraction: float
    neutron_density: float
    proton_density: float
    phi: float
    omega: float
    rho: float
    dirac_mass: float
    kinetic_mu_n: float
    kinetic_mu_p: float
    mu_n: float
    mu_p: float
    energy_density: float
    pressure: float
    free_energy_density: float
    entropy_density: float
    cv_fixed_density_composition: float
    kinetic_energy_density: float
    scalar_field_energy_density: float
    vector_field_energy_density: float
    scalar_density: float
    hessian_n_p: tuple
    scalar_curvature_fm_minus2: float
    response_condition: float
    hessian_min_eigenvalue: float
    normalized_equation_residual: float
    euler_residual: float
    phi_temperature_derivative: float
    antiparticle_density_n: object
    antiparticle_density_p: object
    log_antiparticle_density_n: float
    log_antiparticle_density_p: float
    locally_admitted: bool
    local_label: str
    cold_scalar_bracket: tuple
    warm_scalar_bracket: tuple
    branch: str = "unique scanned canonical upcrossing, connected from cold"
    global_phase: str = "unassessed; restricted uniform branch, no Gibbs/Coulomb construction"
    model: str = "NMB-rounded GM1: common M939; Cs11.785 Co7.148 Cr4.410 b.002947 c-.001070; thermal pairs no vacuum"

    def asdict(self):
        return asdict(self)

    @property
    def density_defined_landau_n(self):
        return math.hypot(HBARC*(3*math.pi**2*self.neutron_density)**(1/3), self.dirac_mass)

    @property
    def density_defined_landau_p(self):
        return math.hypot(HBARC*(3*math.pi**2*self.proton_density)**(1/3), self.dirac_mass)


def _potential(phi):
    return phi*phi/(2*_CS)+_B*_M*phi**3/3+_C*phi**4/4


def _gradient(phi):
    return phi/_CS+_B*_M*phi*phi+_C*phi**3


def _cold_scalar(density, mass):
    p = (3*math.pi**2*density)**(1/3)
    e = math.hypot(p, mass)
    return mass*(p*e-mass*mass*math.asinh(p/mass))/(2*math.pi**2)


def _crossing(fun, intervals):
    grid = np.linspace(0., .985*_M, intervals+1)
    vals = [fun(float(x)) for x in grid]
    if not all(math.isfinite(v) for v in vals):
        raise ArithmeticError("Nonfinite scalar scan")
    brackets = [(float(a), float(b)) for a, b, fa, fb in
                zip(grid[:-1], grid[1:], vals[:-1], vals[1:]) if fa < 0 <= fb]
    if len(brackets) != 1:
        raise ArithmeticError("No unique scalar upcrossing on declared positive-mass interval")
    lo, hi = brackets[0]
    return brentq(fun, lo, hi, xtol=2e-13, rtol=2e-14), (lo, hi)


def _number_root(t, density, mass, controls):
    def fun(nu):
        return (-1. if nu == 0. else projected_number_scalar(t, nu, mass, rtol=controls.rtol, tail=controls.tail)[0]/density-1)
    hi = max(2*_M, math.hypot((3*math.pi**2*density)**(1/3), mass)+10*t)
    if fun(0.) >= 0 or fun(hi) <= 0:
        raise ArithmeticError("Number root not bracketed")
    nu = brentq(fun, 0., hi, xtol=2e-13, rtol=2e-14)
    actual, scalar = projected_number_scalar(t, nu, mass, rtol=controls.rtol, tail=controls.tail)
    if abs(actual/density-1) > 2e-10:
        raise ArithmeticError("Net number root unresolved")
    return nu, scalar


def _seed(initial):
    if initial is None:
        return
    if not isinstance(initial, UniformRMFState) or initial.model != UniformRMFState.__dataclass_fields__["model"].default:
        raise ValueError("Initial seed must be an actual matching GM1 state")
    # Seeds are not trusted state shortcuts: all roots/branch checks are recomputed.
    for name in ("phi", "kinetic_mu_n", "kinetic_mu_p"):
        _real("initial."+name, getattr(initial, name))


def diagnose_density(temperature_mev, baryon_density_fm3, proton_fraction, *,
                     controls=SolveControls(), initial=None):
    """Return labelled stationary local branch; spinodals are explicitly unadmitted.

    Numerical request bounds T1..50,nB.02...4,Yp.05...5 are not a uniform
    physical accuracy certificate. No-root/ambiguous/ill-conditioned solves raise.
    """
    T, nB, Y = [_real(n, v) for n, v in [("T", temperature_mev),
        ("nB", baryon_density_fm3), ("Yp", proton_fraction)]]
    if not 1 <= T <= 50 or not .02 <= nB <= .4 or not .05 <= Y <= .5:
        raise ValueError("Outside selected numerical request bounds T1..50,nB.02...4,Yp.05...5")
    if not isinstance(controls, SolveControls):
        raise ValueError("SolveControls required")
    controls.__post_init__()
    _seed(initial)
    nn, np_ = (1-Y)*nB, Y*nB
    cold_fun = lambda phi: _gradient(phi)-_cold_scalar(nn, _M-phi)-_cold_scalar(np_, _M-phi)
    phi, cold_bracket = _crossing(cold_fun, 128)
    def cold_nu(d):
        return math.hypot((3*math.pi**2*d)**(1/3), _M-phi)
    z = np.array([phi, cold_nu(nn), cold_nu(np_)])
    # A caller seed deliberately never replaces the cold-connected continuation.
    residual = float("inf")
    for temp in np.linspace(T/math.ceil(T/5), T, math.ceil(T/5)):
        t = float(temp)/HBARC
        scale = np.array([nB, nn, np_])
        def evaluate(v):
            a, b, c = map(float, v)
            gn = moments(t, b, _M-a, rtol=controls.rtol, tail=controls.tail)
            gp = moments(t, c, _M-a, rtol=controls.rtol, tail=controls.tail)
            F = np.array([_gradient(a)-gn.scalar-gp.scalar, gn.density-nn, gp.density-np_])/scale
            J = constraint_jacobian(a, gn, gp, _M, _CS, _B, _C)/scale[:, None]
            return F, J, gn, gp
        sol = least_squares(lambda v: evaluate(v)[0], z, jac=lambda v: evaluate(v)[1],
            bounds=([0., 0., 0.], [.985*_M, 8*_M, 8*_M]),
            xtol=2e-12, ftol=2e-12, gtol=2e-12, max_nfev=100)
        F, J, gn, gp = evaluate(sol.x)
        residual = float(np.max(abs(F)))
        if not sol.success or residual > 2e-9:
            raise ArithmeticError("Unresolved three-constraint warm continuation")
        z = sol.x
        rr = response(float(z[0]), gn, gp, _M, _CS, _CO, _CR, _B, _C, nB, Y, HBARC)
        if rr["D"] <= 1e-8 or rr["condition"] >= 1e8:
            raise ArithmeticError("Cold-connected scalar branch not numerically admissible")
    phi, nun, nup = map(float, z)
    t = T/HBARC
    def warm_fun(ph):
        mn = _M-ph
        return _gradient(ph)-_number_root(t, nn, mn, controls)[1]-_number_root(t, np_, mn, controls)[1]
    scan_phi, warm_bracket = _crossing(warm_fun, 48)
    if abs(scan_phi-phi)*HBARC > 2e-7:
        raise ArithmeticError("Warm nested scalar root disagrees with cold-connected branch")
    mass = _M-phi
    gn = moments(t, nun, mass, rtol=controls.rtol, tail=controls.tail)
    gp = moments(t, nup, mass, rtol=controls.rtol, tail=controls.tail)
    rr = response(phi, gn, gp, _M, _CS, _CO, _CR, _B, _C, nB, Y, HBARC)
    H = rr["H"]
    asym = float(np.linalg.norm(H-H.T)/max(1., np.linalg.norm(H)))
    if asym > 1e-9 or rr["condition"] >= 1e8:
        raise ArithmeticError("Chemical response unresolved/asymmetric")
    cv = t*rr["entropy_t"]
    if not math.isfinite(cv) or cv <= 0 or rr["D"] <= 1e-8:
        raise ArithmeticError("Nonpositive canonical response")
    eigen = float(np.linalg.eigvalsh(H)[0])
    margin = 1e-7*max(1., float(np.linalg.norm(H, 2)))
    admitted = eigen > margin
    label = "locally convex nuclear branch" if admitted else ("nuclear spinodal" if eigen < -margin else "marginal unadmitted nuclear branch")
    W, R = _CO*nB, .5*_CR*(2*Y-1)*nB
    uv = .5*_CO*nB*nB+_CR*((2*Y-1)*nB)**2/8
    us = _potential(phi)
    ek = gn.energy+gp.energy
    entropy = gn.entropy+gp.entropy
    energy = (ek+us+uv)*HBARC
    pressure = (gn.pressure+gp.pressure-us+uv)*HBARC
    mun, mup = (nun+W-R/2)*HBARC, (nup+W+R/2)*HBARC
    euler = energy+pressure-T*entropy-mun*nn-mup*np_
    if abs(euler) > 2e-9*max(1., abs(energy), abs(pressure)):
        raise ArithmeticError("Euler/decomposition forward gate unresolved")
    return UniformRMFState(T, nB, Y, gn.density, gp.density, phi*HBARC, W*HBARC,
        R*HBARC, mass*HBARC, nun*HBARC, nup*HBARC, mun, mup, energy, pressure,
        energy-T*entropy, entropy, cv, ek*HBARC, us*HBARC, uv*HBARC,
        gn.scalar+gp.scalar, tuple(tuple(float(x) for x in row) for row in H),
        rr["D"], rr["condition"], eigen, residual, euler, rr["phi_t"],
        gn.antiparticle_density, gp.antiparticle_density,
        gn.log_antiparticle_density, gp.log_antiparticle_density,
        bool(admitted), label, cold_bracket, warm_bracket)


def state_density(temperature_mev, baryon_density_fm3, proton_fraction, *,
                  controls=SolveControls(), initial=None):
    """Require a numerically resolved locally convex nuclear-only GM1 branch."""
    s = diagnose_density(temperature_mev, baryon_density_fm3, proton_fraction,
                         controls=controls, initial=initial)
    if not s.locally_admitted:
        raise ValueError("Physical local admission failed: "+s.local_label)
    return s
