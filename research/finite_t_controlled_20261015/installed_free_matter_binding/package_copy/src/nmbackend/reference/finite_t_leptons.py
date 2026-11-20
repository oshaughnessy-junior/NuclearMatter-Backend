"""CPU ideal relativistic lepton pair states at independent net number or mu.

T, total chemical potential and mass are in MeV; number densities, entropy
(k_B=1) and fixed-net-number heat capacity are in fm^-3. Energy, pressure and
free-energy density are rest-inclusive MeV fm^-3. This scalar NumPy/SciPy
reference does not provide JAX differentiation or a complete matter EOS.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numbers

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import expit

from ..constants import HBARC, M_E, M_MU

__all__ = ["LeptonState", "state_mu", "state_density", "thermal_energy_density"]
_MEASURE = 1.0 / (math.pi**2 * HBARC**3)  # two spin states


@dataclass(frozen=True)
class LeptonState:
    """One pair-equilibrated flavor; antiparticle mu is minus particle mu.

    net_density counts negatively charged leptons minus positive antiparticles;
    electric charge density is -e * net_density. No inter-flavor equilibrium or
    baryonic neutrality is imposed. Response fields hold the stated net number
    or chemical potential fixed, not the particle-only number. Susceptibility
    and dn/dT are in fm^-3 MeV^-1; dmu/dT is dimensionless.
    """
    temperature: float
    chemical_potential: float
    mass: float
    net_density: float
    particle_density: float
    antiparticle_density: float
    energy_density: float
    pressure: float
    entropy_density: float
    free_energy_density: float
    cv_fixed_net_density: float
    dn_dmu_fixed_temperature: float
    dn_dtemperature_fixed_mu: float
    dmu_dtemperature_fixed_net_density: float

    def asdict(self):
        """Return named scalar fields; units are those of this module."""
        return asdict(self)


def _real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(name + " must be a finite real scalar, not Boolean")
    try:
        value = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(name + " is not represented as a finite float64") from exc
    if not math.isfinite(value):
        raise ValueError(name + " must be finite")
    return value


def _inputs(temperature, chemical_potential, mass, rtol, tail):
    T = _real("temperature_mev", temperature)
    mu = _real("chemical_potential_mev", chemical_potential)
    m = _real("mass_mev", mass)
    rtol = _real("rtol", rtol)
    tail = _real("tail", tail)
    if T <= 0 or m < 0 or not 0 < rtol < 1e-4 or not 30 <= tail <= 100:
        raise ValueError("Require T>0, mass>=0, 0<rtol<1e-4, 30<=tail<=100")
    return T, mu, m, rtol, tail


def _integrate(fun, breaks, rtol, epsabs):
    value = None
    for lo, hi in zip(breaks[:-1], breaks[1:]):
        part, error, info = quad_vec(fun, lo, hi, epsabs=epsabs,
                                     epsrel=rtol, norm="max", limit=500,
                                     full_output=True)
        magnitude = float(np.max(np.abs(part)))
        if (not np.all(np.isfinite(part)) or not math.isfinite(float(error))
                or (not info.success and not
                    (info.status == 2 and error <= max(epsabs, rtol*magnitude)))):
            raise ArithmeticError("Unresolved finite-temperature lepton quadrature")
        value = part if value is None else value + part
    if value is None or not np.all(np.isfinite(value)):
        raise ArithmeticError("Unrepresented finite-temperature lepton integral")
    return value


def _log_expit(value):
    # NumPy logaddexp predates the supported NumPy>=1.21 minimum; the
    # scipy.special spelling was introduced after our SciPy>=1.7 minimum.
    return -np.logaddexp(0.0, -value)


def _restore(value, log_scale):
    # Combine dimensional and occupation scales before rounding to float64.
    # A subnormal/zero output has no relative-precision guarantee.
    if value == 0:
        return 0.0
    try:
        return math.copysign(math.exp(math.log(abs(value))+log_scale), value)
    except OverflowError as exc:
        raise ArithmeticError("Unrepresented finite-temperature moment") from exc


def _occupation_ratio(x, edge):
    """f(edge-x)/f(edge), x>=0, without a dilute exponential floor."""
    if edge <= 0:
        return math.exp(-x)*(1+math.exp(edge))/(1+math.exp(edge-x))
    return expit(edge-x)/expit(edge)


def _panels(end, edge):
    return sorted(set([0.0, end]+[x for x in
                       (edge-30.0, edge, edge+30.0) if 0 < x < end]))


def state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_E, *,
             rtol=2e-12, tail=60.0):
    """Evaluate one ideal pair gas at T>0 and total mu (both MeV).

    Signed mu is allowed; mu=0 retains a net-zero pair plasma. Photons,
    interactions and weak equilibrium are absent. Ordinary populations may
    underflow; no relative accuracy or exact absence is implied by such a zero.
    Unresolved susceptibility or nonfinite responses raise ArithmeticError.
    """
    T, mu, m, rtol, tail = _inputs(temperature_mev, chemical_potential_mev,
                                 mass_mev, rtol, tail)
    sign = 1.0 if mu >= 0 else -1.0
    v, u = m/T, abs(mu)/T
    edge = (abs(mu)-m)/T
    end = max(0.0, edge)+tail
    if not all(math.isfinite(x) for x in (v, u, edge, end)):
        raise ArithmeticError("Unrepresented finite-temperature energy interval")
    log_a, log_b = float(_log_expit(edge)), float(_log_expit(-u-v))
    log_pref = 3*math.log(T)+math.log(_MEASURE)
    energy_scale = max(1.0, v, u)
    response_scale = max(1.0, -edge)
    breaks = _panels(end, edge)

    def geometry(x):
        energy = v+x
        momentum = math.sqrt(x*(x+2*v))
        return energy, momentum, momentum*energy

    def bulk(pop_edge):
        def integrand(x):
            energy, momentum, measure = geometry(x)
            occupation = _occupation_ratio(x, pop_edge)
            return measure*occupation*np.array([
                1.0, energy/energy_scale,
                momentum*momentum/(3*energy*energy_scale) if energy else 0.0])
        # Zero absolute tolerance: each population is dimensionless and scaled
        # separately, including a dilute antiparticle beside a degenerate gas.
        return _integrate(integrand, _panels(max(0.0, pop_edge)+tail, pop_edge),
                          rtol, 0.0)

    particle, anti = bulk(edge), bulk(-u-v)
    npart = _restore(particle[0], log_pref+log_a)
    nanti = _restore(anti[0], log_pref+log_b)
    energy = (_restore(particle[1], log_pref+log_a+math.log(T)+math.log(energy_scale))
              + _restore(anti[1], log_pref+log_b+math.log(T)+math.log(energy_scale)))
    pressure = (_restore(particle[2], log_pref+log_a+math.log(T)+math.log(energy_scale))
                + _restore(anti[2], log_pref+log_b+math.log(T)+math.log(energy_scale)))
    ratio_b = math.exp(log_b-log_a)
    net_factor = -math.expm1(-2*u)

    def entropy_scaled(z, log_population):
        q = abs(z)
        e = math.exp(-q)
        quotient = math.log1p(e)/e if e else 1.0
        return math.exp(-q-log_population)*(quotient+q/(1+e))

    def response(x):
        _, _, measure = geometry(x)
        a = _occupation_ratio(x, edge)
        b = ratio_b*_occupation_ratio(x, -u-v)
        d = x-edge
        wa, wb = a*expit(d), b*expit(x+u+v)
        net = a/(1+math.exp(-(x+u+v)))
        if u < 0.1:
            # w(a)-w(b)=(a-b)(1-a-b); retain tiny signed mu.
            difference_w = net*net_factor*(1-expit(-d)-expit(-(x+u+v)))
            first = d*difference_w-2*u*wb
        else:
            first = d*wa-(d+2*u)*wb
        second = d*d*wa+(d+2*u)**2*wb
        entropy = (entropy_scaled(d, log_a)
                   + ratio_b*entropy_scaled(x+u+v, log_b))
        return measure*np.array([net, wa+wb, first/response_scale,
                                 second/response_scale**2, entropy/response_scale])

    net, a0, a1, a2, entropy = _integrate(response, breaks, rtol, 0.0)
    chi = _restore(a0, log_pref+log_a-math.log(T))
    if a0 <= 0 or chi <= 0:
        raise ArithmeticError("Susceptibility underflow; state outside reference domain")
    covariance = a2-a1*a1/a0
    if covariance < -1e-12*max(a2, 1e-300):
        raise ArithmeticError("Unresolved negative fixed-net-number heat capacity")
    n = _restore(net*net_factor, log_pref+log_a)
    entropy = _restore(entropy, log_pref+log_a+math.log(response_scale))
    cv = _restore(max(0.0, covariance),
                  log_pref+log_a+2*math.log(response_scale))
    n_temperature = _restore(a1, log_pref+log_a+math.log(response_scale)-math.log(T))
    result = LeptonState(T, mu, m, sign*n,
                         npart if sign > 0 else nanti,
                         nanti if sign > 0 else npart,
                         energy, pressure, entropy, abs(mu)*n-pressure,
                         cv, chi, sign*n_temperature,
                         -sign*response_scale*a1/a0)
    if not all(math.isfinite(float(x)) for x in result.asdict().values()):
        raise ArithmeticError("Nonfinite finite-temperature lepton response")
    return result


def state_density(temperature_mev, net_density_fm3, mass_mev=M_E, *,
                  rtol=2e-12, tail=60.0):
    """Invert signed net lepton NUMBER density (fm^-3), not electric charge.

    Use mass_mev=M_MU for the independent muon flavor. The scale-aware root
    enforces 1e-9 relative net-number reconstruction; exact zero has its own
    pair-plasma branch. T=0 is not supported by this reference.
    """
    T, _, m, rtol, tail = _inputs(temperature_mev, 0.0, mass_mev, rtol, tail)
    n = _real("net_density_fm3", net_density_fm3)
    if n == 0:
        return state_mu(T, 0.0, m, rtol=rtol, tail=tail)
    target = abs(n)
    pf = HBARC*(3*math.pi**2*target)**(1.0/3.0)
    upper = max(math.hypot(pf, m), T, m)
    if not math.isfinite(upper):
        raise ArithmeticError("Unrepresented net-number root bracket")
    try:
        pair = state_mu(T, 0.0, m, rtol=rtol, tail=tail)
    except ArithmeticError as exc:
        if "Susceptibility underflow" not in str(exc):
            raise
    else:
        estimate = target/pair.dn_dmu_fixed_temperature
        if 0 < estimate < min(T, upper/2):
            upper = 2*estimate
    for _ in range(64):
        if state_mu(T, upper, m, rtol=rtol, tail=tail).net_density >= target:
            break
        upper *= 2
        if not math.isfinite(upper):
            raise ArithmeticError("Unrepresented net-number root bracket")
    else:
        raise ArithmeticError("Net-number inversion failed to bracket")
    fraction = brentq(lambda x: -1.0 if x == 0 else
                      state_mu(T, upper*x, m, rtol=rtol, tail=tail).net_density/target-1,
                      0.0, 1.0, xtol=1e-13, rtol=2e-13)
    result = state_mu(T, math.copysign(upper*fraction, n), m, rtol=rtol, tail=tail)
    if abs(result.net_density/n-1) > 1e-9:
        raise ArithmeticError("Net-number inversion failed relative residual gate")
    return result


def _hole_log2_upper_bound(temperature, mass_over_temperature, cold_edge):
    """Conservative physical hole bound in binary-log units (MeV fm^-3).

    On [0,xc], geometry is monotonic and vacancy<=1, hence the hole is
    <= T^4*measure*sqrt[xc*(xc+2v)]*(v+xc)*xc^2/2. frexp bounds each
    positive factor by a power of two; exponent arithmetic is exact here.
    This avoids rounding a near-threshold logarithm toward underflow.
    """
    ec = math.frexp(cold_edge)[1]
    et = math.frexp(temperature)[1]
    em = math.frexp(_MEASURE)[1]
    if mass_over_temperature == 0:
        ea = eb = ec
    else:
        ev = math.frexp(mass_over_temperature)[1]
        ea = max(ec, ev+1)+1  # xc+2v < this power of two
        eb = max(ec, ev)+1    # xc+v < this power of two
    return 4*et+em+0.5*(ec+ea)+eb+2*ec-1


def thermal_energy_density(state, *, rtol=2e-12, tail=60.0):
    """Stable e(T,n_net)-e(0,n_net), in MeV fm^-3, for a returned state.

    This is a one-flavor fixed-net-number thermal increment, not matter heat.
    The cold reference retains rest energy; no additional mass/Q term is added.
    """
    if not isinstance(state, LeptonState):
        raise ValueError("A LeptonState returned by this module is required")
    T, _, m, rtol, tail = _inputs(state.temperature, state.chemical_potential,
                                 state.mass, rtol, tail)
    for name, value in state.asdict().items():
        _real(name, value)
    fresh = state_mu(T, state.chemical_potential, m, rtol=rtol, tail=tail)
    if ((state.net_density == 0 and fresh.net_density != 0)
            or (state.net_density != 0 and
                abs(fresh.net_density/state.net_density-1) > 1e-9)):
        raise ValueError("State chemical potential and net number are inconsistent")
    u, v = abs(state.chemical_potential)/T, m/T
    edge = (abs(state.chemical_potential)-m)/T
    pf = HBARC*(3*math.pi**2*abs(state.net_density))**(1.0/3.0)/T
    cold_edge = pf*pf/(math.hypot(pf, v)+v) if pf else 0.0
    end = max(0.0, edge, cold_edge)+tail
    if not all(math.isfinite(x) for x in (u, v, cold_edge, end)):
        raise ArithmeticError("Unrepresented thermal energy interval")
    log_pref = 4*math.log(T)+math.log(_MEASURE)

    def measure(x):
        return math.sqrt(x*(x+2*v))*(v+x)

    # The fixed-number subtraction is a sum of positive holes below the cold
    # surface, particle excitations above it, and antiparticle energy. Each
    # occupation is scaled at its own integration-domain maximum.
    holes = 0.0
    # A positive component is omitted only when a conservative dimensional
    # bound is below half the smallest float64 subnormal (2**-1075). The
    # represented excitation/antiparticle contributions are still integrated.
    if cold_edge > 0 and _hole_log2_upper_bound(T, v, cold_edge) > -1075.0:
        log_h = float(_log_expit(cold_edge-edge))
        # x=xc*t factors the tiny interval/amplitude out of the adaptive
        # rule. Its positive unit-interval kernel is O(1), even when the
        # restored physical component is a represented subnormal.
        geometry_a, geometry_b = cold_edge+2*v, cold_edge+v
        def hole_integrand(t):
            x = cold_edge*t
            vacancy = math.exp(float(_log_expit(x-edge))-log_h)
            return (math.sqrt(t)*math.sqrt((2*v+x)/geometry_a)
                    *((v+x)/geometry_b)*(1-t)*vacancy)
        breaks = sorted(set([0.0, 1.0]+[x/cold_edge for x in
                            (edge-30.0, edge, edge+30.0) if 0 < x < cold_edge]))
        holes = _restore(float(_integrate(hole_integrand, breaks, rtol, 0.0)),
                         log_pref+log_h+2.5*math.log(cold_edge)
                         +0.5*math.log(geometry_a)+math.log(geometry_b))
    log_exc = float(_log_expit(edge-cold_edge))
    def excitation(y):
        return measure(y+cold_edge)*y*_occupation_ratio(y, edge-cold_edge)
    excitation_end = end-cold_edge
    excited = _restore(float(_integrate(excitation,
                       _panels(excitation_end, edge-cold_edge), rtol, 0.0)),
                       log_pref+log_exc)
    log_anti = float(_log_expit(-u-v))
    def antiparticle(x):
        return measure(x)*(x+cold_edge+2*v)*_occupation_ratio(x, -u-v)
    anti = _restore(float(_integrate(antiparticle, _panels(tail, -u-v), rtol, 0.0)),
                    log_pref+log_anti)
    return holes+excited+anti
