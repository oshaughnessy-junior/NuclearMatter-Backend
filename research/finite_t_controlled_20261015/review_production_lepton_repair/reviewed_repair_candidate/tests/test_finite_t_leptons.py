"""Independent analytic limits and original-momentum reference for pair states."""
import dataclasses
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit

from nmbackend.constants import HBARC, M_E, M_MU
from nmbackend.reference.finite_t_leptons import (
    LeptonState, state_mu, state_density, thermal_energy_density,
)


@pytest.mark.parametrize("T,mu", [(1., 0.), (10., 3.), (10., -3.), (.01, 100.)])
def test_massless_exact_forward_and_responses(T, mu):
    s = state_mu(T, mu, mass_mev=0.)
    h3 = HBARC**3
    p = (mu**4/(12*math.pi**2)+mu*mu*T*T/6+7*math.pi**2*T**4/180)/h3
    n = (mu**3/(3*math.pi**2)+mu*T*T/3)/h3
    entropy = (mu*mu*T/3+7*math.pi**2*T**3/45)/h3
    chi = (mu*mu/math.pi**2+T*T/3)/h3
    nT = 2*mu*T/(3*h3)
    cv = T*((mu*mu/3+7*math.pi**2*T*T/15)/h3-nT*nT/chi)
    np.testing.assert_allclose([s.pressure, s.energy_density, s.net_density,
                                s.entropy_density, s.dn_dmu_fixed_temperature,
                                s.dn_dtemperature_fixed_mu, s.cv_fixed_net_density],
                               [p, 3*p, n, entropy, chi, nT, cv],
                               rtol=2e-8, atol=1e-30)


def momentum_reference(T, mu, mass):
    """Scalar infinite original-p integral, unlike the production k=E-m rule."""
    scale = max(T, math.sqrt(2*mass*T))
    pref = scale**3/(math.pi**2*HBARC**3)
    def one(moment):
        def integrand(x):
            p = scale*x
            E = math.hypot(p, mass)
            a, b = expit((mu-E)/T), expit((-mu-E)/T)
            vals = [a, b, E*(a+b), p*p/(3*E)*(a+b) if E else 0.]
            return x*x*vals[moment]
        return pref*quad(integrand, 0, np.inf, epsabs=1e-16,
                         epsrel=3e-11, limit=200)[0]
    return np.array([one(i) for i in range(4)])


@pytest.mark.parametrize("T,mu,m", [(1., 2., M_E), (10., 0., M_E),
                                     (2., -1., M_E), (20., -100., M_MU),
                                     (.1, 110., M_MU)])
def test_independent_momentum_forward(T, mu, m):
    s = state_mu(T, mu, m)
    expected = momentum_reference(T, mu, m)
    np.testing.assert_allclose([s.particle_density, s.antiparticle_density,
                                s.energy_density, s.pressure], expected,
                               rtol=3e-8, atol=1e-27)
    np.testing.assert_allclose(s.net_density, expected[0]-expected[1],
                               rtol=3e-8, atol=1e-27)


def test_nonrelativistic_maxwell_boltzmann():
    T, m, eta = 1e-4, M_MU, -15.
    s = state_mu(T, m+eta*T, m)
    n = 2*(m*T/(2*math.pi))**1.5*math.exp(eta)/HBARC**3
    np.testing.assert_allclose(s.net_density, n, rtol=3e-6)
    np.testing.assert_allclose(s.pressure/(s.net_density*T), 1., rtol=2e-6)
    np.testing.assert_allclose(s.cv_fixed_net_density/s.net_density, 1.5, rtol=5e-6)


def test_massive_cold_sommerfeld_and_stable_thermal_energy():
    T, mu, m = .001, 180., M_MU
    s = state_mu(T, mu, m)
    p = math.sqrt(mu*mu-m*m)
    n0 = p**3/(3*math.pi**2*HBARC**3)
    e0 = (p*mu*(2*p*p+m*m)-m**4*math.log((p+mu)/m))/(8*math.pi**2*HBARC**3)
    p0 = (p*mu*(2*p*p-3*m*m)+3*m**4*math.log((p+mu)/m))/(24*math.pi**2*HBARC**3)
    np.testing.assert_allclose([s.net_density, s.energy_density, s.pressure],
                               [n0, e0, p0], rtol=5e-9)
    np.testing.assert_allclose(s.cv_fixed_net_density/(n0*T),
                               math.pi**2*mu/(p*p), rtol=1e-7)
    np.testing.assert_allclose(thermal_energy_density(s),
                               s.cv_fixed_net_density*T/2, rtol=1e-7)


@pytest.mark.parametrize("T,n,m", [(10., 1e-6, M_E), (.001, .01, M_E),
                                    (20., .005, M_MU), (.01, .002, M_MU),
                                    (.1, -1e-12, M_E)])
def test_density_inversion_and_thermodynamic_identities(T, n, m):
    s = state_density(T, n, m)
    np.testing.assert_allclose(s.net_density, n, rtol=1e-9)
    np.testing.assert_allclose(s.energy_density+s.pressure,
                               T*s.entropy_density+s.chemical_potential*n, rtol=2e-9)
    np.testing.assert_allclose(s.free_energy_density,
                               s.chemical_potential*n-s.pressure, rtol=2e-9)
    fine = state_mu(T, s.chemical_potential, m, rtol=5e-13, tail=80.)
    for key in ["energy_density", "pressure", "entropy_density",
                "cv_fixed_net_density", "dn_dmu_fixed_temperature"]:
        np.testing.assert_allclose(getattr(s, key), getattr(fine, key), rtol=5e-8)


@pytest.mark.parametrize("T,n", [(T, sign*n) for T in (.01, 1., 10.)
                                  for n in (1e-18, 1e-24, 1e-30)
                                  for sign in (-1., 1.)])
def test_tiny_signed_net_number_root(T, n):
    s = state_density(T, n, mass_mev=0.)
    exact = (s.chemical_potential**3/(3*math.pi**2)
             +s.chemical_potential*T*T/3)/HBARC**3
    np.testing.assert_allclose(s.net_density, n, rtol=1e-9)
    np.testing.assert_allclose(exact, n, rtol=1e-9)


@pytest.mark.parametrize("mass,T", [(M_E, 1.), (M_MU, 20.)])
def test_pair_zero_and_particle_antiparticle_symmetry(mass, T):
    zero = state_density(T, 0., mass)
    assert zero.chemical_potential == zero.net_density == 0.
    assert zero.particle_density == zero.antiparticle_density > 0.
    plus, minus = state_mu(T, .7*T, mass), state_mu(T, -.7*T, mass)
    np.testing.assert_allclose(plus.net_density, -minus.net_density, rtol=1e-13)
    np.testing.assert_allclose(plus.particle_density, minus.antiparticle_density, rtol=1e-13)
    for field in ["energy_density", "pressure", "entropy_density",
                  "cv_fixed_net_density", "dn_dmu_fixed_temperature"]:
        np.testing.assert_allclose(getattr(plus, field), getattr(minus, field), rtol=1e-13)
    # Independent flavors do not acquire an implicit common chemical potential.
    other = state_density(T, .005, M_MU)
    assert abs(other.chemical_potential-zero.chemical_potential) > 1.


@pytest.mark.parametrize("T,n,m", [(10., 1e-6, M_E), (20., .005, M_MU),
                                    (.01, .002, M_MU), (.1, -1e-12, M_E)])
def test_responses_after_forward_gate(T, n, m):
    s = state_density(T, n, m)
    np.testing.assert_allclose(s.net_density, n, rtol=1e-9)
    h = .002*T
    low, high = state_density(T-h, n, m), state_density(T+h, n, m)
    np.testing.assert_allclose(s.cv_fixed_net_density,
                               (high.energy_density-low.energy_density)/(2*h), rtol=5e-4)
    np.testing.assert_allclose(s.dmu_dtemperature_fixed_net_density,
                               (high.chemical_potential-low.chemical_potential)/(2*h), rtol=5e-4)
    hm = 1e-4*max(T, abs(s.chemical_potential), 1e-3)
    lo, hi = state_mu(T, s.chemical_potential-hm, m), state_mu(T, s.chemical_potential+hm, m)
    np.testing.assert_allclose(s.dn_dmu_fixed_temperature,
                               (hi.net_density-lo.net_density)/(2*hm), rtol=2e-5)
    np.testing.assert_allclose(s.dn_dtemperature_fixed_mu,
                               (hi.entropy_density-lo.entropy_density)/(2*hm), rtol=2e-5)


@pytest.mark.parametrize("bad", [True, False, np.bool_(True), None, "1", complex(1),
                                 float("nan"), float("inf"), -float("inf")])
def test_public_scalar_admission(bad):
    for call in [lambda: state_mu(bad, 0.), lambda: state_mu(1., bad),
                 lambda: state_mu(1., 0., bad), lambda: state_density(1., bad),
                 lambda: state_mu(1., 0., rtol=bad), lambda: state_mu(1., 0., tail=bad)]:
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("kwargs", [{"temperature_mev": 0.}, {"temperature_mev": -1.},
                                    {"mass_mev": -1.}, {"rtol": 0.},
                                    {"rtol": 1e-4}, {"tail": 29.}, {"tail": 101.}])
def test_invalid_physical_or_quadrature_domain(kwargs):
    args = dict(temperature_mev=1., chemical_potential_mev=0.)
    args.update(kwargs)
    with pytest.raises(ValueError):
        state_mu(**args)


def test_susceptibility_underflow_is_not_an_absent_pair_claim():
    with pytest.raises(ArithmeticError, match="Susceptibility underflow"):
        state_mu(.001, 0., M_MU)
    # Degenerate states can still have unrepresented antiparticle populations.
    s = state_mu(.01, 110., M_MU)
    assert s.antiparticle_density == 0.
    assert s.dn_dmu_fixed_temperature > 0.
    assert math.isfinite(s.energy_density)


def test_thermal_state_validation_and_immutability():
    s = state_density(1., 1e-7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.net_density = 0.
    with pytest.raises(ValueError):
        thermal_energy_density(None)
    with pytest.raises(ValueError, match="inconsistent"):
        thermal_energy_density(dataclasses.replace(s, net_density=2*s.net_density))
    assert isinstance(s, LeptonState)


@pytest.mark.parametrize("muons", [False, True])
def test_existing_cold_sector_limit_with_explicit_units(muons):
    # This is a limiting comparison, not an attempt to replace the cold root.
    from nmbackend.eos.leptons import lepton_sector
    mu_fm, ne, nmu, energy_fm4 = lepton_sector(.02, muons=muons)
    electron = state_mu(.001, float(mu_fm)*HBARC, M_E)
    energy = electron.energy_density
    net = electron.net_density
    if muons:
        muon = state_mu(.001, float(mu_fm)*HBARC, M_MU)
        energy += muon.energy_density
        net += muon.net_density
    np.testing.assert_allclose(net, float(ne+nmu), rtol=5e-8)
    np.testing.assert_allclose(energy, float(energy_fm4)*HBARC, rtol=5e-8)


@pytest.mark.parametrize("z", [40., 100., 300., 600.])
@pytest.mark.parametrize("alpha", [0., .4, -.4])
def test_resolved_dilute_all_fields_against_relativistic_bessel(z, alpha):
    # The exact classical relativistic moments use scaled Bessel functions.
    # At z>=40 and |mu/T|<=.4, FD corrections are <7e-18.
    from scipy.special import kve
    T, m = M_E/z, M_E
    mu = alpha*T
    base = math.exp(2*math.log(m)+math.log(T)-math.log(math.pi**2*HBARC**3)
                    +math.log(kve(2, z))-z)
    na, nb = base*math.exp(alpha), base*math.exp(-alpha)
    total, net = na+nb, na-nb
    mean = 3*T+m*kve(1, z)/kve(2, z)
    square = m*m+3*m*T*kve(1, z)/kve(2, z)+12*T*T
    first = mean*net-mu*total
    second = (square+mu*mu)*total-2*mu*mean*net
    energy, pressure = total*mean, total*T
    entropy = (energy+pressure-mu*net)/T
    expected = dict(net_density=net, particle_density=na,
                    antiparticle_density=nb, energy_density=energy,
                    pressure=pressure, entropy_density=entropy,
                    free_energy_density=mu*net-pressure,
                    cv_fixed_net_density=(second-first*(first/total))/(T*T),
                    dn_dmu_fixed_temperature=total/T,
                    dn_dtemperature_fixed_mu=first/(T*T),
                    dmu_dtemperature_fixed_net_density=-first/(T*total))
    s = state_mu(T, mu, m)
    fine = state_mu(T, mu, m, tail=90., rtol=5e-13)
    for field, value in expected.items():
        actual = getattr(s, field)
        if value == 0:
            assert actual == 0.
        else:
            np.testing.assert_allclose(actual, value, rtol=3e-10, atol=0.)
            np.testing.assert_allclose(getattr(fine, field), actual, rtol=3e-10, atol=0.)
    if alpha == 0:
        np.testing.assert_allclose(thermal_energy_density(s), energy, rtol=3e-10, atol=0.)


@pytest.mark.parametrize("T", [1e-30, 1e-9, 1e6])
@pytest.mark.parametrize("alpha", [0., .37, -.37])
def test_massless_dimensionful_scale_closed_form(T, alpha):
    # Independent exact polynomials test dimensionful scaling of every bulk
    # and response field, with no absolute comparison floor at small T.
    h3 = HBARC**3
    nbar = (alpha**3/(3*math.pi**2)+alpha/3)/h3
    pbar = (alpha**4/(12*math.pi**2)+alpha*alpha/6+7*math.pi**2/180)/h3
    sbar = (alpha*alpha/3+7*math.pi**2/45)/h3
    chibar = (alpha*alpha/math.pi**2+1/3)/h3
    ntbar = 2*alpha/(3*h3)
    cvbar = (alpha*alpha/3+7*math.pi**2/15)/h3-ntbar*ntbar/chibar
    s = state_mu(T, alpha*T, 0.)
    expected = dict(net_density=nbar*T**3, pressure=pbar*T**4,
                    energy_density=3*pbar*T**4, entropy_density=sbar*T**3,
                    free_energy_density=(3*pbar-sbar)*T**4,
                    dn_dmu_fixed_temperature=chibar*T*T,
                    dn_dtemperature_fixed_mu=ntbar*T*T,
                    cv_fixed_net_density=cvbar*T**3,
                    dmu_dtemperature_fixed_net_density=-ntbar/chibar)
    for field, value in expected.items():
        if value == 0:
            assert getattr(s, field) == 0.
        else:
            np.testing.assert_allclose(getattr(s, field), value, rtol=3e-10, atol=0.)
    # Separate scalar original-momentum population integrals, not a shared
    # dimensional-state reference, also check both particle populations.
    for field, chemical in [("particle_density", alpha), ("antiparticle_density", -alpha)]:
        number = quad(lambda x: x*x*expit(chemical-x), 0., np.inf,
                      epsabs=0., epsrel=3e-12)[0]/(math.pi**2*h3)*T**3
        np.testing.assert_allclose(getattr(s, field), number, rtol=3e-10, atol=0.)
    pfbar = (3*math.pi**2*abs(nbar)*h3)**(1/3) if alpha else 0.
    coldbar = pfbar**4/(4*math.pi**2*h3)
    np.testing.assert_allclose(thermal_energy_density(s),
                               (3*pbar-coldbar)*T**4, rtol=3e-10, atol=0.)
