"""Particle-only ideal gas: independent forward integrals precede responses."""
import math
import dataclasses
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import expit, kve
from nmbackend.constants import HBARC, M_N, M_P
from nmbackend.reference.finite_t_baryons import ParticleState, state_mu, state_density


def momentum_reference(T, mu, m):
    scale = max(T, math.sqrt(m*T))
    nu = mu-m
    end = max(0., nu)+90*T
    upper = math.sqrt(end*(end+2*m))/scale
    split = math.sqrt(nu*(nu+2*m))/scale if nu > 0 else None
    def moment(fn):
        def fun(q):
            p = scale*q
            E = math.hypot(p,m)
            k = p*p/(E+m)
            d = (k-nu)/T
            f = expit(-d)
            return q*q*fn(p,E,k,d,f,f*expit(d))
        return scale**3/(math.pi**2*HBARC**3)*quad(
            fun,0.,upper,points=[split] if split else None,
            epsabs=0.,epsrel=2e-11,limit=300)[0]
    n = moment(lambda p,E,k,d,f,w:f)
    kinetic = moment(lambda p,E,k,d,f,w:k*f)
    pressure = moment(lambda p,E,k,d,f,w:p*p/(3*E)*f)
    entropy = moment(lambda p,E,k,d,f,w:math.log1p(math.exp(-abs(d)))+abs(d)*expit(-abs(d)))
    a0 = moment(lambda p,E,k,d,f,w:w)
    a1 = moment(lambda p,E,k,d,f,w:d*w)
    center = moment(lambda p,E,k,d,f,w:k*w)/a0
    cv = moment(lambda p,E,k,d,f,w:((k-center)/T)**2*w)
    return dict(particle_density=n,kinetic_energy_density=kinetic,
                energy_density=m*n+kinetic,pressure=pressure,entropy_density=entropy,
                free_energy_density=m*n+kinetic-T*entropy,
                cv_fixed_particle_density=cv,dn_dmu_fixed_temperature=a0/T,
                dn_dtemperature_fixed_mu=a1/T,
                dmu_dtemperature_fixed_particle_density=-a1/a0)


@pytest.mark.parametrize("T,n,m", [(5.,1e-8,M_N),(2.,.002,M_P),
    (.05,.08,M_N),(40.,.01,M_P),(.19,.021,M_P),(250.,.001,500.)])
def test_original_momentum_forward_then_responses(T,n,m):
    s = state_density(T,n,m)
    np.testing.assert_allclose(s.particle_density,n,rtol=1e-9,atol=0.)
    expected = momentum_reference(T,s.chemical_potential,m)
    for key in ['particle_density','kinetic_energy_density','energy_density',
                'pressure','entropy_density','free_energy_density']:
        np.testing.assert_allclose(getattr(s,key),expected[key],rtol=3e-9,atol=0.)
    for key in ['cv_fixed_particle_density','dn_dmu_fixed_temperature',
                'dn_dtemperature_fixed_mu','dmu_dtemperature_fixed_particle_density']:
        np.testing.assert_allclose(getattr(s,key),expected[key],rtol=3e-8,atol=0.)


@pytest.mark.parametrize("z,eta", [(40.,-35.),(600.,-35.),(40.,-600.),(600.,-600.)])
def test_dilute_bessel_all_fields_without_amplitude_floor(z,eta):
    T,m=M_N/z,M_N
    s=state_mu(T,m+eta*T,m)
    n=math.exp(2*math.log(m)+math.log(T)-math.log(math.pi**2*HBARC**3)
               +eta+math.log(kve(2,z)))
    mean=3*T+m*kve(1,z)/kve(2,z)
    square=m*m+3*m*T*kve(1,z)/kve(2,z)+12*T*T
    kmean=mean-m
    expected=dict(particle_density=n,kinetic_energy_density=n*kmean,
        energy_density=n*mean,pressure=n*T,entropy_density=n*(kmean/T+1-eta),
        free_energy_density=n*(m+eta*T-T),
        cv_fixed_particle_density=n*((square-mean*mean)/T**2),
        dn_dmu_fixed_temperature=n/T,
        dn_dtemperature_fixed_mu=n*(kmean/T-eta)/T,
        dmu_dtemperature_fixed_particle_density=eta-kmean/T)
    for key,value in expected.items():
        np.testing.assert_allclose(getattr(s,key),value,rtol=3e-8,atol=0.)
    fine=state_mu(T,m+eta*T,m,rtol=5e-13,tail=90.)
    for key in expected:
        np.testing.assert_allclose(getattr(s,key),getattr(fine,key),rtol=3e-9,atol=0.)


@pytest.mark.parametrize("n", [1e-24,1e-120,1e-280])
def test_tiny_positive_particle_root(n):
    s=state_density(1.,n,M_N)
    assert s.particle_density>0.
    np.testing.assert_allclose(s.particle_density,n,rtol=1e-9,atol=0.)


@pytest.mark.parametrize("T,n,m", [(3.7,.0034,M_N),(.19,.021,M_P),(40.,.01,M_N)])
def test_legendre_pressure_and_responses_after_forward(T,n,m):
    s=state_density(T,n,m)
    np.testing.assert_allclose(s.particle_density,n,rtol=1e-9,atol=0.)
    np.testing.assert_allclose(s.energy_density+s.pressure-s.chemical_potential*n,
                               T*s.entropy_density,rtol=3e-8,atol=0.)
    hm=.001*T
    lo,hi=state_mu(T,s.chemical_potential-hm,m),state_mu(T,s.chemical_potential+hm,m)
    np.testing.assert_allclose((hi.pressure-lo.pressure)/(2*hm),n,rtol=3e-6)
    np.testing.assert_allclose((hi.particle_density-lo.particle_density)/(2*hm),
                               s.dn_dmu_fixed_temperature,rtol=3e-6)
    np.testing.assert_allclose((hi.entropy_density-lo.entropy_density)/(2*hm),
                               s.dn_dtemperature_fixed_mu,rtol=3e-6)
    for fraction in [.004,.002]:
        ht=fraction*T
        low,high=state_density(T-ht,n,m),state_density(T+ht,n,m)
        # Rest energy is constant at fixed particle n; use its direct kinetic
        # integral to avoid a rest-dominated numerical energy subtraction.
        np.testing.assert_allclose((high.kinetic_energy_density-low.kinetic_energy_density)/(2*ht),
                                   s.cv_fixed_particle_density,rtol=3e-4)
        np.testing.assert_allclose((high.chemical_potential-low.chemical_potential)/(2*ht),
                                   s.dmu_dtemperature_fixed_particle_density,rtol=3e-4)


def test_relativistic_cold_sommerfeld_and_nonrelativistic_limits():
    T,mu,m=.001,1200.,M_N
    s=state_mu(T,mu,m)
    p=math.sqrt(mu*mu-m*m)
    n0=p**3/(3*math.pi**2*HBARC**3)
    e0=(p*mu*(2*p*p+m*m)-m**4*math.log((p+mu)/m))/(8*math.pi**2*HBARC**3)
    p0=(p*mu*(2*p*p-3*m*m)+3*m**4*math.log((p+mu)/m))/(24*math.pi**2*HBARC**3)
    np.testing.assert_allclose([s.particle_density,s.energy_density,s.pressure],[n0,e0,p0],rtol=3e-9)
    np.testing.assert_allclose(s.cv_fixed_particle_density/n0/T,math.pi**2*mu/(p*p),rtol=3e-8)
    T,eta=1e-4,-15.
    s=state_mu(T,m+eta*T,m)
    n=2*(m*T/(2*math.pi))**1.5*math.exp(eta)/HBARC**3
    np.testing.assert_allclose(s.particle_density,n,rtol=3e-6)
    np.testing.assert_allclose(s.kinetic_energy_density/(n*T),1.5,rtol=3e-6)
    np.testing.assert_allclose(s.cv_fixed_particle_density/s.particle_density,1.5,rtol=3e-6)


@pytest.mark.parametrize("bad", [True,False,np.bool_(False),None,'1',1j,float('nan'),float('inf')])
def test_nonfinite_nonreal_boolean_admission(bad):
    for call in [lambda:state_mu(bad,930.),lambda:state_mu(1.,bad),
                 lambda:state_mu(1.,930.,bad),lambda:state_density(1.,bad),
                 lambda:state_mu(1.,930.,rtol=bad),lambda:state_mu(1.,930.,tail=bad)]:
        with pytest.raises(ValueError):call()


@pytest.mark.parametrize("T,n,m", [(0.,.01,M_N),(-1.,.01,M_N),(1.,0.,M_N),
    (1.,-1e-30,M_N),(1.,.01,0.),(1.,.01,-1.)])
def test_invalid_particle_physical_inputs(T,n,m):
    with pytest.raises(ValueError):state_density(T,n,m)


def test_model_convention_fixed_mass_and_unrepresented_policy():
    s=state_mu(500.,-200.,M_P)
    assert s.particle_density>0. and s.mass==M_P
    # Deliberately no pair ansatz veto: this is a particle-only mathematical
    # gas, even where omitted equilibrium antiparticles would matter physically.
    assert not hasattr(s,'net_density') and not hasattr(s,'antiparticle_density')
    assert isinstance(s,ParticleState)
    with pytest.raises(dataclasses.FrozenInstanceError):s.mass=1.
    with pytest.raises(ArithmeticError):state_mu(1.,-1e6,M_N)
    for kwargs in [dict(rtol=0.),dict(rtol=1e-4),dict(tail=29.),dict(tail=101.)]:
        with pytest.raises(ValueError):state_mu(1.,930.,**kwargs)


@pytest.mark.parametrize("T", [1e-24,1e-9,1e6])
def test_dimensionful_mass_temperature_scaling_without_absolute_floor(T):
    m=13.*T
    s=state_mu(T,m-.37*T,m)
    expected=momentum_reference(T,s.chemical_potential,m)
    for field,value in expected.items():
        np.testing.assert_allclose(getattr(s,field),value,rtol=3e-9,atol=0.)
    recovered=state_density(T,s.particle_density,m)
    np.testing.assert_allclose(recovered.particle_density,s.particle_density,rtol=1e-9,atol=0.)
    fine=state_mu(T,s.chemical_potential,m,rtol=5e-13,tail=90.)
    for field in expected:
        np.testing.assert_allclose(getattr(s,field),getattr(fine,field),rtol=3e-9,atol=0.)


def test_unrepresented_total_mu_offset_does_not_fake_tiny_root():
    # At this T the required offset from the rest mass cannot be encoded in
    # the returned ordinary total mu; the forward reconstruction must reject.
    with pytest.raises(ArithmeticError,match="representation residual"):
        state_density(1e-20,1e-35,M_N)
