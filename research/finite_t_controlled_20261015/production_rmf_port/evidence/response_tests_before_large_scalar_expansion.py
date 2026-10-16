"""Stationarity-aware response and admission tests after the forward gate."""
import math
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
import pytest


def original_gas(t,nu,m,kind):
    end=max(abs(nu),m)+90*t;pmax=math.sqrt((end-m)*(end+m))
    def f(p):
        e=math.hypot(p,m);x,y=(e-nu)/t,(e+nu)/t;a,b=expit(-x),expit(-y)
        if kind=='n':v=a-b
        elif kind=='scalar':v=m/e*(a+b)
        elif kind=='entropy':v=sum(np.logaddexp(0.,-abs(z))+abs(z)*expit(-abs(z))for z in[x,y])
        elif kind=='energy':v=e*(a+b)
        else:raise ValueError(kind)
        return p*p*v/math.pi**2
    return quad(f,0,pmax,epsabs=2e-15,epsrel=2e-12,limit=300)[0]


@pytest.mark.parametrize('t,nu,m',[(.5,0.,2.),(.3,.4,1.5)])
def test_pair_mass_thermal_moments(t,nu,m):
    from nmbackend.reference._rmf_moments import moments
    g=moments(t,nu,m);h=1e-4
    np.testing.assert_allclose([g.density,g.scalar,g.energy,g.entropy],
        [original_gas(t,nu,m,k)for k in['n','scalar','energy','entropy']],rtol=2e-9,atol=1e-13)
    d=lambda kind,axis:(original_gas(t+(h if axis=='t'else 0),nu+(h if axis=='nu'else 0),m+(h if axis=='m'else 0),kind)-original_gas(t-(h if axis=='t'else 0),nu-(h if axis=='nu'else 0),m-(h if axis=='m'else 0),kind))/(2*h)
    np.testing.assert_allclose([g.chi,g.scalar_nu,g.scalar_mass,g.n_temperature,g.scalar_temperature,g.s_temperature],
        [d('n','nu'),d('scalar','nu'),d('scalar','m'),d('n','t'),d('scalar','t'),d('entropy','t')],rtol=2e-6,atol=1e-10)
    np.testing.assert_allclose(d('n','m'),-g.scalar_nu,rtol=2e-6,atol=1e-10)
    assert g.antiparticle_density>0


@pytest.mark.parametrize('T,n,Y',[(7.,.18,.17),(25.,.28,.35)])
def test_relaxed_Cv_and_full_H(T,n,Y):
    from nmbackend.reference.finite_t_rmf import state_density,HBARC
    from nmbackend.reference._rmf_moments import moments
    from nmbackend.reference._rmf_response import response
    s=state_density(T,n,Y);cv=[];hess=[]
    for eps in [.002,.001,.0005]:
        a=state_density(T*(1+eps),n,Y);b=state_density(T*(1-eps),n,Y)
        cv.append((a.energy_density-b.energy_density)/(2*eps*T))
        pair=np.array([n*(1-Y),n*Y]);H=np.zeros((2,2))
        for j in range(2):
            h=eps*pair[j];up=pair.copy();down=pair.copy();up[j]+=h;down[j]-=h
            a=state_density(T,float(up.sum()),float(up[1]/up.sum()));b=state_density(T,float(down.sum()),float(down[1]/down.sum()))
            H[:,j]=(np.array([a.mu_n,a.mu_p])-np.array([b.mu_n,b.mu_p]))/(2*h)
        hess.append(H)
    cvr=(4*cv[-1]-cv[-2])/3;Hr=(4*hess[-1]-hess[-2])/3
    np.testing.assert_allclose(s.cv_fixed_density_composition,cvr,rtol=3e-6)
    assert np.linalg.norm(Hr-np.asarray(s.hessian_n_p))/np.linalg.norm(Hr)<2e-6
    assert np.linalg.norm(Hr-Hr.T)/np.linalg.norm(Hr)<2e-6
    gn=moments(T/HBARC,s.kinetic_mu_n/HBARC,s.dirac_mass/HBARC);gp=moments(T/HBARC,s.kinetic_mu_p/HBARC,s.dirac_mass/HBARC)
    r=response(s.phi/HBARC,gn,gp,939/HBARC,11.785,7.148,4.410,.002947,-.001070,n,Y,HBARC)
    np.testing.assert_allclose(s.cv_fixed_density_composition,(T/HBARC)*r['entropy_t'],rtol=1e-12)
    assert s.cv_fixed_density_composition>(T/HBARC)*r['entropy_t_frozen_mass']


def test_potential_firstlaw():
    from nmbackend.reference.finite_t_rmf import state_density
    T,n,Y=20.,.24,.3;s=state_density(T,n,Y);e=2e-4
    ta,tb=state_density(T*(1+e),n,Y),state_density(T*(1-e),n,Y)
    na,nb=state_density(T,n*(1+e),Y),state_density(T,n*(1-e),Y)
    ya,yb=state_density(T,n,Y+e),state_density(T,n,Y-e)
    np.testing.assert_allclose((ta.free_energy_density-tb.free_energy_density)/(2*e*T),-s.entropy_density,rtol=3e-6)
    np.testing.assert_allclose((na.free_energy_density-nb.free_energy_density)/(2*e*n),(1-Y)*s.mu_n+Y*s.mu_p,rtol=3e-6)
    np.testing.assert_allclose((ya.free_energy_density-yb.free_energy_density)/(2*e),n*(s.mu_p-s.mu_n),rtol=3e-6)
    np.testing.assert_allclose(n*(na.free_energy_density-nb.free_energy_density)/(2*e*n)-s.free_energy_density,s.pressure,rtol=3e-6)


def test_counterexample_and_scalar_helpers():
    from nmbackend.reference.finite_t_rmf import diagnose_density,state_density,_crossing,_M
    s=diagnose_density(1.,.06,.1);H=np.asarray(s.hessian_n_p);v=np.array([.9,.1])
    assert s.scalar_curvature_fm_minus2>0 and .06*(v@H@v)>0
    assert not s.locally_admitted and s.hessian_min_eigenvalue<0
    with pytest.raises(ValueError,match='admission'):state_density(1.,.06,.1)
    with pytest.raises(ArithmeticError,match='unique'):_crossing(lambda x:1.,8)
    with pytest.raises(ArithmeticError,match='unique'):_crossing(lambda x:-math.sin(6*math.pi*x/(.985*_M)),48)
    with pytest.raises(ArithmeticError,match='Nonfinite'):_crossing(lambda x:float('nan'),8)


@pytest.mark.parametrize('args',[(True,.16,.1),(10.,float('nan'),.1),(10.,.16,False),(0.,.16,.1),(51.,.16,.1),(10.,.019,.1),(10.,.16,.01),(10.,.16,.501)])
def test_invalid_state(args):
    from nmbackend.reference.finite_t_rmf import state_density
    with pytest.raises(ValueError):state_density(*args)


@pytest.mark.parametrize('kwargs',[{'rtol':True},{'rtol':float('nan')},{'rtol':.001},{'tail':29.},{'tail':101.},{'tail':float('inf')}])
def test_invalid_controls(kwargs):
    from nmbackend.reference.finite_t_rmf import SolveControls
    with pytest.raises(ValueError):SolveControls(**kwargs)


def test_actual_numerical_failure_and_seed(monkeypatch):
    import nmbackend.reference.finite_t_rmf as R
    with pytest.raises(ValueError):R.state_density(10.,.16,.1,initial={'phi':.2})
    with pytest.raises(TypeError):R.state_density(10.,.16,.1,params={'Lambda_v':.1})
    monkeypatch.setattr(R,'least_squares',lambda *a,**kw:SimpleNamespace(x=np.array([.5,4.3,4.2]),success=False))
    with pytest.raises(ArithmeticError,match='continuation'):R.state_density(10.,.16,.1)


def test_refined_tail_and_underflow_component():
    from nmbackend.reference.finite_t_rmf import state_density,SolveControls
    s=state_density(7.,.18,.17);f=state_density(7.,.18,.17,controls=SolveControls(rtol=2e-12,tail=90.))
    for k in ['energy_density','pressure','entropy_density','cv_fixed_density_composition','mu_n','mu_p','dirac_mass']:
        np.testing.assert_allclose(getattr(s,k),getattr(f,k),rtol=2e-9)
    np.testing.assert_allclose(s.hessian_n_p,f.hessian_n_p,rtol=2e-9)
    cold=state_density(1.,.213,.237)
    assert cold.antiparticle_density_n is None and math.isfinite(cold.log_antiparticle_density_n)
    assert cold.neutron_density>0 and cold.cv_fixed_density_composition>0


def test_H_from_potential_second_difference():
    from nmbackend.reference.finite_t_rmf import state_density
    T,n,Y=7.,.18,.17;s=state_density(T,n,Y);v=np.array([n*(1-Y),n*Y]);h=.001*v
    def F(d):return state_density(T,float(d.sum()),float(d[1]/d.sum())).free_energy_density
    H=np.zeros((2,2));f0=s.free_energy_density
    for i in range(2):
        d=np.zeros(2);d[i]=h[i];H[i,i]=(F(v+d)-2*f0+F(v-d))/h[i]**2
    H[0,1]=H[1,0]=(F(v+h)-F(v+np.array([h[0],-h[1]]))-F(v+np.array([-h[0],h[1]]))+F(v-h))/(4*h[0]*h[1])
    assert np.linalg.norm(H-np.asarray(s.hessian_n_p))/np.linalg.norm(H)<1e-4


def test_pair_Cv_fixed_NET_number():
    from scipy.optimize import brentq
    from nmbackend.reference._rmf_moments import moments
    t,nu,m=.3,.4,1.5;g=moments(t,nu,m);target=original_gas(t,nu,m,'n');h=1e-4
    def energy(temp):
        v=brentq(lambda a:original_gas(temp,a,m,'n')-target,0.,2.,xtol=1e-13)
        return original_gas(temp,v,m,'energy')
    cv=(energy(t+h)-energy(t-h))/(2*h)
    np.testing.assert_allclose(t*(g.s_temperature-g.n_temperature**2/g.chi),cv,rtol=2e-6)


def test_selected_cold_Sommerfeld_limit():
    from nmbackend.reference.finite_t_rmf import state_density,HBARC
    s=state_density(1.,.213,.237)
    gamma=sum((3*math.pi**2*n)**(1/3)*math.hypot((3*math.pi**2*n)**(1/3),s.dirac_mass/HBARC)/(3*HBARC)for n in[s.neutron_density,s.proton_density])
    assert abs(s.cv_fixed_density_composition/(gamma*s.temperature)-1)<.02
