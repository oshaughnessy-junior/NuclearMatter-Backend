"""Analytic eigenvalue, convention, continuation and real r-mode benchmark gates."""
from __future__ import annotations
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from nmbackend.modes import (Pencil,ModeError,solve,select_simple,directional_derivative,
    frequencies,stationary_patterns,conjugate_partner,exterior_contour,
    match_branches,toroidal_rmode,ModeBackground,rns_polytropic_background)
from nmbackend.modes.autodiff import simple_root
from nmbackend.rotating.rns import RNSBackground


def assert_roots(actual,expected):
    cost=abs(np.asarray(actual)[:,None]-np.asarray(expected)[None,:])
    a,b=linear_sum_assignment(cost)
    np.testing.assert_allclose(np.asarray(actual)[a],np.asarray(expected)[b],rtol=1e-11,atol=1e-12)


def test_damped_and_growing_quadratic_roots():
    gamma=np.array([.03,-.02])
    omega=np.array([2.,3.])
    p=Pencil(np.diag(omega**2),2j*np.diag(gamma),-np.eye(2))
    s=solve(p)
    expected=np.concatenate([1j*gamma+np.sqrt(omega**2-gamma**2),1j*gamma-np.sqrt(omega**2-gamma**2)])
    assert_roots(s.sigma,expected)
    assert s.infinite_roots==0 and max(s.residual)<1e-12 and max(s.left_residual)<1e-12
    assert frequencies(expected[0],2,.1)['behavior']=='damped'
    assert frequencies(expected[1],2,.1)['behavior']=='growing'


def test_singular_k2_is_not_inverted_and_zero_pencil_rejected():
    p=Pencil(np.diag([1.,2.]),-np.eye(2),np.zeros((2,2)))
    s=solve(p)
    assert_roots(s.sigma,[1.,2.])
    assert s.infinite_roots==2
    with pytest.raises(ModeError,match='indeterminate'):
        solve(Pencil(np.zeros((1,1)),np.zeros((1,1)),np.zeros((1,1))))
    with pytest.raises(ModeError):Pencil([1],[[1]],[[1]])
    with pytest.raises(ModeError):solve(p,max_dimension=1)


def test_nonhermitian_left_vectors_and_implicit_derivative():
    transform=np.array([[1.,2.],[0.,1.]])
    inverse=np.linalg.inv(transform)
    def pencil(t):
        return Pencil(transform@np.diag([2+t,3.])@inverse,
                      transform@np.diag([.1j,.2j])@inverse,-np.eye(2))
    p=pencil(.2)
    s,j=select_simple(p,1.5+.05j)
    delta0=transform@np.diag([1.,0.])@inverse
    ad=directional_derivative(p,s,j,delta0,np.zeros((2,2)),np.zeros((2,2)))
    for h in (1e-3,1e-4,1e-5):
        a,k=select_simple(pencil(.2+h),s.sigma[j])
        b,l=select_simple(pencil(.2-h),s.sigma[j])
        np.testing.assert_allclose(ad,(a.sigma[k]-b.sigma[l])/(2*h),rtol=1e-6)


def test_jax_root_primal_jvp_reverse_and_finite_difference():
    def root(k):
        return simple_root(jnp.array([[k]]),jnp.array([[.06j]]),jnp.array([[-1.]]),jnp.asarray(2.+.03j))
    expected=np.sqrt(4.-.03**2)+.03j
    np.testing.assert_allclose(jax.jit(root)(4.),expected,rtol=1e-12)
    value,tangent=jax.jvp(root,(jnp.asarray(4.),),(jnp.asarray(1.),))
    reverse=jax.jit(jax.grad(lambda k:jnp.real(root(k))))(4.)
    expected_slope=1/(2*np.sqrt(4.-.03**2))
    np.testing.assert_allclose(tangent,expected_slope,rtol=1e-12)
    np.testing.assert_allclose(reverse,expected_slope,rtol=1e-12)
    np.testing.assert_allclose((root(4.+1e-5)-root(4.-1e-5))/2e-5,reverse,rtol=1e-8)
    np.testing.assert_allclose(value,root(4.),rtol=1e-13)
    def damping(g):
        return jnp.imag(simple_root(jnp.array([[4.]]),jnp.array([[2j*g]]),jnp.array([[-1.]]),jnp.asarray(expected)))
    np.testing.assert_allclose(jax.grad(damping)(.03),1.,rtol=1e-12)


def test_exceptional_and_repeated_roots_reject_scalar_derivatives():
    p=Pencil(np.array([[1j,1.],[0.,1j]]),-np.eye(2),np.zeros((2,2)))
    with pytest.raises(ModeError): select_simple(p,1j)


@pytest.mark.parametrize('coupling',[.05,.2])
def test_avoided_crossing_and_rate_exchange(coupling):
    a,b=1.+.1j,1.+.3j
    h=np.array([[a,coupling],[coupling,b]])
    roots=solve(Pencil(h,-np.eye(2),np.zeros((2,2)))).sigma
    splitting=np.sqrt(complex((a-b)**2/4+coupling**2))
    assert_roots(roots,[(a+b)/2+splitting,(a+b)/2-splitting])
    if coupling>.1: assert np.ptp(roots.real)>0 and np.ptp(roots.imag)<1e-12
    else: assert np.ptp(roots.imag)>0 and np.ptp(roots.real)<1e-12


def test_tracking_uses_vectors_at_a_frequency_crossing():
    def spectrum(t):return solve(Pencil(np.diag([t,-t]),-np.eye(2),np.zeros((2,2))))
    previous,current=spectrum(-.1),spectrum(.1)
    indices,overlap=match_branches(previous,current,minimum_overlap=.99)
    np.testing.assert_array_equal(indices,[1,0])
    np.testing.assert_allclose(overlap,1.)
    with pytest.raises(ModeError):match_branches(previous,current,weight=np.diag([1.,-1.]))
    # A scalar quadratic has identical displacement vectors for both roots;
    # frequency continuity resolves the otherwise identical overlaps.
    a=solve(Pencil([[1.]],[[0.]],[[-1.]]))
    b=solve(Pencil([[1.21]],[[0.]],[[-1.]]))
    indices,_=match_branches(a,b)
    np.testing.assert_array_equal(indices,[0,1])


@pytest.mark.parametrize('m',[2,3,4])
def test_numerical_toroidal_rmode_integrals(m):
    omega=3.7
    for order in (2*m+2,24,40):
        p,metadata=toroidal_rmode(m,omega,angular_points=order)
        s=solve(p)
        assert len(s.sigma)==1 and s.infinite_roots==1
        info=frequencies(s.sigma[0],m,omega)
        np.testing.assert_allclose(info['omega_ro']/omega,2*m/(m*(m+1)),rtol=1e-12)
        np.testing.assert_allclose(info['omega_in'],omega*(2/(m+1)-m),rtol=1e-12)
        assert info['behavior']=='conservative' and np.isinf(info['efolding_time'])
        assert metadata['radiation']=='not included'
    with pytest.raises(ValueError):toroidal_rmode(1,omega)


def test_frame_neutrality_conjugation_and_outgoing_sign():
    assert stationary_patterns(0.,2,1.,angular_tolerance=1e-12)=={'inertial_stationary':True,'corotating_stationary':False}
    assert stationary_patterns(-2.,2,1.,angular_tolerance=1e-12)=={'inertial_stationary':False,'corotating_stationary':True}
    partner,m=conjugate_partner(1.+.1j,2)
    assert partner==-1.+.1j and m==-2
    r,jac=exterior_contour(np.arange(4),2.,-.4,stellar_outer_radius=1.,sigma=1.+.1j)
    assert np.imag((1.+.1j)*jac)<0
    assert abs(np.exp(-1j*(1.+.1j)*(r[-1]-r[0])))<1
    with pytest.raises(ValueError):exterior_contour([0,1],2.,.4,stellar_outer_radius=1.,sigma=1.+.1j)
    with pytest.raises(ValueError):exterior_contour([0,1],.5,-.4,stellar_outer_radius=1.,sigma=1.+.1j)


def test_rns_background_exchange_requires_perturbative_closure():
    shape=(3,3)
    fields={k:np.zeros(shape) for k in ('rho','gama','alpha','omega','velocity_sq','enthalpy')}
    fields.update(energy=np.full(shape,.2),pressure=np.full(shape,.02))
    star=RNSBackground(np.array([0.,.5,1.]),np.array([0.,.5,1.]),fields,{'Gamma':2.,'r_ratio':1.,'r_e':1.,'Omega':0.})
    background=rns_polytropic_background(star)
    np.testing.assert_allclose(background.sound_speed_sq,2*.02/.22)
    assert background.closure=='barotropic'
    with pytest.raises(ModeError):ModeBackground(star,2.,'barotropic','test')
    with pytest.raises(ModeError):ModeBackground(star,.2,'finite-reaction','test')
    with pytest.raises(ModeError):ModeBackground({},.2,'barotropic','unqualified learned fields')


def test_maclaurin_integrals_match_independent_closed_coefficients():
    from nmbackend.modes import bar_mode
    for e in (.2,.5,.9):
        _,meta=bar_mode(e)
        q=np.sqrt(1-e*e)
        b=(3*e-5*e**3+2*e**5+q*(4*e*e-3)*np.arcsin(e))/(4*e**5)
        omega_sq=2*((3-2*e*e)*q*np.arcsin(e)/e-3*(1-e*e))/(e*e)
        beta=-1+3/(2*e*e)-3*q/(2*e*np.arcsin(e))
        np.testing.assert_allclose(meta['b11'],b,rtol=1e-10)
        np.testing.assert_allclose(meta['omega']**2,omega_sq,rtol=1e-10)
        np.testing.assert_allclose(meta['beta'],beta,rtol=1e-10)


def test_maclaurin_sphere_neutrality_and_dynamical_growth():
    from nmbackend.modes import bar_mode,neutrality_thresholds
    p,meta=bar_mode(0.)
    assert_roots(solve(p).sigma,[-np.sqrt(16/15),np.sqrt(16/15)])
    thresholds=neutrality_thresholds()
    # Published thresholds are rounded to four decimals; do not invent a
    # higher-precision target from a rounded 0.2738 reference.
    assert round(thresholds['secular_pattern_e'],4)==.8127
    assert round(thresholds['secular_pattern_beta'],4)==.1375
    assert round(thresholds['dynamical_e'],4)==.9529
    assert round(thresholds['dynamical_beta'],4)==.2738
    p,meta=bar_mode(.97)
    roots=solve(p).sigma
    assert min(roots.imag)<0<max(roots.imag)
    np.testing.assert_allclose(roots.real,-meta['omega'],rtol=1e-12)
    assert meta['radiation']=='not included'
