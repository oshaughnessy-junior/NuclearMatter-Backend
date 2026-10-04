"""Native thermal geometry against an exact GR interior, roots against algebra."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from nmbackend.eos.base import EOSTable
from nmbackend.eos.coolcomp import CoolingComposition
from nmbackend.tov import solve_profile
from nmbackend.cooling.physics import core_nu, heat
from nmbackend.cooling.physics.native import inner_core_sums,onset_density


def composition(n, xp):
    return CoolingComposition(n,1.-xp,xp,xp,jnp.zeros_like(n),
                              jnp.full_like(n,.8),jnp.full_like(n,.8))


@pytest.mark.parametrize('xp',[.05,.16])
def test_inner_integrals_match_exact_interior(xp):
    h=jnp.linspace(1e-10,1.,2049)
    eps=jnp.full_like(h,1e-3)
    p=eps*jnp.expm1(h)
    table=EOSTable(jnp.ones_like(h),eps,p,h,jnp.ones_like(h),eps,p)
    hc=.3
    profile=solve_profile(table,hc,n_points=1024,rtol=1e-9,atol=1e-11)
    comp=composition(jnp.array([.01,2.]),jnp.full(2,xp))
    sums=inner_core_sums(table,profile,hc,comp,3e8,n_cut=.16)
    x=np.expm1(hc)
    surface=(1+x)/(1+3*x)
    C=(1-surface**2)/2
    R=np.sqrt(3*C/(4*np.pi*1e-3))
    metric=lambda r:np.sqrt(1-2*C*(r/R)**2)
    lapse=lambda r:.5*(3*surface-metric(r))
    integrate=lambda f:quad(lambda r:4*np.pi*r*r*f(r)/metric(r),0,R,epsabs=1e-6,epsrel=1e-10)[0]*1e15
    cv0=float(heat.cv_coeff((3*np.pi**2*(1-xp))**(1/3),939.56*.8)
              +heat.cv_coeff((3*np.pi**2*xp)**(1/3),938.27*.8)
              +heat.lepton_cv_coeff((3*np.pi**2*xp)**(1/3),.511))
    np.testing.assert_allclose(sums.volume_cm3,integrate(lambda r:1.),rtol=2e-5)
    np.testing.assert_allclose(sums.C,cv0*3e8*integrate(lambda r:1/lapse(r)),rtol=2e-5)
    if xp<1/9:
        assert float(sums.L_durca)==0.
    else:
        q0=float(core_nu.durca_unpaired_channel(3e8,.8,.8,1.,xp))
        np.testing.assert_allclose(sums.L_durca,q0*integrate(lambda r:lapse(r)**-4),rtol=2e-5)


def test_moving_threshold_matches_npe_algebra():
    n=jnp.linspace(.16,1.,50)
    def threshold(a):
        comp=composition(n,a+.12*n)
        return onset_density(comp,.16,1.)
    a=.04
    np.testing.assert_allclose(threshold(a),(1/9-a)/.12,rtol=1e-12)
    np.testing.assert_allclose(jax.jacfwd(threshold)(a),-1/.12,rtol=1e-10)


def test_moving_threshold_shell_volume_derivative():
    from nmbackend.cooling.physics.native import _shell_integral
    n=jnp.linspace(.16,1.,50)
    r=jnp.array([0.,.4,.8,1.2])
    def volume(a):
        threshold=onset_density(composition(n,a+.12*n),.16,1.)
        return _shell_integral(r,jnp.ones_like(r),(1.3-threshold)**1.5)
    exact=lambda a:4*jnp.pi/3*1e15*(1.3-(1/9-a)/.12)**1.5
    np.testing.assert_allclose(volume(.04),exact(.04),rtol=1e-12)
    np.testing.assert_allclose(jax.grad(volume)(.04),jax.grad(exact)(.04),rtol=1e-10)
