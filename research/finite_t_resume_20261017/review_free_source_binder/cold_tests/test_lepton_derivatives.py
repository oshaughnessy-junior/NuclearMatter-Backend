"""Cold Fermi-gas thermodynamic identities test the implicit lepton Hessian."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from nmbackend.constants import HBARC,M_E,M_MU
from nmbackend.eos.leptons import lepton_sector


@pytest.mark.parametrize('muons',[False,True])
@pytest.mark.parametrize('n',[.001,.02,.1,.3])
def test_energy_hessian_matches_compressibility(n,muons):
    energy=lambda charge:lepton_sector(charge,muons=muons)[3]
    mu,ne,nmu,_=lepton_sector(n,muons=muons)
    np.testing.assert_allclose(ne+nmu,n,rtol=2e-13)
    np.testing.assert_allclose(jax.grad(energy)(n),mu,rtol=2e-10)
    ke=np.sqrt(float(mu)**2-(M_E/HBARC)**2)
    km=np.sqrt(max(float(mu)**2-(M_MU/HBARC)**2,0.)) if muons else 0.
    expected=np.pi**2/(float(mu)*(ke+km))
    np.testing.assert_allclose(jax.grad(jax.grad(energy))(n),expected,rtol=2e-9)


def test_vector_implicit_hessian_and_jit():
    n=jnp.array([.001,.02,.3])
    energy=lambda charge:lepton_sector(charge,muons=True)[3].sum()
    hessian=jax.jit(jax.hessian(energy))(n)
    mu=np.asarray(lepton_sector(n)[0])
    ke=np.sqrt(mu**2-(M_E/HBARC)**2)
    km=np.sqrt(np.maximum(mu**2-(M_MU/HBARC)**2,0.))
    np.testing.assert_allclose(hessian,np.diag(np.pi**2/(mu*(ke+km))),rtol=2e-9,atol=1e-12)
