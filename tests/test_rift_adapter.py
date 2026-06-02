"""RIFT EOSConcrete-compatible adapter surface."""
import numpy as np
from nmbackend.eos import MetaModelEOS
from nmbackend.adapters import EOSConcreteNMB


def _adapter():
    return EOSConcreteNMB(MetaModelEOS(), name="MM-default", n_models=100)


def test_interface_attributes():
    a = _adapter()
    for attr in ("name", "mMaxMsun", "lambda_from_m",
                 "lambda_from_m_vector", "pressure_density_on_grid"):
        assert hasattr(a, attr)
    assert 1.9 < a.mMaxMsun < 2.6


def test_lambda_from_m_scalar_and_vector():
    a = _adapter()
    l14 = a.lambda_from_m(1.4)
    assert 100.0 < l14 < 1200.0
    # accepts SI kg too (RIFT convention)
    from nmbackend.constants import MSUN_SI
    assert abs(a.lambda_from_m(1.4 * MSUN_SI) - l14) / l14 < 1e-6
    vec = a.lambda_from_m_vector(np.array([1.2, 1.4, 1.6]))
    assert vec.shape == (3,)
    assert vec[0] > vec[1] > vec[2]            # decreasing with mass
    assert a.lambda_from_m(a.mMaxMsun + 0.1) == 0.0


def test_pressure_density_on_grid_monotone():
    a = _adapter()
    logrho = np.log10(np.geomspace(a._rho_si[5], a._rho_si[-5], 30))
    logp = a.pressure_density_on_grid(logrho)
    assert np.all(np.diff(logp) > 0)           # pressure rises with density
