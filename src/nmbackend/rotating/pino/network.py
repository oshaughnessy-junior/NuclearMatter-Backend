"""Independent JAX implementation of Ray (2611.05790v1), Eqs. 37–50.

No upstream code or weights used. Based on the published architecture; see
ATTRIBUTION.md. Parameters are ordinary JAX pytrees, with float64 arrays.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


def _mlp_init(key, dims):
    keys = jax.random.split(key, len(dims)-1)
    return tuple((jax.random.normal(k, (n_out, n_in), dtype=jnp.float64)
                  * jnp.sqrt(1.0/n_in), jnp.zeros(n_out, dtype=jnp.float64))
                 for k, n_in, n_out in zip(keys, dims[:-1], dims[1:]))


def _mlp(layers, x):
    h = jnp.tanh(layers[0][0] @ x + layers[0][1])
    for index, (weight, bias) in enumerate(layers[1:-1], start=1):
        transformed = jnp.tanh(weight @ h + bias)
        h = h + transformed if index % 2 else transformed
    return layers[-1][0] @ h + layers[-1][1]


def initialize(key, eos_points=16, width=24, hidden_layers=2, modes=12):
    """Initialize a weak spherical seed plus small learned metric corrections."""
    kb, kt = jax.random.split(key)
    return {
        'branch': _mlp_init(kb, [2*eos_points+2] + [width]*(hidden_layers+1) + [3*modes]),
        'trunk': _mlp_init(kt, [2] + [width]*(hidden_layers+1) + [modes]),
        'scale': jnp.full(3, .01, dtype=jnp.float64),
        'offset': jnp.array([-.02, -1.0, .03], dtype=jnp.float64),
    }


def evaluate(params, branch_input, point):
    """Return (gamma/r_e², rho/r_e², r_e*omega) at (s, mu).

    This independent variant also enforces a latitude-independent centre and
    zero first radial derivative there. It uses a smooth radial coordinate and
    decay factor instead of the paper's literal (1-s) prefactor. The Green-map
    residual and unsupervised training procedure are unchanged.
    """
    s, mu = point
    q = branch_input[-1]
    normalized = branch_input.at[-2].set((branch_input[-2]-.3)/.15)
    normalized = normalized.at[-1].set((q-.885)/.085)
    coefficients = _mlp(params['branch'], normalized).reshape(3, -1)
    radial_coordinate = s*s/(s*s+(1-s)**2)
    basis = (jnp.tanh(_mlp(params['trunk'], jnp.array([radial_coordinate, mu*mu])))
             - jnp.tanh(_mlp(params['trunk'], jnp.array([0., mu*mu])))
             + jnp.tanh(_mlp(params['trunk'], jnp.array([0., 0.]))))
    decay = (1-s)/jnp.sqrt(s*s+(1-s)**2)
    # The D4 inverse fixes gamma's monopole decay to r^-2; rho has r^-1
    # and frame dragging r^-3. Enforce all three asymptotic powers exactly.
    prefactor = jnp.array([decay**2, decay, decay**3*jnp.sqrt(jnp.maximum(1-q, 0.))])
    return prefactor * (params['offset'] + params['scale'] * (coefficients @ basis))
