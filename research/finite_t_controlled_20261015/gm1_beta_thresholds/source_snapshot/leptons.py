"""Cold lepton sector (electrons, optionally muons) at fixed charge density.

Shared by the RMFT and metamodel EOSs.  Units: fm (hbar c = 1), i.e. momenta and
chemical potentials in fm^-1, densities in fm^-3, energy densities in fm^-4.
"""
from __future__ import annotations

import math
import jax
import jax.numpy as jnp

from ..constants import HBARC, M_E, M_MU

_PI2 = math.pi ** 2


def eps_kin_one(kf, m):
    """Kinetic + rest energy density of one degeneracy-2 Fermi gas (fm^-4)."""
    ef = jnp.sqrt(kf**2 + m**2)
    return (1.0 / (8.0 * _PI2)) * (
        kf * (2.0 * kf**2 + m**2) * ef - m**4 * jnp.log((kf + ef) / m))


def _kf_of_mu(mu, m):
    """sqrt(mu^2 - m^2) for mu > m, else 0; NaN-free derivative at threshold."""
    x = mu**2 - m**2
    return jnp.where(x > 0.0, jnp.sqrt(jnp.where(x > 0.0, x, 1.0)), 0.0)


def lepton_sector(n_charge, muons=True, n_bis=50):
    """Leptons in chemical equilibrium (mu_e = mu_mu) with n_e + n_mu = n_charge.

    Muons switch on once mu_e > m_mu (~106 MeV); ``muons=False`` gives the
    electron-only (npe) limit. Bisection plus a Newton polish gives the primal;
    ``custom_root`` supplies implicit derivatives, including the energy Hessian
    needed when differentiating pressure. Threshold contacts are excluded.

    Returns (mu_e [fm^-1], n_e, n_mu [fm^-3], eps_lep [fm^-4]).
    """
    m_e, m_mu = M_E / HBARC, M_MU / HBARC
    n_charge = jnp.maximum(n_charge, 1e-30)
    mu_on = 1.0 if muons else 0.0

    def nlep(mu):
        ke, kmu = _kf_of_mu(mu, m_e), _kf_of_mu(mu, m_mu)
        return (ke**3 + mu_on * kmu**3) / (3.0 * _PI2)

    mu_lo = jnp.full_like(n_charge, m_e)
    mu_hi = jnp.sqrt((3.0 * _PI2 * n_charge) ** (2.0 / 3.0) + m_e**2) * 1.0001

    def body(_i, c):
        lo, hi = c
        mid = 0.5 * (lo + hi)
        f = nlep(mid) - n_charge           # increasing in mu
        return jnp.where(f < 0.0, mid, lo), jnp.where(f < 0.0, hi, mid)

    def solve(fun, initial):
        lo, hi = jax.lax.fori_loop(0, n_bis, body, (mu_lo, mu_hi))
        mu0 = 0.5 * (lo + hi)
        dn = jax.grad(lambda m: jnp.sum(nlep(m)))(mu0)
        return mu0 - fun(mu0) / dn

    # Each charge-density entry has an independent scalar root: the linearized
    # Jacobian is diagonal. A stop-gradient start plus one Newton polish recovers
    # only the first derivative; pressure sensitivities need the true second one.
    mu_e = jax.lax.custom_root(lambda mu: nlep(mu)-n_charge,
                              0.5*(mu_lo+mu_hi), solve,
                              lambda g, y: y/g(jnp.ones_like(y)))
    ke = _kf_of_mu(mu_e, m_e)
    kmu = mu_on * _kf_of_mu(mu_e, m_mu)
    n_e = ke**3 / (3.0 * _PI2)
    n_mu = kmu**3 / (3.0 * _PI2)
    eps_lep = eps_kin_one(ke, m_e) + eps_kin_one(kmu, m_mu)   # =0 if kf=0
    return mu_e, n_e, n_mu, eps_lep

