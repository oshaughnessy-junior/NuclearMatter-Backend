"""Private fixed-coupling GM1 implicit responses; internal fm units."""
from __future__ import annotations
import math
import numpy as np


def constraint_jacobian(phi, n, p, M, Cs, b, c):
    upp = 1/Cs+2*b*M*phi+3*c*phi*phi
    return np.array([[upp+n.scalar_mass+p.scalar_mass, -n.scalar_nu, -p.scalar_nu],
                     [n.scalar_nu, n.chi, 0.], [p.scalar_nu, 0., p.chi]])


def response(phi, n, p, M, Cs, Co, Cr, b, c, nB, Yp, hbarc):
    """Full scalar relaxation and NET-pair constraint, not fixed-mass Cv."""
    upp = 1/Cs+2*b*M*phi+3*c*phi*phi
    D = upp+sum(g.scalar_mass+g.scalar_nu**2/g.chi for g in (n, p))
    if not math.isfinite(D) or abs(D) <= 1e-8:
        raise ArithmeticError("Near-singular canonical scalar response")
    a = np.array([n.scalar_nu/n.chi, p.scalar_nu/p.chi])
    H = np.diag([1/n.chi, 1/p.chi])-np.outer(a, a)/D
    H += np.array([[Co+Cr/4, Co-Cr/4], [Co-Cr/4, Co+Cr/4]])
    J = constraint_jacobian(phi, n, p, M, Cs, b, c)
    scaled = J/np.array([nB, nB*(1-Yp), nB*Yp])[:, None]
    cond = float(np.linalg.cond(scaled))
    K = sum(g.scalar_temperature-g.scalar_nu*g.n_temperature/g.chi for g in (n, p))
    # Temperature factors are supplied by caller after checking internal t.
    entropy_t_frozen_m = sum(g.s_temperature-g.n_temperature**2/g.chi for g in (n, p))
    return dict(D=float(D), H=H*hbarc, condition=cond, phi_t=float(K/D),
                entropy_t=float(entropy_t_frozen_m+K*K/D),
                entropy_t_frozen_mass=float(entropy_t_frozen_m), jacobian=J)
