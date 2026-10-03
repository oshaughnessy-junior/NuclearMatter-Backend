"""Cubic splines as NSCool evaluates them (Numerical Recipes SPLINE/SPLINT).

The second derivatives depend only on the tabulated data, so they are
computed once in numpy; evaluation is jnp and differentiable in x.
Outside the table NSCool's SPLINT extrapolates the end interval's cubic,
and so does ``splint``.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = ["spline_y2", "splint", "natural_y2_matrix", "splint2"]


def spline_y2(x, y, yp1=None, ypn=None) -> np.ndarray:
    """NR SPLINE: second derivatives; ``None`` end slope means natural."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    y2, u = np.zeros(n), np.zeros(n)
    if yp1 is not None:
        y2[0] = -0.5
        u[0] = (3.0 / (x[1] - x[0])) * ((y[1] - y[0]) / (x[1] - x[0]) - yp1)
    for i in range(1, n - 1):
        sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
        p = sig * y2[i - 1] + 2.0
        y2[i] = (sig - 1.0) / p
        u[i] = (6.0 * ((y[i + 1] - y[i]) / (x[i + 1] - x[i])
                       - (y[i] - y[i - 1]) / (x[i] - x[i - 1]))
                / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
    qn = un = 0.0
    if ypn is not None:
        qn = 0.5
        un = (3.0 / (x[-1] - x[-2])) * (ypn - (y[-1] - y[-2]) / (x[-1] - x[-2]))
    y2[-1] = (un - qn * u[-2]) / (qn * y2[-2] + 1.0)
    for k in range(n - 2, -1, -1):
        y2[k] = y2[k] * y2[k + 1] + u[k]
    return y2


def splint(xa, ya, y2a, x):
    """NR SPLINT at x; xa ascending.

    Either x has any shape and ya, y2a are 1D, or x is a scalar and ya, y2a
    are tables of rows sharing xa (the result has one value per row).
    """
    xa, ya, y2a = (jnp.asarray(v) for v in (xa, ya, y2a))
    # NR bisection: largest k with xa[k] <= x, clipped to [0, n-2]
    k = jnp.clip(jnp.searchsorted(xa, x, side="right") - 1, 0, xa.shape[0] - 2)
    h = xa[k + 1] - xa[k]
    a = (xa[k + 1] - x) / h
    b = (x - xa[k]) / h
    at = lambda v, j: jnp.take(v, j, axis=-1)
    return (a * at(ya, k) + b * at(ya, k + 1)
            + ((a ** 3 - a) * at(y2a, k) + (b ** 3 - b) * at(y2a, k + 1)) * h ** 2 / 6.0)


def natural_y2_matrix(x) -> np.ndarray:
    """Matrix M with natural-spline second derivatives y2 = M @ y."""
    return np.stack([spline_y2(x, e) for e in np.eye(len(x))], axis=1)


def splint2(x1a, x2a, ya, y2a, m1, x1, x2):
    """NR SPLINT2 (NSCool Tools.f) at scalar (x1, x2).

    Splines along x2 for each x1 row (``y2a``, rows from ``spline_y2``),
    then a natural spline along x1 built at run time (``m1`` =
    ``natural_y2_matrix(x1a)``).
    """
    yy = splint(x2a, ya, y2a, x2)
    return splint(x1a, yy, jnp.asarray(m1) @ yy, x1)
