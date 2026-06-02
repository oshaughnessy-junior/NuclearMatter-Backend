"""EOS-level (model) cross-code diagnostic -- compares the EOS *itself*, not just the
TOV observables.

Two EOS (parametric models or ingested tables, anything exposing an ``EOSTable``)
are evaluated on a common baryon-density grid and their thermodynamic quantities are
tabulated with relative differences:

    eps(n)   energy density          [MeV/fm^3]
    p(n)     pressure                [MeV/fm^3]
    mu(n)    baryon chem. potential  = (eps+p)/n   [MeV-ish, geometric]
    cs2(n)   sound speed squared     dp/deps

This catches model/convention disagreements (couplings, units, composition) that an
M-R-Lambda comparison can hide, and is the natural way to cross-check against an
external code's EOS table (e.g. MUSES cmf_solver) before ever solving a star.
"""
from __future__ import annotations

import numpy as np

from ..eos.base import EOSTable
from ..constants import MEVFM3_TO_GEOM


def _interp_loglog(n_q, n, y):
    """log-log interpolation of y(n) at n_q (n, y > 0, n increasing)."""
    return np.exp(np.interp(np.log(n_q), np.log(n), np.log(np.maximum(y, 1e-300))))


def _quantities(table: EOSTable, n_q):
    n = np.asarray(table.n)
    eps_g = np.asarray(table.eps)            # geometric km^-2
    p_g = np.asarray(table.p)
    cs2 = np.asarray(table.cs2)
    eps = _interp_loglog(n_q, n, eps_g) / MEVFM3_TO_GEOM      # -> MeV/fm^3
    p = _interp_loglog(n_q, n, p_g) / MEVFM3_TO_GEOM
    mu = (eps + p) / n_q                                      # MeV/fm^3 per fm^-3
    cs2_q = np.interp(np.log(n_q), np.log(n), cs2)
    return dict(eps=eps, p=p, mu=mu, cs2=cs2_q)


def model_diagnostic(table_a: EOSTable, table_b: EOSTable, n_lo=0.05, n_hi=1.0,
                     npts=12, label_a="A", label_b="B", verbose=True):
    """Compare two EOS tables on a common density grid.  Returns a dict of arrays
    and the max relative differences; prints a table if ``verbose``."""
    n_q = np.geomspace(n_lo, n_hi, npts)
    A, B = _quantities(table_a, n_q), _quantities(table_b, n_q)
    out = {"n": n_q}
    maxrel = {}
    for q in ("eps", "p", "mu", "cs2"):
        rel = (A[q] - B[q]) / np.where(np.abs(B[q]) > 1e-30, B[q], 1e-30)
        out[q + "_a"], out[q + "_b"], out[q + "_rel"] = A[q], B[q], rel
        maxrel[q] = float(np.max(np.abs(rel)))
    out["maxrel"] = maxrel

    if verbose:
        print(f"EOS model diagnostic: {label_a} vs {label_b}   "
              f"(relative diff = ({label_a}-{label_b})/{label_b})")
        print(f"{'n':>7} {'eps_'+label_a[:3]:>10} {'eps_'+label_b[:3]:>10} "
              f"{'d_eps':>9} {'d_p':>9} {'d_mu':>9} {'d_cs2':>9}")
        for i, n in enumerate(n_q):
            print(f"{n:7.3f} {A['eps'][i]:10.2f} {B['eps'][i]:10.2f} "
                  f"{out['eps_rel'][i]:9.1e} {out['p_rel'][i]:9.1e} "
                  f"{out['mu_rel'][i]:9.1e} {out['cs2_rel'][i]:9.1e}")
        print(f"max |rel diff|:  eps {maxrel['eps']:.2e}  p {maxrel['p']:.2e}  "
              f"mu {maxrel['mu']:.2e}  cs2 {maxrel['cs2']:.2e}")
    return out
