"""NSCool star EOS file ("NEW" version, like NSCool's EOS/APR_EOS_Cat.dat).

Layout, read by NSCool ``precool.f`` (get_core_chemistry) and ``TOV/TOV.f``
(subroutine eos):

  * line 1: ``6 Imax Icore``; lines 2-7 text (TOV.f skips exactly 6 lines);
  * Icore core rows, densest first, 17 columns: rho [g cm^-3], P [dyn cm^-2],
    n_B [fm^-3], Ye, Ymu, Yn, Yp, Yla, Ysm, Ys0, Ysp, then m*/m of p, n, la,
    sm, s0, sp (hyperon columns zero);
  * crust rows, densest first: rho, P, n_B.

The rows are the nodes of the model's own EOSTable, so TOV.f integrates the
same p(eps) as nmbackend.  The table must be built with the SLy crust
(``crust="sly"``, the paper-6 default) at join density ``n_t``.  Core rows
are the uniform-matter nodes above the crust-core power-law segment of
:func:`nmbackend.eos.crust.attach_crust`; their composition is interpolated
(linear in n) from a :class:`nmbackend.eos.coolcomp.CoolingComposition`.

Crust rows carry no A_cell, A_ion, Z: TOV.f reads only three columns, and
NSCool takes the crust chemistry from its separate crust file.
"""
from __future__ import annotations

import numpy as np

from .eos.crust import (N_CRUST_PTS, N_SEG_PTS, N_T_DEFAULT,
                        _DYN_CM2_TO_MEVFM3, _G_CM3_TO_MEVFM3)

__all__ = ["write_tables", "read_star_eos", "MAX_CORE_ROWS", "MAX_ROWS"]

MAX_CORE_ROWS = 500       # NSCool precool.f arrays
MAX_ROWS = 1000           # TOV.f read limit
_FMT = "{:13.6E}"
_COLS = ("Rho Press nbar Ye Ymu Yn Yp Yla Ysm Ys0 Ysp "
         "mstp mstn mstla mstsm msts0 mstsp")


def _core_start(n, n_t):
    """Index of the first core node of a SLy-crust table (see module doc)."""
    i = N_CRUST_PTS + N_SEG_PTS
    if len(n) <= i or abs(n[N_CRUST_PTS - 1] / n_t - 1.0) > 0.05:
        raise ValueError(f"table does not have the SLy crust at n_t={n_t}")
    return i


def write_tables(table, composition, crust=N_T_DEFAULT, path="EOS_Cat.dat",
                 title="nmbackend EOS"):
    """Write the NSCool star EOS file for one EOS draw.

    ``table``: EOSTable built with the SLy crust joined at ``crust`` = n_t
    (fm^-3).  ``composition``: CoolingComposition of the same model and
    parameters, covering the core rows.  Returns dict(path, icore, imax,
    n_core, rho_core), where n_core and rho_core are the least dense core row
    (fm^-3, g cm^-3).  Raises ValueError for NaN tables (rejected cores),
    non-monotone rows or tables beyond NSCool's row limits."""
    n = np.asarray(table.n, float)
    rho = np.asarray(table.eps_mevfm3, float) / _G_CM3_TO_MEVFM3
    P = np.asarray(table.p_mevfm3, float) / _DYN_CM2_TO_MEVFM3
    if not np.all(np.isfinite(n) & np.isfinite(rho) & np.isfinite(P)):
        raise ValueError("non-finite EOS table (rejected core)")
    ic = _core_start(n, crust)
    icore, imax = len(n) - ic, len(n)
    if icore > MAX_CORE_ROWS or imax > MAX_ROWS:
        raise ValueError(f"{icore} core / {imax} rows; NSCool limits are "
                         f"{MAX_CORE_ROWS} / {MAX_ROWS}")
    nc = np.asarray(composition.n, float)
    if n[ic] < nc[0] or n[-1] > nc[-1]:
        raise ValueError(f"composition covers n = {nc[0]:.3g}-{nc[-1]:.3g}, "
                         f"core rows {n[ic]:.3g}-{n[-1]:.3g} fm^-3")
    f = lambda a: np.interp(n[ic:], nc, np.asarray(a, float))
    x_p, x_e, x_mu = f(composition.x_p), f(composition.x_e), f(composition.x_mu)
    if np.max(np.abs(x_p - x_e - x_mu)) > 1e-3:
        raise ValueError("composition not charge neutral")
    zero = np.zeros_like(x_p)
    core = np.stack([rho[ic:], P[ic:], n[ic:], x_e, x_mu, 1.0 - x_p, x_p,
                     zero, zero, zero, zero, f(composition.mst_p),
                     f(composition.mst_n), zero, zero, zero, zero], axis=1)
    rows = [" ".join(_FMT.format(v) for v in r) for r in core[::-1]]
    rows += [" ".join(_FMT.format(v) for v in r)
             for r in np.stack([rho[:ic], P[:ic], n[:ic]], axis=1)[::-1]]
    chk = np.array([[float(v) for v in r.split()[:3]] for r in rows])
    if not np.all(np.diff(chk, axis=0) < 0.0):
        raise ValueError("rho, P, n_B not strictly monotone after formatting")
    head = [f"{6:7d}{imax:7d}{icore:7d}          Itext Imax Icore", "",
            f"  {title}", "", "  " + _COLS,
            "  g/cm3  dyne/cm2  #/fm3  (crust rows: rho P nbar only)", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(head + rows) + "\n")
    return dict(path=path, icore=icore, imax=imax, n_core=float(n[ic]),
                rho_core=float(rho[ic]))


def read_star_eos(path):
    """Parse a star EOS file: dict(core=(Icore, 17) array, crust=(Imax-Icore,
    3) array (first three columns), itext, imax, icore)."""
    with open(path) as fh:
        lines = fh.read().splitlines()
    itext, imax, icore = (int(v) for v in lines[0].split()[:3])
    data = [ln.split() for ln in lines[1 + itext:] if ln.strip()][:imax]
    core = np.array([[float(v) for v in r[:17]] for r in data[:icore]])
    crust = np.array([[float(v) for v in r[:3]] for r in data[icore:]])
    return dict(core=core, crust=crust, itext=itext, imax=imax, icore=icore)
