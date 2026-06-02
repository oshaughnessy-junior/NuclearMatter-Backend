"""Ingestion of tabulated EOS, incl. the CompOSE format (compose.obspm.fr).

Reads a cold, beta-equilibrium 1D EOS table into the standard :class:`EOSTable`, so
any community / MUSES-generated table flows through the same thermodynamically
consistent h(mu)-interpolation + TOV pipeline as the parametric models.

Two entry points:

* :func:`load_compose` -- the CompOSE ``eos.nb`` + ``eos.thermo`` pair (the standard
  distribution format).  Column convention (Typel et al. CompOSE manual): the
  ``eos.thermo`` first line is ``m_n m_p I_flag``; each data row is
  ``i_t i_nb i_yq Q1 Q2 Q3 Q4 Q5 Q6 Q7 N_add [extra...]`` with
  ``Q1 = p/n_B`` [MeV] and ``Q7 = e/(n_B m_n) - 1`` (internal energy per baryon,
  rest mass included).  Hence p = Q1 n_B and eps = (Q7+1) n_B m_n.
  NOTE: verify against a real CompOSE download -- the eos.nb header and the exact
  thermo columns vary slightly between releases; this targets the documented v2/v3
  format and is self-consistent with :func:`write_compose` (round-trip tested).

* :func:`load_simple` -- an unambiguous 3-column ``n_B[fm^-3]  eps[MeV/fm^3]
  p[MeV/fm^3]`` text table, for derived tables where the CompOSE columns are not at
  hand.

``write_compose`` emits the eos.nb/eos.thermo pair (for round-trip validation and
for exporting our parametric EOS to CompOSE consumers).
"""
from __future__ import annotations

import os
import numpy as np

from ..constants import M_N
from .base import build_h_and_cs2, extend_to_surface, EOSModel, EOSTable


# ---------------- low-level readers (return n[fm^-3], eps, p [MeV/fm^3]) -----------
def read_compose(path_dir, thermo="eos.thermo", nb="eos.nb"):
    """Read CompOSE eos.nb + eos.thermo (cold, 1D beta-eq) -> (n, eps, p, m_n)."""
    nb_path = os.path.join(path_dir, nb) if os.path.isdir(path_dir) else nb
    th_path = os.path.join(path_dir, thermo) if os.path.isdir(path_dir) else thermo

    # eos.nb : a header line (point count) then the n_B values, one per line.
    nb_tokens = np.loadtxt(nb_path, ndmin=1)
    n_arr = nb_tokens[1:] if nb_tokens.size > 1 and float(
        nb_tokens[0]).is_integer() and int(nb_tokens[0]) == nb_tokens.size - 1 \
        else nb_tokens

    with open(th_path) as fh:
        header = fh.readline().split()
        m_n = float(header[0]) if header else M_N
        rows = [list(map(float, ln.split())) for ln in fh if ln.strip()]
    rows = np.array([r for r in rows if len(r) >= 11])
    i_nb = rows[:, 1].astype(int) - 1            # 1-based index into eos.nb
    Q1 = rows[:, 3]                              # p / n_B [MeV] (col index 3)
    Q7 = rows[:, 9]                              # e/(n_B m_n) - 1 (col index 9)
    n = n_arr[i_nb]
    p = Q1 * n                                   # MeV/fm^3
    eps = (Q7 + 1.0) * n * m_n                   # MeV/fm^3 (incl. rest mass)
    order = np.argsort(n)
    return n[order], eps[order], p[order], m_n


def read_cmf_csv(path, col_nb=5, col_eps=8, col_p=9, sep=",", skip_header=1):
    """Read a MUSES CMF (or CompOSE-style) CSV EOS table -> (n, eps, p).

    The MUSES CMF output schema (CMF_output_stable.csv / _for_Lepton_*.csv) has, in
    1-based columns: ... 5=baryon density [1/fm^3], 8=energy density [MeV/fm^3],
    9=pressure [MeV/fm^3] (Cruz-Camacho et al. 2409.06837; docs/muses-integration.md).
    The CMF grid runs over (mu_B, mu_S, mu_Q), so n_B can repeat / be non-monotone;
    we keep the physical, sorted, strictly-increasing-density points (for the cold
    beta-eq NS EOS, point CMF at its Lepton/Synthesis output along the NS line).
    """
    d = np.genfromtxt(path, delimiter=sep, skip_header=skip_header)
    d = np.atleast_2d(d)
    n, eps, p = d[:, col_nb - 1], d[:, col_eps - 1], d[:, col_p - 1]
    ok = (n > 0) & (eps > 0) & np.isfinite(n) & np.isfinite(eps) & np.isfinite(p)
    n, eps, p = n[ok], eps[ok], p[ok]
    o = np.argsort(n)
    n, eps, p = n[o], eps[o], p[o]
    keep = np.concatenate([[True], np.diff(n) > 1e-12])     # dedupe repeated n_B
    return n[keep], eps[keep], p[keep]


def read_simple(path):
    """Read a 3-column n_B[fm^-3] eps[MeV/fm^3] p[MeV/fm^3] table -> (n, eps, p)."""
    d = np.loadtxt(path, comments="#", ndmin=2)
    n, eps, p = d[:, 0], d[:, 1], d[:, 2]
    o = np.argsort(n)
    return n[o], eps[o], p[o]


# ---------------- writer (for round-trip / export) ----------------
def write_compose(path_dir, n, eps, p, m_n=M_N, m_p=M_N):
    """Write eos.nb + eos.thermo for a cold 1D EOS (n[fm^-3], eps, p [MeV/fm^3])."""
    os.makedirs(path_dir, exist_ok=True)
    n, eps, p = map(np.asarray, (n, eps, p))
    with open(os.path.join(path_dir, "eos.nb"), "w") as fh:
        fh.write(f"{n.size}\n")
        for v in n:
            fh.write(f"{v:.10e}\n")
    Q1 = p / n
    Q7 = eps / (n * m_n) - 1.0
    z = np.zeros_like(n)
    with open(os.path.join(path_dir, "eos.thermo"), "w") as fh:
        fh.write(f"{m_n:.6f} {m_p:.6f} 1\n")
        for j in range(n.size):
            fh.write(f"1 {j + 1} 1 {Q1[j]:.10e} {z[j]:.1e} {z[j]:.1e} {z[j]:.1e} "
                     f"{z[j]:.1e} {z[j]:.1e} {Q7[j]:.10e} 0\n")


# ---------------- table / model construction ----------------
def eos_table_from_arrays(n, eps, p, n_surface=1e-7, gamma_crust=1.3) -> EOSTable:
    """Build an EOSTable from (n[fm^-3], eps, p [MeV/fm^3]); optional surface tail."""
    n = np.asarray(n, float)
    eps = np.maximum.accumulate(np.asarray(eps, float))
    p = np.maximum.accumulate(np.maximum(np.asarray(p, float), 1e-12))
    if n_surface is not None and n[0] > n_surface:
        n, eps, p = extend_to_surface(n, eps, p, n_surface=n_surface,
                                      gamma_crust=gamma_crust)
    return build_h_and_cs2(n, eps, p)


def load_compose(path_dir, **kw) -> EOSTable:
    n, eps, p, _ = read_compose(path_dir)
    return eos_table_from_arrays(n, eps, p, **kw)


def load_simple(path, **kw) -> EOSTable:
    return eos_table_from_arrays(*read_simple(path), **kw)


def load_cmf(path, **kw) -> EOSTable:
    """Load a MUSES CMF CSV EOS table into an EOSTable (see :func:`read_cmf_csv`)."""
    return eos_table_from_arrays(*read_cmf_csv(path), **kw)


class TabularEOS(EOSModel):
    """Wrap a loaded EOSTable as an EOSModel (for the RIFT adapter, families, etc.)."""
    name = "TabularEOS"

    def __init__(self, table: EOSTable, name=None):
        self._table = table
        if name:
            self.name = name

    def eos_table(self, params: dict = None):
        return self._table
