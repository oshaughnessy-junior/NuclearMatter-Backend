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
  ``eos.nb`` starts with two lines, the indices of its first and last point, then
  one n_B per line; row ``i_nb`` of ``eos.thermo`` is point ``i_nb`` of that range.
  Tested on the real CompOSE eos/116 and eos/188 files (tests/data/compose_*).

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
def _numbers(line_or_text):
    """Floats of a whitespace-separated text; Fortran ``D`` exponents allowed."""
    return [float(t.replace("D", "E").replace("d", "e")) for t in line_or_text.split()]


def read_compose_nb(path):
    """(index of the first point, n_B array) from a CompOSE ``eos.nb``.

    The standard file (CompOSE manual v3) has two header lines, the first and the
    last point index, then ``last - first + 1`` values.  The one-line form (point
    count, then the values) is what :func:`write_compose` wrote before 2026-10-25
    and is still read.  Anything else raises ``ValueError``."""
    with open(path) as fh:
        v = _numbers(fh.read())
    if len(v) >= 3 and v[0].is_integer() and v[1].is_integer() \
            and len(v) - 2 == int(v[1]) - int(v[0]) + 1:
        return int(v[0]), np.asarray(v[2:], float)
    if len(v) >= 2 and v[0].is_integer() and int(v[0]) == len(v) - 1:
        return 1, np.asarray(v[1:], float)
    head = v[:2]
    raise ValueError(f"{path}: header {head} matches neither 'first, last' nor 'count' "
                     f"for {len(v)} numbers")


def read_compose(path_dir, thermo="eos.thermo", nb="eos.nb"):
    """Read CompOSE eos.nb + eos.thermo (cold, 1D beta-eq) -> (n, eps, p, m_n).

    Raises ``ValueError`` if the table has more than one temperature or Y_q index,
    or if an n_B index is out of the ``eos.nb`` range or repeated."""
    nb_path = os.path.join(path_dir, nb) if os.path.isdir(path_dir) else nb
    th_path = os.path.join(path_dir, thermo) if os.path.isdir(path_dir) else thermo

    i_first, n_arr = read_compose_nb(nb_path)
    with open(th_path) as fh:
        header = fh.readline().split()
        m_n = _numbers(header[0])[0] if header else M_N
        rows = [_numbers(ln) for ln in fh if ln.strip()]
    rows = np.array([r[:11] for r in rows if len(r) >= 11])
    if rows.size == 0:
        raise ValueError(f"{th_path}: no data rows with >= 11 columns")
    if np.unique(rows[:, 0]).size != 1 or np.unique(rows[:, 2]).size != 1:
        raise ValueError(f"{th_path}: not a 1D table (several T or Y_q indices)")
    k = rows[:, 1].astype(int) - i_first         # i_nb counts from eos.nb's first index
    if np.any(k < 0) or np.any(k >= n_arr.size) or np.unique(k).size != k.size:
        raise ValueError(f"{th_path}: n_B indices out of the eos.nb range "
                         f"[{i_first}, {i_first + n_arr.size - 1}] or repeated")
    Q1 = rows[:, 3]                              # p / n_B [MeV] (col index 3)
    Q7 = rows[:, 9]                              # e/(n_B m_n) - 1 (col index 9)
    n = n_arr[k]
    p = Q1 * n                                   # MeV/fm^3
    eps = (Q7 + 1.0) * n * m_n                   # MeV/fm^3 (incl. rest mass)
    order = np.argsort(n)
    return n[order], eps[order], p[order], m_n


def _count_header_lines(path, sep=","):
    """Number of leading lines of ``path`` that are not data rows.

    MUSES CMF++ writes its CSV tables **headerless** (``pandas.to_csv(header=False)``
    in ``clean_output.py``), so for a real CMF file this returns 0.  A table with a
    text header row (e.g. a hand-made or CompOSE-style export) returns the number of
    lines to skip so that :func:`numpy.genfromtxt` starts on the first data row.
    A UTF-8 BOM (Excel / Windows round-trip) is ignored, and the all-integer
    ``0,1,2,...`` header that ``pandas.to_csv`` emits for an unnamed DataFrame is
    treated as a header, not as a data row.  Stops at the first data row.
    """
    with open(path, encoding="utf-8-sig") as fh:
        for i, line in enumerate(fh):
            toks = line.strip().split(sep)
            try:
                vals = [float(tok) for tok in toks]
            except ValueError:
                continue
            if len(vals) > 1 and vals == list(range(len(vals))):
                continue                                     # pandas default header
            return i
    return 0


def read_cmf_csv(path, col_nb=5, col_eps=8, col_p=9, sep=",", skip_header=None):
    """Read a MUSES CMF (or CompOSE-style) CSV EOS table -> (n, eps, p).

    MUSES CMF++ (v1.1.0) output CSVs are **intentionally headerless**; the
    ``CMF_output_{stable,metastable,unstable}.csv`` layout (1-based columns,
    ``api/OpenAPI_Specifications_CMF.yaml`` schema ``CMF_output_all_EoS_variables``) is

        1 T [MeV]   2 mu_B   3 mu_S   4 mu_Q [MeV]
        5 n_B   6 n_S   7 n_Q [1/fm^3]   8 eps   9 p [MeV/fm^3]   10 s [1/fm^3]
        11 sigma  12 zeta  13 delta  14 omega  15 phi  16 rho [MeV]  17 Phi_order
        18 n_B_without_Phi_order  19 n_quark  20 n_octet  21 n_decuplet [1/fm^3]

    The ``CMF_output_for_Lepton_*.csv`` files carry only columns 1-10 plus 18 (11
    columns), so the (5, 8, 9) default mapping is the same for both families.  Column 5
    is the full baryon density (including the Polyakov-loop ``Phi_order`` term in
    mu_B; it is what pairs thermodynamically with eps and p), column 18 excludes it
    (Cruz-Camacho et al. 2409.06837; docs/muses-integration.md).

    ``skip_header=None`` (default) auto-detects: leading non-numeric lines are skipped,
    so a real headerless CMF file loses no rows (the first row of the default C4 run is
    the mu_B = 922 MeV nuclear-matter onset) while a file with a text header still
    reads correctly.  Pass an int to force a fixed number of skipped lines.
    The CMF grid runs over (mu_B, mu_S, mu_Q), so n_B can repeat / be non-monotone and
    the Lepton files include metastable points with p < 0; we keep the physical
    (n, eps, p > 0, finite), sorted, strictly-increasing-density points (for the cold
    beta-eq NS EOS, point CMF at its Lepton/Synthesis output along the NS line).
    """
    if skip_header is None:
        skip_header = _count_header_lines(path, sep)
    d = np.genfromtxt(path, delimiter=sep, skip_header=skip_header, encoding="utf-8-sig")
    d = np.atleast_2d(d)
    need = max(col_nb, col_eps, col_p)
    if d.size == 0 or d.shape[1] < need:
        raise ValueError(f"{path}: expected >= {need} numeric columns after skipping "
                         f"{skip_header} header line(s); got shape {d.shape}")
    n, eps, p = d[:, col_nb - 1], d[:, col_eps - 1], d[:, col_p - 1]
    ok = (n > 0) & (eps > 0) & (p > 0) & np.isfinite(n) & np.isfinite(eps) & np.isfinite(p)
    if not np.any(ok):
        raise ValueError(f"{path}: no rows with finite n_B, eps, p > 0 in columns "
                         f"({col_nb}, {col_eps}, {col_p}); is this a CMF EOS table?")
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
        fh.write(f"1\n{n.size}\n")              # first and last point index
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
def eos_table_from_arrays(n, eps, p, n_surface=1e-7, gamma_crust=1.3,
                          xp=None) -> EOSTable:
    """Build an EOSTable from (n[fm^-3], eps, p [MeV/fm^3]); optional surface tail.
    ``xp`` is passed to :func:`build_h_and_cs2`.

    eps and p are made non-decreasing by a running maximum.  The table starts at
    the first row with p > 0; rows below it are dropped.  A run of equal pressures
    at the bottom (from the running maximum) is a layer of zero thickness and is
    collapsed onto its last row, so the first cell has a positive h-width.
    Positive pressures are not floored.  Until 2026-10-25 p was floored at 1e-12
    MeV/fm^3: a table with two or more rows below that started with cells of zero
    h-width, and the TOV solve ended (at h[0]) inside one of them, where the
    tidal variable's dh/dx is 0/0, so Lambda was NaN while M and R stayed finite
    (docs/devlog/2026-10-25-compose-header-floor.md)."""
    n = np.asarray(n, float)
    eps = np.asarray(eps, float)
    p = np.asarray(p, float)
    pos = np.flatnonzero(p > 0)
    if pos.size == 0:
        raise ValueError("eos_table_from_arrays: no row with p > 0")
    n, eps, p = n[pos[0]:], eps[pos[0]:], p[pos[0]:]
    eps = np.maximum.accumulate(eps)
    p = np.maximum.accumulate(p)
    j = int(np.searchsorted(p, p[0], side="right")) - 1   # last row of the bottom run
    n, eps, p = n[j:], eps[j:], p[j:]
    if p.size < 2:
        raise ValueError("eos_table_from_arrays: fewer than two rows with distinct p > 0")
    if n_surface is not None and n[0] > n_surface:
        n, eps, p = extend_to_surface(n, eps, p, n_surface=n_surface,
                                      gamma_crust=gamma_crust)
    if xp is None:
        return build_h_and_cs2(n, eps, p)
    return build_h_and_cs2(n, eps, p, xp=xp)


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
