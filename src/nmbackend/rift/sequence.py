"""Neutron-star EOS *sequence* I/O for RIFT (and beyond).

Two layers:

* ``NSSequence`` + :func:`write_nss` -- a robust, self-describing, efficient format
  that stores every quantity as a function of **central pseudo-enthalpy h_c** (which
  is monotone along the sequence), so M(h_c), R(h_c), Lambda(h_c) are single-valued
  and branch structure (twin stars / phase transitions) is explicit via a ``stable``
  flag.  Lookups M -> Lambda are **branch-aware**.  Regular ``(n_eos, n_pts,
  n_fields)`` array -> contiguous, lazy-sliceable, vectorizable.  See
  docs/rift-sequence-audit.md.

* :func:`write_landry_legacy` + ``LandryCompat`` -- emit / read the exact
  ``EOSSequenceLandry`` HDF5 used by today's RIFT, so the new backend drops into the
  existing tabular hyperpipeline unchanged.  We write only the **stable branch**, so
  RIFT's single-valued M->Lambda interpolation is correct by construction (fixing the
  twin-star bug without touching RIFT code).
"""
from __future__ import annotations

import abc
import json
import datetime
import numpy as np

try:
    import h5py
    _HAVE_H5PY = True
except Exception:  # pragma: no cover
    _HAVE_H5PY = False

import jax.numpy as jnp

from ..tov.family import build_family
from .. import __version__ as _NMB_VERSION

NSS_FIELDS = ["hc", "M", "R", "Lambda", "M_b", "I", "rho_c", "cs2_c", "stable"]
NSS_UNITS = {"hc": "dimensionless", "M": "Msun", "R": "km", "Lambda": "dimensionless",
             "M_b": "Msun", "I": "g cm^2", "rho_c": "fm^-3", "cs2_c": "c^2",
             "stable": "bool"}

# 1 MeV/fm^3 -> g/cm^3 (i.e. /c^2), and baryon rest-mass density factor (m_b=1.66e-24 g)
_MEVFM3_TO_GCM3 = 1.602176634e33 / (2.99792458e10) ** 2     # ~1.7827e12
_NB_TO_GCM3 = 1e39 * 1.66e-24                                # n[fm^-3] -> g/cm^3


# ---------------- build per-EOS sequence arrays ----------------
def _one_eos(table, n_points=80, frac_lo=0.04, frac_hi=0.95):
    hc_grid = np.linspace(frac_lo * float(table.h[-1]),
                          frac_hi * float(table.h[-1]), n_points)
    fam = build_family(table, jnp.asarray(hc_grid))
    hc = np.asarray(fam.h_c)
    M = np.asarray(fam.M_msun)
    R = np.asarray(fam.R_km)
    Lam = np.asarray(fam.Lambda)
    stable = np.asarray(fam.stable).astype(float)
    hh, n, cs2 = np.asarray(table.h), np.asarray(table.n), np.asarray(table.cs2)
    rho_c = np.interp(hc, hh, n)
    cs2_c = np.interp(hc, hh, cs2)
    nan = np.full_like(M, np.nan)                # M_b, I: solver TODO
    return np.stack([hc, M, R, Lam, nan, nan, rho_c, cs2_c, stable], axis=1)


def _stable_branches(M, R, Lam, stable):
    """Return contiguous stable, mass-rising branches in central-coordinate order.

    A disconnected (twin-star) family can become stable again after an unstable
    interval.  Selecting every row before the global stable maximum mixes that
    interval and the later branch into the primary ``M -> (R, Lambda)`` map.  A
    branch therefore ends whenever a row is invalid/unstable or mass ceases to
    rise.  Runs shorter than two samples cannot support interpolation and are
    omitted.
    """
    M = np.asarray(M)
    R = np.asarray(R)
    Lam = np.asarray(Lam)
    valid = (np.isfinite(M) & np.isfinite(R) & np.isfinite(Lam)
             & (M > 0) & (np.asarray(stable) > 0.5))
    branches = []
    start = None
    for k in range(len(M)):
        continues = (valid[k] and start is not None and k > 0
                     and valid[k - 1] and M[k] > M[k - 1])
        if valid[k] and start is None:
            start = k
        elif valid[k] and not continues:
            if k - start >= 2:
                branches.append((M[start:k], R[start:k], Lam[start:k]))
            start = k
        elif not valid[k] and start is not None:
            if k - start >= 2:
                branches.append((M[start:k], R[start:k], Lam[start:k]))
            start = None
    if start is not None and len(M) - start >= 2:
        branches.append((M[start:], R[start:], Lam[start:]))
    return branches


def _stable_rising(M, R, Lam, stable):
    """Return the primary stable branch used by the NMB v1 RIFT contract."""
    branches = _stable_branches(M, R, Lam, stable)
    if branches:
        return branches[0]
    empty = np.array([], dtype=float)
    return empty, empty.copy(), empty.copy()


# ---------------- representation-agnostic accessor protocol ----------------
class SequenceRepresentation(abc.ABC):
    """Accessor protocol any EOS-sequence representation must provide.

    A 'sequence' is an indexed family of EOS.  Consumers (RIFT inference) depend ONLY
    on these methods -- NOT on how the family is stored.  Today's representation is
    **tabular draws** stored vs central enthalpy (:class:`NSSequence`,
    ``representation='tabular_hc/1'``).  The interface is deliberately storage-agnostic
    so the internal representation can later be a **compressed / functional** form --
    a Gaussian process over the EOS function space, PCA / spectral / empirical
    interpolants, etc. -- modelling functional spaces rather than pure draws, without
    changing any consumer.  ``open_sequence`` dispatches on the ``representation`` tag.
    """
    n_eos: int = 0
    params = None

    @abc.abstractmethod
    def lambda_of_m(self, m, e): ...
    @abc.abstractmethod
    def radius_of_m(self, m, e): ...
    @abc.abstractmethod
    def m_max(self, e): ...

    def ordering_values(self, m_ref, kind="Lambda"):
        """Ordering statistic (Lambda or R at m_ref) for every EOS."""
        fn = self.lambda_of_m if kind.lower() == "lambda" else self.radius_of_m
        return np.array([fn(m_ref, e) for e in range(self.n_eos)])

    def n_stable_branches(self, e):
        """Number of resolved stable branches for EOS ``e``.

        Scalar v1 accessors intentionally continue to use the primary branch.
        This topology query lets consumers detect when that reduction discards
        additional stable branches.
        """
        return int(self.stable_branch_counts[e])


def open_sequence(fname):
    """Open an EOS sequence, dispatching on its stored ``representation`` tag.

    Returns a :class:`SequenceRepresentation`.  Future compressed representations
    (GP / interpolant) register here; today only 'tabular_hc' is implemented.
    """
    if not _HAVE_H5PY:
        raise ImportError("open_sequence requires h5py")
    with h5py.File(fname, "r") as f:
        rep = str(f.attrs.get("representation", "tabular_hc/1"))
    if rep.startswith("tabular"):
        return NSSequence(fname)
    if rep.startswith("pca"):
        from .compressed import PCASequence
        return PCASequence(fname)
    raise NotImplementedError(
        f"sequence representation {rep!r} not yet supported "
        "(reserved for future GP / interpolant representations)")


# ---------------- parallel sequence builds ----------------
def _one_eos_worker(args):
    """Subprocess worker: rebuild the EOSTable from numpy and run _one_eos."""
    arrays, n_points = args
    from ..eos.base import EOSTable
    t = EOSTable(*[jnp.asarray(a) for a in arrays])
    return _one_eos(t, n_points)


def _build_rows(tables, n_points, n_jobs=None, progress=False, desc="TOV sequences"):
    """Per-EOS sequence rows, parallel across EOS (process pool, spawn context).

    ``n_jobs=None`` -> max(5, cpu_count-1) workers (sequential for tiny batches,
    where spawn+jax-init overhead would dominate).  Workers are separate
    processes, so per-EOS numerics are identical to the sequential path
    (verified bitwise).  NOTE: the spawn context requires the CALLING script to
    use the standard ``if __name__ == "__main__":`` guard (all our CLIs do).
    """
    import os as _os
    if n_jobs is None:
        n_jobs = max(5, (_os.cpu_count() or 5) - 1)
    bar = None
    if progress:
        from ..priors.bridge import _pbar
        bar = _pbar(len(tables), desc)
    rows = [None] * len(tables)
    if n_jobs <= 1 or len(tables) < max(8, n_jobs):
        for k, t in enumerate(tables):
            rows[k] = _one_eos(t, n_points)
            if bar:
                bar.update(1)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")          # jax-safe (no fork-after-init)
        payload = [(tuple(np.asarray(x) for x in t), n_points) for t in tables]
        with ctx.Pool(min(n_jobs, len(tables))) as pool:
            for k, row in enumerate(pool.imap(_one_eos_worker, payload)):
                rows[k] = row
                if bar:
                    bar.update(1)
    if bar:
        bar.close()
    return rows


# ---------------- new self-describing format (tabular representation) ----------------
def write_nss(fname, tables, params=None, n_points=80, progress=False,
              n_jobs=None):
    """Write the robust ``NSSequence`` HDF5 (tabular_hc representation) from EOSTables.

    The per-EOS TOV family builds (the dominant cost) run across ``n_jobs``
    worker processes (default: max(5, cpu_count-1); sequential for small
    batches).  ``progress=True`` shows a per-EOS bar.
    """
    if not _HAVE_H5PY:
        raise ImportError("write_nss requires h5py")
    seq = np.stack(_build_rows(tables, n_points, n_jobs=n_jobs,
                               progress=progress), axis=0)
    with h5py.File(fname, "w") as f:
        f.attrs["schema_version"] = "nmbackend.nss/1"
        f.attrs["representation"] = "tabular_hc/1"      # dispatch tag (see open_sequence)
        f.attrs["generator"] = f"nmbackend {_NMB_VERSION}"
        f.attrs["created"] = datetime.datetime.utcnow().isoformat()
        f.attrs["fields"] = json.dumps(NSS_FIELDS)
        f.attrs["units"] = json.dumps(NSS_UNITS)
        if params is not None:
            f.attrs["eos_params"] = json.dumps(params)
        f.create_dataset("sequence", data=seq, compression="gzip")
    return fname


class NSSequence(SequenceRepresentation):
    """Tabular EOS sequence stored vs central enthalpy; branch-aware accessors."""

    def __init__(self, fname):
        if not _HAVE_H5PY:
            raise ImportError("NSSequence requires h5py")
        with h5py.File(fname, "r") as f:
            self.fields = json.loads(f.attrs["fields"])
            self.units = json.loads(f.attrs["units"])
            self.params = (json.loads(f.attrs["eos_params"])
                           if "eos_params" in f.attrs else None)
            self.seq = f["sequence"][:]                 # (n_eos, n_pts, n_fields)
        self._i = {k: j for j, k in enumerate(self.fields)}
        self.n_eos = self.seq.shape[0]
        self.stable_branch_counts = np.array([
            len(_stable_branches(
                s[:, self._i["M"]], s[:, self._i["R"]],
                s[:, self._i["Lambda"]], s[:, self._i["stable"]]
            ))
            for s in self.seq
        ], dtype=int)

    def _branch(self, e):
        s = self.seq[e]
        return _stable_rising(s[:, self._i["M"]], s[:, self._i["R"]],
                              s[:, self._i["Lambda"]], s[:, self._i["stable"]])

    def lambda_of_m(self, m, e):
        M, R, Lam = self._branch(e)
        if M.size < 2:
            return np.nan
        return float(np.exp(np.interp(m, M, np.log(Lam),
                                      left=np.nan, right=np.nan)))

    def radius_of_m(self, m, e):
        M, R, Lam = self._branch(e)
        if M.size < 2:
            return np.nan
        return float(np.interp(m, M, R, left=np.nan, right=np.nan))

    def m_max(self, e):
        M, _, _ = self._branch(e)
        return float(np.max(M)) if M.size else -np.inf

    def ordering_values(self, m_ref, kind="Lambda"):
        """Ordering statistic (Lambda or R at m_ref) for every EOS -- vectorized."""
        fn = self.lambda_of_m if kind.lower() == "lambda" else self.radius_of_m
        return np.array([fn(m_ref, e) for e in range(self.n_eos)])


# ---------------- legacy EOSSequenceLandry format (drop-in for RIFT) ----------------
def write_landry_legacy(fname, tables, eos_tables=None, n_points=120,
                        n_jobs=None, progress=False):
    """Write the exact ``EOSSequenceLandry`` HDF5 (id, ns/<name>{M,R,Lambda},
    eos/<name>{pressurec2,energy_densityc2,baryon_density} in g/cm^3).

    Only the **stable branch** (M rising to M_max) is stored, so RIFT's single-valued
    M->Lambda interpolation is correct for phase-transition EOS without code changes.
    ``eos_tables`` (list of the same EOSTables) populates the optional ``eos`` group.
    """
    if not _HAVE_H5PY:
        raise ImportError("write_landry_legacy requires h5py")
    with h5py.File(fname, "w") as f:
        f.create_dataset("id", data=np.arange(len(tables)))
        gns = f.create_group("ns")
        geos = f.create_group("eos") if eos_tables is not None else None
        rows = _build_rows(tables, n_points, n_jobs=n_jobs, progress=progress,
                           desc="TOV sequences (legacy)")
        for k, t in enumerate(tables):
            seq = rows[k]
            M, R, Lam = _stable_rising(seq[:, 1], seq[:, 2], seq[:, 3], seq[:, 8])
            ns = np.zeros(M.size, dtype=[("M", "f8"), ("R", "f8"), ("Lambda", "f8")])
            ns["M"], ns["R"], ns["Lambda"] = M, R, Lam
            gns.create_dataset(f"eos_{k}", data=ns)
            if geos is not None:
                te = eos_tables[k]
                n = np.asarray(te.n)
                p = np.asarray(te.p_mevfm3) * _MEVFM3_TO_GCM3
                e = np.asarray(te.eps_mevfm3) * _MEVFM3_TO_GCM3
                nb = n * _NB_TO_GCM3
                et = np.zeros(n.size, dtype=[("pressurec2", "f8"),
                                             ("energy_densityc2", "f8"),
                                             ("baryon_density", "f8")])
                et["pressurec2"], et["energy_densityc2"], et["baryon_density"] = p, e, nb
                geos.create_dataset(f"eos_{k}", data=et)
    return fname
