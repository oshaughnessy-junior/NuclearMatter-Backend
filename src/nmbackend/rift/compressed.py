"""Compressed / functional EOS-sequence representations (PCA over M-R-Lambda curves).

This is the first *non-tabular* :class:`SequenceRepresentation` -- it delivers the
design mandate to "model functional spaces rather than pure draws".  Instead of
storing every EOS sequence as raw samples, we treat each EOS's stellar sequence as a
set of functions of a normalized central coordinate u in [0,1]:

    M(u),  R(u),  log Lambda(u)

stack them across the n_eos draws, and take a per-channel principal-component (SVD)
decomposition::

    M(u)        ~ mean_M(u)   + sum_k  a^M_k   * phi^M_k(u)
    R(u)        ~ mean_R(u)   + sum_k  a^R_k   * phi^R_k(u)
    log Lambda  ~ mean_L(u)   + sum_k  a^L_k   * phi^L_k(u)

On disk we keep only ``mean`` (3, n_pts), the basis ``components`` (3, n_comp, n_pts),
and the per-EOS coefficients ``coeffs`` (n_eos, 3, n_comp) -- a large compression once
n_eos >> n_comp.  Crucially this is a *function-space* model: ANY coefficient vector
reconstructs a smooth, valid-looking M-R-Lambda sequence, so the same machinery later
supports interpolating/sampling new EOS between the training draws (empirical
interpolants, GP priors over the coefficients, ...), not just replaying the draws.

The consumer API (``lambda_of_m``, ``radius_of_m``, ``m_max``, ``ordering_values``)
is identical to the tabular :class:`NSSequence`; ``open_sequence`` dispatches on the
``representation="pca_hc/1"`` tag, so RIFT (and any other consumer) is unchanged.
"""
from __future__ import annotations

import json
import datetime
import numpy as np

try:
    import h5py
    _HAVE_H5PY = True
except Exception:  # pragma: no cover
    _HAVE_H5PY = False

from .sequence import (SequenceRepresentation, _one_eos, _stable_rising,
                       NSS_FIELDS, _MEVFM3_TO_GCM3, _NB_TO_GCM3)
from .. import __version__ as _NMB_VERSION

PCA_FIELDS = ["M", "R", "logLambda"]   # channels decomposed (u-indexed)


def _curves_from_tables(tables, n_points):
    """Per-EOS (M, R, logLambda) curves on the common u-index grid -> (n_eos,3,n_pts)."""
    iM, iR, iL = NSS_FIELDS.index("M"), NSS_FIELDS.index("R"), NSS_FIELDS.index("Lambda")
    rows = []
    for t in tables:
        s = _one_eos(t, n_points)
        M = s[:, iM]
        R = s[:, iR]
        Lam = np.maximum(s[:, iL], 1e-30)
        rows.append(np.stack([M, R, np.log(Lam)], axis=0))   # (3, n_pts)
    return np.stack(rows, axis=0)                            # (n_eos, 3, n_pts)


def _curves_from_sequence(seq):
    """Per-EOS (M, R, logLambda) curves from a stored NSS ``sequence`` array.

    ``seq`` is the ``(n_eos, n_pts, n_fields)`` dataset; no TOV re-run.
    """
    iM, iR, iL = NSS_FIELDS.index("M"), NSS_FIELDS.index("R"), NSS_FIELDS.index("Lambda")
    M = seq[:, :, iM]
    R = seq[:, :, iR]
    logL = np.log(np.maximum(seq[:, :, iL], 1e-30))
    return np.stack([M, R, logL], axis=1)                    # (n_eos, 3, n_pts)


def _fit_pca_curves(curves, n_components):
    """Per-channel SVD of (n_eos, 3, n_pts) curves -> mean/components/coeffs dict."""
    n_eos, n_ch, n_pts = curves.shape
    k = int(min(n_components, n_eos, n_pts))
    mean = curves.mean(axis=0)                               # (3, n_pts)
    comps = np.zeros((n_ch, k, n_pts))
    coeffs = np.zeros((n_eos, n_ch, k))
    for c in range(n_ch):
        X = curves[:, c, :] - mean[c]                       # (n_eos, n_pts)
        # economy SVD: X = U S Vt; principal directions are rows of Vt
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        comps[c] = Vt[:k]                                    # (k, n_pts)
        coeffs[:, c, :] = (U[:, :k] * S[:k])                # (n_eos, k)
    return dict(mean=mean, components=comps, coeffs=coeffs, n_points=n_pts)


def fit_sequence_pca(tables, n_points=80, n_components=6):
    """Fit a per-channel PCA basis to a list of EOSTables (runs TOV via _one_eos).

    Returns dict with ``mean`` (3,n_pts), ``components`` (3,n_comp,n_pts),
    ``coeffs`` (n_eos,3,n_comp).  ``n_components`` is clipped to min(n_eos, n_pts).
    Prefer :func:`write_pca_from_nss` to compress an *existing* sequence file without
    recomputing the stellar models.
    """
    return _fit_pca_curves(_curves_from_tables(tables, n_points), n_components)


def _write_pca_file(fname, fit, params, n_points):
    with h5py.File(fname, "w") as f:
        f.attrs["schema_version"] = "nmbackend.pca/1"
        f.attrs["representation"] = "pca_hc/1"
        f.attrs["generator"] = f"nmbackend {_NMB_VERSION}"
        f.attrs["created"] = datetime.datetime.utcnow().isoformat()
        f.attrs["channels"] = json.dumps(PCA_FIELDS)
        f.attrs["n_points"] = int(n_points)
        f.attrs["n_components"] = int(fit["components"].shape[1])
        if params is not None:
            f.attrs["eos_params"] = json.dumps(params)
        f.create_dataset("mean", data=fit["mean"], compression="gzip")
        f.create_dataset("components", data=fit["components"], compression="gzip")
        f.create_dataset("coeffs", data=fit["coeffs"], compression="gzip")
    return fname


def write_pca_from_nss(nss_fname, pca_fname, n_components=6):
    """Compress an existing tabular ``NSSequence`` file into a ``pca_hc/1`` file.

    Reads the stored sequence arrays (no TOV re-run), fits the PCA basis, and writes
    the compact representation.  This is the natural compression workflow.
    """
    if not _HAVE_H5PY:
        raise ImportError("write_pca_from_nss requires h5py")
    with h5py.File(nss_fname, "r") as f:
        seq = f["sequence"][:]
        params = (json.loads(f.attrs["eos_params"])
                  if "eos_params" in f.attrs else None)
    fit = _fit_pca_curves(_curves_from_sequence(seq), n_components)
    return _write_pca_file(pca_fname, fit, params, seq.shape[1])


def write_pca(fname, tables, params=None, n_points=80, n_components=6):
    """Write a compressed ``pca_hc/1`` file from a list of EOSTables (runs TOV)."""
    if not _HAVE_H5PY:
        raise ImportError("write_pca requires h5py")
    fit = fit_sequence_pca(tables, n_points=n_points, n_components=n_components)
    return _write_pca_file(fname, fit, params, n_points)


def compression_ratio(n_eos, n_points, n_components, n_channels=3):
    """Float-count compression vs storing the raw (n_eos, n_pts, n_channels) curves."""
    raw = n_eos * n_points * n_channels
    comp = (n_channels * n_points                       # mean
            + n_channels * n_components * n_points       # basis
            + n_eos * n_channels * n_components)         # coeffs
    return raw / comp


class PCASequence(SequenceRepresentation):
    """Reader for the compressed ``pca_hc/1`` representation; reconstructs on demand."""

    def __init__(self, fname):
        if not _HAVE_H5PY:
            raise ImportError("PCASequence requires h5py")
        with h5py.File(fname, "r") as f:
            self.channels = json.loads(f.attrs["channels"])
            self.params = (json.loads(f.attrs["eos_params"])
                           if "eos_params" in f.attrs else None)
            self.mean = f["mean"][:]                         # (3, n_pts)
            self.components = f["components"][:]             # (3, n_comp, n_pts)
            self.coeffs = f["coeffs"][:]                     # (n_eos, 3, n_comp)
        self.n_eos = self.coeffs.shape[0]
        self._iM, self._iR, self._iL = (self.channels.index("M"),
                                        self.channels.index("R"),
                                        self.channels.index("logLambda"))

    def reconstruct(self, e):
        """Reconstruct (M, R, Lambda) curves on the u grid for EOS index ``e``."""
        # curve[c] = mean[c] + coeffs[e,c,:] @ components[c]
        rec = self.mean + np.einsum("ck,ckp->cp", self.coeffs[e], self.components)
        M, R, logL = rec[self._iM], rec[self._iR], rec[self._iL]
        return M, R, np.exp(logL)

    def _branch(self, e):
        M, R, Lam = self.reconstruct(e)
        # rising branch: stable where dM/du > 0 (u monotone in central enthalpy)
        stable = np.concatenate([[True], np.diff(M) > 0]).astype(float)
        return _stable_rising(M, R, Lam, stable)

    def lambda_of_m(self, m, e):
        M, R, Lam = self._branch(e)
        if M.size < 2:
            return np.nan
        return float(np.exp(np.interp(m, M, np.log(np.maximum(Lam, 1e-30)),
                                      left=np.nan, right=np.nan)))

    def radius_of_m(self, m, e):
        M, R, Lam = self._branch(e)
        if M.size < 2:
            return np.nan
        return float(np.interp(m, M, R, left=np.nan, right=np.nan))

    def m_max(self, e):
        M, _, _ = self.reconstruct(e)
        stable = np.concatenate([[True], np.diff(M) > 0])
        return float(np.max(np.where(stable, M, -np.inf)))
