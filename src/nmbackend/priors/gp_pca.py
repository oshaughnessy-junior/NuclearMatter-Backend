"""Adaptive Gaussian prior over the compressed (PCA) EOS-sequence space.

This is the substrate for **adaptive nonparametric EOS priors** (RIFT-GP-style
iterative refinement): the compressed ``pca_hc/1`` representation expresses every
EOS sequence as coefficients c in R^{3k} on a fixed functional basis
(mean + components for the M(u), R(u), log Lambda(u) channels).  A Gaussian
distribution over c IS a Gaussian process over the sequence function space,
restricted to (and with covariance expressed in) the PCA basis -- so fitting,
sampling, and *re-fitting* the coefficient distribution gives a fully functional
GP prior whose draws are new, smooth, never-seen sequences (not replays).

The adaptive loop (one round):

    1. ``sample``  : draw n coefficient vectors  -> n new EOS sequences;
    2. evaluate    : per-EOS (marginal) evidence lnZ_e for each draw -- in
       production this is the RIFT tabular per-EOS evidence (CIP); in tests a
       synthetic likelihood;
    3. ``refine``  : importance-reweight the draws (w ~ exp(beta lnZ)) and re-fit
       the Gaussian (weighted mean + covariance), optionally tempered (beta<1)
       and shrunk toward the previous round for stability.

Iterating concentrates the prior in the high-evidence region of sequence space --
the prior *adapts to the inference results*, which (per the 2026-06 competition
scoping) is unpublished in the NS-EOS literature.  ``iterative_refinement`` drives
the loop against an abstract ``evidence_fn`` so the same code runs synthetic
demonstrations and the real RIFT pipeline.

Plain numpy; draws feed the standard sequence machinery (PCASequence files that
RIFT opens via EOSSequenceFromFile).
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

from ..rift.compressed import PCASequence, PCA_FIELDS
from .. import __version__ as _NMB_VERSION


class CoefficientPrior:
    """Gaussian prior over flattened PCA coefficients (3*k dims).

    Equivalent to a GP over the sequence function space in the PCA basis.  Carries
    the basis (mean curves + components) so draws reconstruct directly.
    """

    def __init__(self, mean_coeff, cov_coeff, basis_mean, basis_components,
                 generation=0):
        self.mean_coeff = np.asarray(mean_coeff, float)          # (3k,)
        self.cov_coeff = np.asarray(cov_coeff, float)            # (3k, 3k)
        self.basis_mean = np.asarray(basis_mean, float)          # (3, n_pts)
        self.basis_components = np.asarray(basis_components, float)  # (3,k,n_pts)
        self.generation = int(generation)
        self._n_ch, self._k, self._n_pts = self.basis_components.shape

    # ---------------------------------------------------------------- fitting --
    @classmethod
    def fit(cls, coeffs, basis_mean, basis_components, weights=None,
            jitter=1e-10, generation=0):
        """Fit (weighted) Gaussian to coefficients ``coeffs`` (n_eos, 3, k)."""
        X = np.asarray(coeffs, float).reshape(len(coeffs), -1)   # (n, 3k)
        if weights is None:
            w = np.ones(X.shape[0])
        else:
            w = np.asarray(weights, float)
            w = np.clip(w, 0.0, None)
        w = w / w.sum()
        mu = w @ X
        D = X - mu
        cov = (D * w[:, None]).T @ D
        # effective sample size guard: inflate toward isotropic if ESS tiny
        cov = cov + jitter * np.eye(cov.shape[0]) * max(np.trace(cov), 1.0)
        return cls(mu, cov, basis_mean, basis_components, generation=generation)

    @classmethod
    def from_pca_file(cls, fname, weights=None):
        """Fit the prior from an existing ``pca_hc/1`` file's stored coefficients."""
        seq = PCASequence(fname)
        return cls.fit(seq.coeffs, seq.mean, seq.components, weights=weights)

    # --------------------------------------------------------------- sampling --
    def sample_coeffs(self, n, rng):
        L = np.linalg.cholesky(
            self.cov_coeff + 1e-12 * np.eye(self.cov_coeff.shape[0]))
        z = rng.standard_normal((n, self.cov_coeff.shape[0]))
        return (self.mean_coeff + z @ L.T).reshape(n, self._n_ch, self._k)

    def reconstruct_raw(self, coeff):
        """Raw channel curves (3, n_pts) = (M, R, logLambda) for one coeff (3, k)."""
        return self.basis_mean + np.einsum("ck,ckp->cp",
                                           np.asarray(coeff),
                                           self.basis_components)

    def reconstruct(self, coeff):
        """(M, R, Lambda) curves for one coefficient vector (3, k)."""
        rec = self.reconstruct_raw(coeff)
        iM, iR, iL = (PCA_FIELDS.index("M"), PCA_FIELDS.index("R"),
                      PCA_FIELDS.index("logLambda"))
        return rec[iM], rec[iR], np.exp(rec[iL])

    def sample_sequences(self, n, rng, max_tries=10):
        """Draw n *valid* new sequences; returns (coeffs (n,3,k), kept fraction).

        Validity: finite curves, positive M and R, a non-trivial rising branch
        (M_max above the low end), Lambda positive (guaranteed by the log channel).
        """
        out, tried, kept = [], 0, 0
        while len(out) < n and tried < max_tries * n + 10:
            need = n - len(out)
            cand = self.sample_coeffs(max(need, 8), rng)
            tried += cand.shape[0]
            for c in cand:
                M, R, Lam = self.reconstruct(c)
                if (np.all(np.isfinite(M)) and np.all(np.isfinite(R))
                        and np.all(M > 0) and np.all(R > 0)
                        and np.max(M) > M[0] + 0.05):
                    out.append(c)
                    kept += 1
                    if len(out) >= n:
                        break
        if len(out) < n:
            raise RuntimeError("could not draw enough valid sequences "
                               f"({len(out)}/{n}); prior too wide?")
        return np.array(out), kept / max(tried, 1)

    # -------------------------------------------------------------- adaptation --
    def refine(self, coeffs, lnZ, beta=1.0, shrink=0.0):
        """One adaptive-refinement step: importance-reweight and re-fit.

        ``coeffs`` (n,3,k) are draws from THIS prior; ``lnZ`` their per-EOS
        (marginal) evidences.  Weights w ~ exp(beta (lnZ - max)); ``beta`` < 1
        tempers the update, ``shrink`` in [0,1) blends the new moments toward the
        current ones (regularization for small effective sample sizes).  Returns a
        new :class:`CoefficientPrior` (generation+1).
        """
        lnZ = np.asarray(lnZ, float)
        ok = np.isfinite(lnZ)
        w = np.zeros_like(lnZ)
        w[ok] = np.exp(beta * (lnZ[ok] - np.max(lnZ[ok])))
        new = CoefficientPrior.fit(coeffs, self.basis_mean, self.basis_components,
                                   weights=w, generation=self.generation + 1)
        if shrink > 0.0:
            new.mean_coeff = (1 - shrink) * new.mean_coeff + shrink * self.mean_coeff
            new.cov_coeff = (1 - shrink) * new.cov_coeff + shrink * self.cov_coeff
        return new

    def ess(self, lnZ, beta=1.0):
        """Effective sample size of the refinement weights (diagnostic)."""
        lnZ = np.asarray(lnZ, float)
        ok = np.isfinite(lnZ)
        w = np.exp(beta * (lnZ[ok] - np.max(lnZ[ok])))
        w = w / w.sum()
        return 1.0 / np.sum(w ** 2)


def sequence_lambda_of_m(prior, coeff, m=1.4):
    """Branch-aware Lambda(m) of one reconstructed draw (NaN if m off-branch)."""
    from ..rift.sequence import _stable_rising
    M, R, Lam = prior.reconstruct(coeff)
    stable = np.concatenate([[True], np.diff(M) > 0]).astype(float)
    Mb, Rb, Lb = _stable_rising(M, R, Lam, stable)
    if Mb.size < 2 or not (Mb[0] <= m <= Mb[-1]):
        return np.nan
    return float(np.exp(np.interp(m, Mb, np.log(np.maximum(Lb, 1e-30)))))


def save_prior(fname, prior):
    """Persist a CoefficientPrior to npz (for round-by-round driving)."""
    np.savez(fname, mean_coeff=prior.mean_coeff, cov_coeff=prior.cov_coeff,
             basis_mean=prior.basis_mean, basis_components=prior.basis_components,
             generation=prior.generation)
    return fname


def load_prior(fname):
    d = np.load(fname)
    return CoefficientPrior(d["mean_coeff"], d["cov_coeff"], d["basis_mean"],
                            d["basis_components"], generation=int(d["generation"]))


def adapt_basis(prior, coeffs, lnZ, n_components=None, beta=1.0):
    """Re-fit the PCA **basis itself** from evidence-weighted draws.

    This is the genuine kernel adaptation of the functional GP: the covariance
    operator (mean curves + principal components + spectrum) is re-estimated from
    the weighted ensemble of *curves*, not merely the coefficient moments on a
    frozen basis.  Where the posterior concentrates, the adapted basis re-allocates
    its resolution.

    Parameters: ``coeffs`` (n, 3, k) draws from ``prior``; ``lnZ`` their evidences
    (weights ~ exp(beta lnZ)); ``n_components`` defaults to the current k.

    Returns ``(new_prior, diag)`` -- the new :class:`CoefficientPrior` on the
    adapted basis (coefficient Gaussian re-fit on it) and diagnostics:

    * ``captured``      : per-channel weighted variance fraction captured by the
      new basis on the input draws (re-expansion residual = 1 - captured);
    * ``principal_angles`` : per-channel principal angles (rad) between old and
      new component subspaces (0 = unrotated);
    * ``mean_shift``    : per-channel rms shift of the mean curve.
    """
    coeffs = np.asarray(coeffs, float)
    lnZ = np.asarray(lnZ, float)
    ok = np.isfinite(lnZ)
    w = np.zeros_like(lnZ)
    w[ok] = np.exp(beta * (lnZ[ok] - np.max(lnZ[ok])))
    if w.sum() <= 0:
        raise ValueError("all refinement weights vanished")
    w = w / w.sum()
    k_new = int(n_components or prior.basis_components.shape[1])

    # weighted ensemble of raw curves (n, 3, n_pts)
    curves = np.stack([prior.reconstruct_raw(c) for c in coeffs], axis=0)
    n_ch, n_pts = curves.shape[1], curves.shape[2]
    k_new = min(k_new, curves.shape[0] - 1, n_pts)

    new_mean = np.einsum("n,ncp->cp", w, curves)
    comps = np.zeros((n_ch, k_new, n_pts))
    new_coeffs = np.zeros((curves.shape[0], n_ch, k_new))
    captured = np.zeros(n_ch)
    angles = []
    for c in range(n_ch):
        D = curves[:, c, :] - new_mean[c]
        Xw = D * np.sqrt(w)[:, None]
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
        comps[c] = Vt[:k_new]
        new_coeffs[:, c, :] = D @ comps[c].T
        tot = float(np.sum(S ** 2))
        captured[c] = float(np.sum(S[:k_new] ** 2) / tot) if tot > 0 else 1.0
        # principal angles between old and new subspaces (orthonormal rows)
        M = prior.basis_components[c] @ comps[c].T
        sv = np.clip(np.linalg.svd(M, compute_uv=False), -1.0, 1.0)
        angles.append(np.arccos(sv))
    new_prior = CoefficientPrior.fit(new_coeffs, new_mean, comps, weights=w,
                                     generation=prior.generation + 1)
    diag = dict(captured=captured,
                principal_angles=np.array(angles),
                mean_shift=np.sqrt(np.mean((new_mean - prior.basis_mean) ** 2,
                                           axis=1)))
    return new_prior, diag


def write_pca_draws(fname, prior, coeffs, params=None):
    """Write new coefficient draws as a standard ``pca_hc/1`` file.

    The file carries the SAME basis as the parent representation, with the new
    per-EOS coefficients -- it opens through ``open_sequence`` /
    RIFT ``EOSSequenceFromFile`` like any other compressed sequence.
    """
    if not _HAVE_H5PY:
        raise ImportError("write_pca_draws requires h5py")
    coeffs = np.asarray(coeffs, float)
    with h5py.File(fname, "w") as f:
        f.attrs["schema_version"] = "nmbackend.pca/1"
        f.attrs["representation"] = "pca_hc/1"
        f.attrs["generator"] = (f"nmbackend {_NMB_VERSION} "
                                f"(adaptive draw, generation {prior.generation})")
        f.attrs["created"] = datetime.datetime.utcnow().isoformat()
        f.attrs["channels"] = json.dumps(PCA_FIELDS)
        f.attrs["n_points"] = int(prior.basis_mean.shape[1])
        f.attrs["n_components"] = int(prior.basis_components.shape[1])
        f.attrs["adaptive_generation"] = int(prior.generation)
        if params is not None:
            f.attrs["eos_params"] = json.dumps(params)
        f.create_dataset("mean", data=prior.basis_mean, compression="gzip")
        f.create_dataset("components", data=prior.basis_components,
                         compression="gzip")
        f.create_dataset("coeffs", data=coeffs, compression="gzip")
    return fname


def iterative_refinement(prior, evidence_fn, rng, n_rounds=3, n_per_round=64,
                         beta=1.0, shrink=0.2, adapt_basis_every=None,
                         verbose=False):
    """RIFT-GP-style adaptive loop over the sequence-space prior.  Two stages:

    * **Stage 1 (default): moment refit.**  Each round importance-reweights the
      draws by their evidence and re-fits the coefficient Gaussian
      (:meth:`CoefficientPrior.refine`).  Cheap, stable, and sufficient when the
      posterior lives inside the initial basis span -- the easy win.
    * **Stage 2 (opt-in stretch): basis adaptation.**  With
      ``adapt_basis_every=k``, every k-th round additionally re-fits the PCA basis
      itself from the weighted curves (:func:`adapt_basis`) -- re-centering the
      mean and re-orienting the components to the posterior-concentrated
      directions.  NOTE: draws are in the current basis span by construction, so
      adaptation re-allocates resolution *within* the span (rotation, mean shift,
      effective truncation); expanding the span requires injecting fresh external
      training curves (e.g. new bridge draws near the posterior).

    ``evidence_fn(prior, coeffs) -> lnZ array`` abstracts the per-EOS evidence:
    in production it writes the draws via :func:`write_pca_draws`, runs the RIFT
    tabular per-EOS evidence (CIP) on the file, and returns the per-EOS lnZ; in
    tests it is a synthetic likelihood over (M, R, Lambda) curves.

    Returns (final_prior, history); history records per round the draws,
    evidences, weight-ESS, and (when stage 2 fires) the basis diagnostics.
    """
    history = []
    for r in range(n_rounds):
        coeffs, keep_frac = prior.sample_sequences(n_per_round, rng)
        lnZ = np.asarray(evidence_fn(prior, coeffs), float)
        ess = prior.ess(lnZ, beta=beta)
        rec = dict(round=r, coeffs=coeffs, lnZ=lnZ, ess=float(ess),
                   keep_frac=float(keep_frac), generation=prior.generation)
        if verbose:
            print(f"round {r}: gen={prior.generation} ESS={ess:.1f}/"
                  f"{n_per_round} keep={keep_frac:.2f} "
                  f"lnZ max={np.nanmax(lnZ):.2f}")
        if adapt_basis_every and (r + 1) % adapt_basis_every == 0:
            prior, diag = adapt_basis(prior, coeffs, lnZ, beta=beta)
            rec["basis_diag"] = diag
            if verbose:
                print(f"  basis adapted: captured={diag['captured']} "
                      f"mean_shift={diag['mean_shift']}")
        else:
            prior = prior.refine(coeffs, lnZ, beta=beta, shrink=shrink)
        history.append(rec)
    return prior, history
