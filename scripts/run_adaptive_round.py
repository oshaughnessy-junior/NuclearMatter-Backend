#!/usr/bin/env python3
"""Run ONE round of the adaptive sequence-space prior against RIFT CIP evidence.

State-persisted driver: each invocation loads the prior (or initializes it from a
training PCA file), draws, runs one CIP job (CIPTabularEvidence), refines the
moments (stage 1; ``--adapt-basis`` switches this round to stage-2 basis
adaptation), and saves the new state -- so long campaigns are driven round-by-round
(cluster jobs, makefiles, or repeated CLI calls).

Example (round 0 initializes from training PCA):
  PYTHONPATH=$RIFT_CODE:src python scripts/run_adaptive_round.py \
      --state prior_state.npz --train-pca train.pca.h5 \
      --cip-exe $RIFT_CODE/bin/util_ConstructIntrinsicPosterior_GenericCoordinates.py \
      --net all_tabular.net --workdir adapt_work --n-per-round 12
"""
import argparse
import json
import os

import numpy as np

from nmbackend.priors.gp_pca import (CoefficientPrior, save_prior, load_prior,
                                     sequence_lambda_of_m, adapt_basis)
from nmbackend.rift.evidence import CIPTabularEvidence


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", required=True, help="prior state npz (in/out)")
    ap.add_argument("--train-pca", default=None,
                    help="training pca_hc file (round-0 initialization)")
    ap.add_argument("--cip-exe", required=True)
    ap.add_argument("--net", required=True, help="all.net-style composite")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--rift-code", default=None)
    ap.add_argument("--n-per-round", type=int, default=12)
    ap.add_argument("--n-eff", type=int, default=10)
    ap.add_argument("--n-max", type=int, default=150000)
    ap.add_argument("--n-output-samples", type=int, default=4000)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--shrink", type=float, default=0.2)
    ap.add_argument("--adapt-basis", action="store_true",
                    help="stage-2: adapt the basis this round instead of moments")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()

    if os.path.exists(a.state):
        prior = load_prior(a.state)
    else:
        if not a.train_pca:
            raise SystemExit("first round needs --train-pca")
        prior = CoefficientPrior.from_pca_file(a.train_pca)
    rng = np.random.default_rng(a.seed if a.seed is not None
                                else 1000 + prior.generation)

    ev = CIPTabularEvidence(a.cip_exe, a.net, a.workdir, rift_code=a.rift_code,
                            n_max=a.n_max, n_eff=a.n_eff,
                            n_output_samples=a.n_output_samples, verbose=True)
    coeffs, keep = prior.sample_sequences(a.n_per_round, rng)
    lnZ = ev(prior, coeffs)
    ess = prior.ess(lnZ, beta=a.beta)

    lam = np.array([sequence_lambda_of_m(prior, c) for c in coeffs])
    okL = np.isfinite(lam)
    w = np.exp(a.beta * (lnZ - np.max(lnZ))); w /= w.sum()
    lam_w = float(np.sum(w[okL] * lam[okL]) / np.sum(w[okL])) if okL.any() else float("nan")

    if a.adapt_basis:
        new_prior, diag = adapt_basis(prior, coeffs, lnZ, beta=a.beta)
        extra = dict(stage="basis", captured=diag["captured"].tolist(),
                     mean_shift=diag["mean_shift"].tolist())
    else:
        new_prior = prior.refine(coeffs, lnZ, beta=a.beta, shrink=a.shrink)
        extra = dict(stage="moments")
    save_prior(a.state, new_prior)

    rec = dict(generation=prior.generation, n=a.n_per_round, ess=float(ess),
               keep_frac=float(keep),
               lam14_draws_mean=float(np.nanmean(lam)),
               lam14_draws_std=float(np.nanstd(lam)),
               lam14_posterior_weighted=lam_w, **extra)
    hist = os.path.join(a.workdir, "history.jsonl")
    os.makedirs(a.workdir, exist_ok=True)
    with open(hist, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
