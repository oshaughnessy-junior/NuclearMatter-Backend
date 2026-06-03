#!/usr/bin/env python3
"""Draw an EOS-prior ensemble from the constrained Brownian bridge (GKKS 2025).

Each draw is stable, causal, and thermodynamically consistent by construction,
anchored at low density to OUR metamodel band (NEPs sampled per draw) and at high
density to the partial-N3LO pQCD point at mu = 2.6 GeV (X log-uniform in [1,4]).
Writes the NSSequence file (and optionally the legacy Landry + compressed PCA
files) ready for the RIFT tabular pipeline.

Example
-------
PYTHONPATH=src python scripts/draw_bridge_prior.py --n 100 --seed 1 \
    --out bridge_prior.nss.h5 --legacy bridge_prior.landry.h5 \
    --pca bridge_prior.pca.h5
"""
import argparse
import json

import numpy as np

from nmbackend.priors import draw_prior_ensemble
from nmbackend.rift import write_nss, write_landry_legacy, write_pca_from_nss


def default_nep_sampler(rng):
    """Modest NEP prior for the low-density anchor band."""
    return {"L_sym": float(rng.uniform(30.0, 90.0)),
            "K_sym": float(rng.uniform(-240.0, 40.0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="number of EOS draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--corr", default="0.2,0.4",
                    help="hierarchical sigma/n range, e.g. '0.2,0.4'")
    ap.add_argument("--levels", type=int, default=9, help="fractal depth")
    ap.add_argument("--out", required=True, help="output NSSequence HDF5")
    ap.add_argument("--legacy", default=None, help="also write legacy Landry HDF5")
    ap.add_argument("--pca", default=None, help="also write compressed PCA HDF5")
    ap.add_argument("--pca-components", type=int, default=8)
    ap.add_argument("--n-points", type=int, default=80, help="sequence points/EOS")
    a = ap.parse_args()

    lo, hi = (float(x) for x in a.corr.split(","))
    tables, records = draw_prior_ensemble(a.n, seed=a.seed, corr_range=(lo, hi),
                                          nep_sampler=default_nep_sampler,
                                          levels=a.levels)
    mis = np.array([r["p_end_mismatch"] for r in records])
    print(f"drew {len(tables)} EOS  (p_H mismatch: median "
          f"{np.median(np.abs(mis))*100:.1f}%, max {np.max(np.abs(mis))*100:.1f}%)")

    write_nss(a.out, tables, params=records, n_points=a.n_points)
    print(f"wrote NSSequence -> {a.out}")
    if a.legacy:
        write_landry_legacy(a.legacy, tables, eos_tables=tables,
                            n_points=max(a.n_points, 120))
        print(f"wrote legacy EOSSequenceLandry -> {a.legacy}")
    if a.pca:
        write_pca_from_nss(a.out, a.pca, n_components=a.pca_components)
        print(f"wrote compressed PCA -> {a.pca}")


if __name__ == "__main__":
    main()
