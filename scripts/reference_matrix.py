#!/usr/bin/env python3
"""Phase-1 reference-matrix harness CLI.

Run several TOV solvers on an adversarial EOS ladder and tabulate R/Lambda
differences at fixed gravitational mass.  RePrimAnd / lalsim / MUSES-qlimr columns
appear automatically when those backends are installed (see docs/environment.md);
in their absence the comparison is our diffrax solver vs the independent SciPy
(H,beta) reference, which establishes the internal numerical floor (~1-2 m in R).

    PYTHONPATH=src python scripts/reference_matrix.py
    # reproduces the committed refmatrix.csv (RePrimAnd not installed, so the
    # reference column is the SciPy (H,beta) solver; default masses 1.2..2.0):
    PYTHONPATH=src python scripts/reference_matrix.py --masses 1.2 1.4 1.6 1.8 2.0 \
        --n-models 80 --csv refmatrix.csv
"""
import argparse

from nmbackend.validation.reference_matrix import run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--masses", type=float, nargs="+",
                    default=[1.2, 1.4, 1.6, 1.8, 2.0],
                    help="target gravitational masses [Msun]")
    ap.add_argument("--n-models", type=int, default=60,
                    help="sequence resolution per EOS")
    ap.add_argument("--reference", default=None,
                    choices=[None, "reprimand", "scipy_hbeta", "diffrax"],
                    help="force the reference solver (default: best available)")
    ap.add_argument("--csv", default=None, help="write long-format results to CSV")
    a = ap.parse_args()
    run(target_masses=tuple(a.masses), n_models=a.n_models,
        reference=a.reference, csv_path=a.csv)


if __name__ == "__main__":
    main()
