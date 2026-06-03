#!/usr/bin/env python3
"""Compress a tabular NSSequence file into the compact ``pca_hc/1`` representation.

Reads an existing ``*.nss.h5`` (no TOV re-run), fits a per-channel PCA basis to the
M(u), R(u), logLambda(u) curves, writes ``mean + components + coeffs``, and reports the
reconstruction error and compression ratio.  The compressed file is a drop-in
``SequenceRepresentation`` (RIFT opens it via ``EOSSequenceFromFile``; nmbackend via
``open_sequence``).

Example
-------
PYTHONPATH=src python scripts/compress_sequence.py seq.nss.h5 \
    --out seq.pca.h5 --n-components 8 --m-ref 1.4
"""
import argparse

import numpy as np

from nmbackend.rift import (NSSequence, PCASequence, write_pca_from_nss,
                            compression_ratio)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nss", help="input tabular NSSequence HDF5")
    ap.add_argument("--out", required=True, help="output compressed pca_hc HDF5")
    ap.add_argument("--n-components", type=int, default=8)
    ap.add_argument("--m-ref", type=float, default=1.4,
                    help="reference mass for the reconstruction-error report")
    a = ap.parse_args()

    write_pca_from_nss(a.nss, a.out, n_components=a.n_components)

    ref, comp = NSSequence(a.nss), PCASequence(a.out)
    n = ref.n_eos
    lr = np.array([ref.lambda_of_m(a.m_ref, e) for e in range(n)])
    lc = np.array([comp.lambda_of_m(a.m_ref, e) for e in range(n)])
    ok = np.isfinite(lr) & np.isfinite(lc)
    rel = np.abs(lc[ok] - lr[ok]) / lr[ok]
    npts = comp.components.shape[2]
    print(f"compressed {n} EOS -> {a.out}")
    print(f"  components={a.n_components}  n_points={npts}  "
          f"compression~{compression_ratio(n, npts, a.n_components):.1f}x")
    print(f"  Lambda({a.m_ref}) reconstruction: median {np.median(rel)*100:.2f}%  "
          f"max {np.max(rel)*100:.2f}%")


if __name__ == "__main__":
    main()
