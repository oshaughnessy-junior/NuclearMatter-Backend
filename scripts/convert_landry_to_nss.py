#!/usr/bin/env python3
"""Convert an EOSSequenceLandry HDF5 (e.g. the published LCEHL prior) to NSSequence.

Purpose: recover the published LCEHL prior inside this backend's pipeline for paper
comparisons -- the converted file drops into everything that consumes an NSSequence
(open_sequence, compress_sequence.py, the RIFT factory, ...).

Two modes:

* ``--repackage`` (default): wrap the *published* TOV results (M, R, Lambda per EOS)
  into the NSS layout, resampling each EOS onto a fixed number of points by row
  index.  The h_c column is a normalized index (the published files do not store
  central enthalpy) and rho_c / cs2_c are NaN; the branch-aware accessors only need
  M, R, Lambda, stable, so all sequence functionality works.

* ``--resolve``: ingest the microphysical ``eos`` tables and re-run OUR TOV solver,
  producing a fully native NSS file (h_c, rho_c, cs2_c populated).  Apples-to-apples
  with nmbackend-generated sequences but costs one family build per EOS -- use
  ``--max-eos`` to subset.

Example
-------
PYTHONPATH=src python scripts/convert_landry_to_nss.py \
    LCEHL_EOS_posterior_samples_PSR.h5 --out lcehl_psr.nss.h5 --max-eos 1000
"""
import argparse
import datetime
import json

import numpy as np
import h5py

from nmbackend import __version__ as _V
from nmbackend.rift.sequence import NSS_FIELDS, NSS_UNITS

# g/cm^3 -> our units (see rift.sequence emitter constants)
_GCM3_TO_MEVFM3 = 1.0 / 1.7827e12          # pressurec2/energy_densityc2 -> MeV/fm^3
_GCM3_TO_NB = 1.0 / (1e39 * 1.66e-24)      # baryon mass density -> fm^-3


def _natkey(s):
    import re
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _resample(col, n_points):
    idx = np.linspace(0.0, 1.0, col.size)
    return np.interp(np.linspace(0.0, 1.0, n_points), idx, col)


def repackage(in_fname, out_fname, n_points=120, max_eos=None):
    with h5py.File(in_fname, "r") as fin:
        names = sorted(fin["ns"].keys(), key=_natkey)
        if max_eos:
            names = names[:max_eos]
        rows = []
        for nm in names:
            d = fin["ns"][nm][:]
            M = _resample(np.asarray(d["M"], float), n_points)
            R = _resample(np.asarray(d["R"], float), n_points)
            Lam = _resample(np.asarray(d["Lambda"], float), n_points)
            hc = np.linspace(0.0, 1.0, n_points)          # normalized index
            nan = np.full(n_points, np.nan)
            # published rows: mark the rising part to M_max as stable
            imax = int(np.argmax(M))
            stable = (np.arange(n_points) <= imax).astype(float)
            rows.append(np.stack([hc, M, R, Lam, nan, nan, nan, nan, stable],
                                 axis=1))
        seq = np.stack(rows, axis=0)
    with h5py.File(out_fname, "w") as f:
        f.attrs["schema_version"] = "nmbackend.nss/1"
        f.attrs["representation"] = "tabular_hc/1"
        f.attrs["generator"] = f"nmbackend {_V} (converted from Landry: {in_fname})"
        f.attrs["created"] = datetime.datetime.utcnow().isoformat()
        f.attrs["fields"] = json.dumps(NSS_FIELDS)
        f.attrs["units"] = json.dumps({**NSS_UNITS, "hc": "normalized index"})
        f.attrs["converted_from"] = str(in_fname)
        f.create_dataset("sequence", data=seq, compression="gzip")
    return len(rows)


def resolve(in_fname, out_fname, n_points=80, max_eos=None):
    from nmbackend.eos.compose import eos_table_from_arrays
    from nmbackend.rift import write_nss
    with h5py.File(in_fname, "r") as fin:
        names = sorted(fin["eos"].keys(), key=_natkey)
        if max_eos:
            names = names[:max_eos]
        tables = []
        for nm in names:
            d = fin["eos"][nm][:]
            n = np.asarray(d["baryon_density"], float) * _GCM3_TO_NB
            p = np.asarray(d["pressurec2"], float) * _GCM3_TO_MEVFM3
            e = np.asarray(d["energy_densityc2"], float) * _GCM3_TO_MEVFM3
            o = np.argsort(n)
            tables.append(eos_table_from_arrays(n[o], e[o], p[o]))
    write_nss(out_fname, tables,
              params=[{"converted_from": f"{in_fname}:{nm}"} for nm in names],
              n_points=n_points)
    return len(tables)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("landry", help="input EOSSequenceLandry HDF5 (e.g. LCEHL)")
    ap.add_argument("--out", required=True, help="output NSSequence HDF5")
    ap.add_argument("--mode", choices=["repackage", "resolve"], default="repackage")
    ap.add_argument("--n-points", type=int, default=120)
    ap.add_argument("--max-eos", type=int, default=None)
    a = ap.parse_args()
    fn = repackage if a.mode == "repackage" else resolve
    n = fn(a.landry, a.out, n_points=a.n_points, max_eos=a.max_eos)
    print(f"converted {n} EOS ({a.mode}) -> {a.out}")


if __name__ == "__main__":
    main()
