#!/usr/bin/env python3
"""Build an EOS *sequence* for RIFT from parametric models or ingested tables.

Produces the robust self-describing ``NSSequence`` HDF5 (stored vs central enthalpy)
and, optionally, the legacy ``EOSSequenceLandry`` HDF5 that today's RIFT consumes via
``--internal-tabular-eos-file``.

Examples
--------
# A metamodel parameter grid (cartesian product) -> sequence:
PYTHONPATH=src python scripts/build_eos_sequence.py \
    --model metamodel --grid "L_sym=40,50,60,70;cs2_tail=0.3,0.4,0.5" \
    --out seq.nss.h5 --legacy seq.landry.h5

# Explicit parameter draws from a JSON list of dicts:
PYTHONPATH=src python scripts/build_eos_sequence.py \
    --model rmft --params-file draws.json --out seq.nss.h5

# Ingest tabular EOS (CompOSE dirs / CMF CSVs / simple tables) as the sequence:
PYTHONPATH=src python scripts/build_eos_sequence.py \
    --ingest compose:/path/eosA cmf:/path/CMF_stable.csv --out seq.nss.h5
"""
import argparse
import itertools
import json

from nmbackend.eos import (MetaModelEOS, RMFTEOS,
                           load_compose, load_cmf, load_simple)
from nmbackend.rift import write_nss, write_landry_legacy


def _model(name):
    return {"metamodel": MetaModelEOS, "rmft": RMFTEOS}[name]()


def _parse_grid(spec):
    """'a=1,2;b=0.3,0.4' -> list of dicts (cartesian product), values floats."""
    axes = {}
    for part in spec.split(";"):
        if not part.strip():
            continue
        key, vals = part.split("=")
        axes[key.strip()] = [float(v) for v in vals.split(",")]
    keys = list(axes)
    return [dict(zip(keys, combo)) for combo in itertools.product(*axes.values())]


def _ingest(spec):
    kind, path = spec.split(":", 1)
    return {"compose": load_compose, "cmf": load_cmf, "simple": load_simple}[kind](path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="metamodel",
                    choices=["metamodel", "rmft"])
    ap.add_argument("--grid", default=None, help="param grid 'a=1,2;b=0.3,0.4'")
    ap.add_argument("--params-file", default=None, help="JSON list of param dicts")
    ap.add_argument("--ingest", nargs="+", default=None,
                    help="tabular EOS as 'compose:DIR' / 'cmf:FILE' / 'simple:FILE'")
    ap.add_argument("--out", required=True, help="output NSSequence HDF5")
    ap.add_argument("--legacy", default=None, help="also write legacy Landry HDF5")
    ap.add_argument("--n-points", type=int, default=80, help="points per sequence")
    ap.add_argument("--no-eos-tables", action="store_true",
                    help="omit microphysical EOS tables from the legacy file")
    a = ap.parse_args()

    if a.ingest:
        tables = [_ingest(s) for s in a.ingest]
        params = [{"ingest": s} for s in a.ingest]
    else:
        model = _model(a.model)
        if a.params_file:
            with open(a.params_file) as fh:
                params = json.load(fh)
        elif a.grid:
            params = _parse_grid(a.grid)
        else:
            params = [{}]                      # single default EOS
        tables = [model.eos_table(p) for p in params]
        for p in params:
            p["model"] = a.model

    print(f"built {len(tables)} EOS; writing NSSequence -> {a.out}")
    write_nss(a.out, tables, params=params, n_points=a.n_points)
    if a.legacy:
        write_landry_legacy(a.legacy, tables,
                            eos_tables=(None if a.no_eos_tables else tables),
                            n_points=max(a.n_points, 120))
        print(f"wrote legacy EOSSequenceLandry -> {a.legacy}")


if __name__ == "__main__":
    main()
