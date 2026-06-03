# RIFT tabular-EOS pipeline: where the nmbackend sequence plugs in

This documents the exact call chain by which a `nmbackend` EOS sequence produces
per-EOS marginal evidence through RIFT's tabular hyperpipeline, and the two
integration points we added.

## Call chain

```
util_RIFT_pseudo_pipe.py  --internal-tabular-eos-file FILE.h5
    │   (util_RIFT_pseudo_pipe.py:195 arg; :735 forwards to the workflow generator;
    │    :1598 adds --use-tabular-eos-file)
    ▼
workflow / DAG generator
    │   sets per-CIP-worker:  --tabular-eos-file FILE.h5   --input-eos-index N
    ▼
util_ConstructIntrinsicPosterior_GenericCoordinates.py   (CIP)
    │   --fname all.net            (intrinsic-likelihood interpolant from ILE)
    │   --fname event-<id>.net
    │   --tabular-eos-file FILE.h5 --input-eos-index N
    ▼
EOSManager.EOSSequenceFromFile(fname=FILE.h5, load_ns=True,
                               oned_order_name='Lambda', oned_order_mass=m_ref)
    │   auto-detects format from HDF5 attrs:
    │     representation="tabular_hc/1" or schema_version="nmbackend.nss/1"
    │       → EOSSequenceNMB        (our NSSequence, branch-aware, vs central enthalpy)
    │     otherwise
    │       → EOSSequenceLandry     (legacy; incl. the upstream LCEHL file)
    ▼
per-EOS evidence:
    ordering coordinate  = Λ(m_ref) for each EOS         (oned_order_values)
    Λ(m1), Λ(m2)         = lambda_of_m_indx(m, eos_indx) (stable branch, log-interp)
    marginalize all.net over masses → evidence per EOS index
```

The legacy per-EOS path (`--using-eos file:grid.dat --using-eos-index N --eos-param
spectral`, used in the older NICER hyperpost makefiles) builds a *single* lalsim
spectral EOS object per CIP call. The **tabular** path above is the one we target: the
EOS index is the sampled ordering coordinate and one run sweeps the whole sequence.

## The two integration points (committed)

1. `EOSManager.EOSSequenceNMB` — a self-contained reader for the `NSSequence` HDF5
   that subclasses `EOSSequenceLandry`, extracting the stable rising branch into the
   same in-memory `eos_ns_tov` dict so every inherited accessor works unchanged and is
   branch-safe by construction.

2. `EOSManager.EOSSequenceFromFile` — a format-detecting factory; CIP line ~1982 now
   calls it instead of `EOSSequenceLandry` directly. Either file type — ours or the
   upstream Landry/LCEHL file — drops in with no further code change.

A NUL-cost fallback also exists: `scripts/build_eos_sequence.py --legacy` writes the
exact `EOSSequenceLandry` HDF5 (only the stable branch, so RIFT's single-valued
`M→Λ` interpolation is correct for phase-transition EOS), which the *unmodified*
upstream RIFT also consumes.

## Generating the sequence

```
PYTHONPATH=src python scripts/build_eos_sequence.py \
    --model metamodel --grid "L_sym=40,50,60,70;cs2_tail=0.3,0.4,0.5,0.6" \
    --n-points 120 --out seq.nss.h5 --legacy seq.landry.h5
```

`--model rmft`, `--params-file draws.json`, or `--ingest compose:DIR cmf:FILE.csv`
select the RMFT solver, explicit parameter draws, or ingested tables instead. See
`examples/gw170817_tabular/` for the GW170817 end-to-end recipe.

## Future: enthalpy-path-integral and compressed representations

The single-branch `Λ(m_ref)` ordering is a limitation of today's RIFT framing, not of
the stored data: `NSSequence` keeps everything vs central enthalpy `h_c` (monotone),
so a future driver can path-integrate along `h_c` and represent disconnected (twin)
branches natively. Likewise, `SequenceRepresentation` + `open_sequence` make the
internal storage pluggable — a GP / spectral / PCA representation registers as a new
`representation` tag with the same accessor protocol, so neither CIP nor the generator
changes when we move off pure tabular draws. See `docs/backlog.md`.
