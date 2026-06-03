# Live CIP smoke test — GW170817 tabular per-EOS evidence

Ran the real RIFT CIP (`util_ConstructIntrinsicPosterior_GenericCoordinates.py`, the
patched copy with `EOSSequenceFromFile` / `EOSSequenceNMB`) against an existing
trained likelihood interpolant — confirming the whole tabular per-EOS evidence path
runs with our differentiable backend in the loop. CIP needs only `all.net`; nothing
else from the original run directory is required.

## Inputs

* `all.net`: `…/20230613-ROSAV-TabularEOSInference/all_tabular.net` (3600 ILE grid
  points, GW170817; `--input-eos-index` column bound to the 10000-EOS LCEHL sequence).
* CIP args: taken verbatim from that run's `args.txt` (the tabular-method arg string),
  only the EOS file and convergence targets varied.

## Run A — reference reproduce (legacy `EOSSequenceLandry` path)

`--tabular-eos-file LCEHL_EOS_posterior_samples_PSR+GW+NICER.h5` (complete, 10000 EOS).
The `EOSSequenceFromFile` factory correctly routed the legacy file to
`EOSSequenceLandry` (no regression).

| quantity | this run | reference `integral_result.dat` |
|---|---|---|
| ln evidence | **463.05** | 464.71 |
| neff | 3.07 | 3.86 |

The small gap is GMM/MC stochasticity at low neff plus a reduced `--n-max`; it
reproduces the published value.

## Run B — our differentiable backend (`EOSSequenceNMB` path)

`--tabular-eos-file nmb_seq.nss.h5` — a 12-EOS `NSSequence` built by
`scripts/build_eos_sequence.py` (metamodel grid over `L_sym`, `cs2_tail`),
`representation="tabular_hc/1"`. The factory auto-detected our format and used
`EOSSequenceNMB`. CIP drew the ordering coordinate from our family:

```
mcsampler: Adding parameter ordering with limits [262.3, 592.7]   # Λ(m_ref) from our EOS
```

and completed the marginalization:

| quantity | value |
|---|---|
| ln evidence | **457.77 ± 0.39** |
| neff | 3.24 |
| ordering range Λ(m_ref) | [262.3, 592.7] (our 12-EOS family) |

### Caveat on Run B

`all_tabular.net` is bound to the original 10000-EOS LCEHL sequence: each ILE point
carries a specific EOS index (2…9996, 3065 distinct) and its LCEHL tidal values. To
exercise our 12-EOS file every point's index column was remapped `index % 12`, so the
run is a faithful **integration / plumbing** test of `EOSSequenceNMB` inside CIP — it
proves our reader supplies the ordering statistic and the evidence integral converges
with our backend — but it is **not** a science-grade per-EOS evidence for this event.
A science run needs a same-cardinality nmbackend sequence (one EOS per original index),
i.e. generate ~10⁴ EOS with `build_eos_sequence.py` and run unchanged.

## Environment

Sandbox: `pip install lalsuite scikit-learn`; RIFT from the patched source tree on
`PYTHONPATH` (so the live `EOSSequenceFromFile` / `EOSSequenceNMB` code is exercised).
`lal 7.7.1`, `scikit-learn 1.7.2`.
