# Audit: RIFT `EOSSequenceLandry`, and a more robust replacement

*2026-06-02.  Before emitting EOS sequences for RIFT we audited the format/class it
uses (`RIFT/.../physics/EOSManager.py::EOSSequenceLandry`) and how the inference
driver consumes it.  Conclusion: the format is serviceable but has real weaknesses;
we design a more robust, efficient, self-describing replacement that ALSO emits the
legacy format for drop-in use today.*

## How RIFT consumes it (the contract)

`util_ConstructIntrinsicPosterior_GenericCoordinates.py` reduces the EOS ensemble to
a **1-D ordering statistic** = Λ at a reference mass:

```python
seq = EOSSequenceLandry(fname, load_ns=True, oned_order_name='Lambda',
                        oned_order_mass=m_ref, no_sort=True)
prior_range['ordering'] = [min(seq.oned_order_values), max(seq.oned_order_values)]
order_vals[i] = seq.lambda_of_m_indx(m_ref, eos_indx_i)        # per data point
# ... and in the likelihood: lookup_closest(order) -> indx;
#     lambda_of_m_indx(m1,indx), lambda_of_m_indx(m2,indx), m_max_of_indx(indx)
```

So the **consumer needs**: `oned_order_values`, `lambda_of_m_indx`, `R_of_m_indx`,
`m_max_of_indx`, `lookup_closest`, and the sorted/unsorted index bookkeeping.

## Weaknesses found

1. **Single-valued M↔Λ assumption (scientific bug for our use case).**
   `lambda_of_m_indx`/`R_of_m_indx` do `np.interp(m, M_sorted, log Λ)` with an
   explicit comment "ASSUME single-valued / no phase transition!".  For
   **twin-star / first-order phase-transition** EOS (which we now model) M(R) is
   multivalued and there can be two stable branches at the same mass — the interp
   silently mixes the stable and unstable branches and returns a wrong Λ.  This
   guts the value of our phase-transition work.

2. **Memory / I-O inefficiency.**  `load_ns` reads *every* per-EOS TOV table into a
   Python dict (their comments: "sucks up a lot of i/o time", "loads a LOT into
   memory").  Storage is one variable-length HDF5 dataset **per EOS** under a group,
   keyed by string names + `natsort`.  For GP ensembles (10^3–10^4 EOS × ~10^3
   points) this is slow and heavy; no lazy/vectorized access.

3. **Fragile index bookkeeping.**  `eos_ids`, `eos_names`, `oned_order_indx_original`,
   `oned_order_indx_sorted`, `oned_order_sorted` — multiple parallel index spaces
   with the warning "must be used with GREAT CARE to preserve order".

4. **Not self-describing.**  Units are passed at *load* time (`eos_tables_units`),
   not stored in the file; no schema version, no provenance (what generated it, EOS
   parameters, solver settings).  The dtype renaming
   (`eos_table_conv_units.dtype.names = ...`) is an error-prone hack.

5. **Thin field set.**  `ns` stores only M, R, Λ.  No baryon mass M_b, moment of
   inertia I, central density, stability flag, or c_s² — quantities wanted by
   hyperpipeline / multimessenger / I-Love-Q analyses.

6. **Read-time recompute.**  Λ(m) interpolation is redone per call in the inner
   inference loop; nearest-neighbor-in-Λ EOS selection is crude.

## Replacement design: `NSSequence` (robust, efficient, self-describing)

Key idea: **store everything as a function of central pseudo-enthalpy `h_c`**, which
is *monotone* along the sequence — so M(h_c), R(h_c), Λ(h_c) are single-valued and
**branch structure is explicit** (a `stable` flag + the sign of dM/dh_c), eliminating
the multivalued-M bug.  Lookups M→Λ are then done **branch-aware** (restrict to a
stable, monotone M segment).

**HDF5 layout (self-describing):**
- root attrs: `schema_version`, `generator`, `created`, `units` (explicit strings).
- `sequence` — a single regular dataset, shape `(n_eos, n_points, n_fields)`, fields
  `[hc, M, R, Lambda, M_b, I, rho_c, cs2_c, stable]` on a common h_c grid
  (contiguous, lazy-sliceable, vectorizable).  Missing quantities (M_b, I) stored as
  NaN until the solver provides them — the schema is future-proof.
- `eos_params` — per-EOS parameter dict / GP-draw label (provenance + parametric map).
- `eos_tables` — optional microphysical EOS `(n_eos, n_pts, [n, p, eps, cs2, mu])`
  with explicit units.

**Reader (`NSSequence`):** lazy/contiguous load; **branch-aware** vectorized
`lambda_of_m`, `radius_of_m`, `m_max` over all/selected EOS; multi-branch resolution
for twins; precomputed ordering statistics.

**Backward compatibility (drop-in today):** `write_landry_legacy()` emits the exact
`EOSSequenceLandry` HDF5 (`id`, `ns/<name>`{M,R,Λ}, `eos/<name>`{pressurec2,
energy_densityc2, baryon_density}); and a `LandryCompat` accessor class provides
`oned_order_values`, `lambda_of_m_indx`, `R_of_m_indx`, `m_max_of_indx`,
`lookup_closest` — **with the branch-aware fix** — so existing RIFT drivers run
unchanged but correctly on phase-transition EOS.

## Plan
1. ✅ Audit (this doc).
2. Writer for the new `NSSequence` HDF5 (vs h_c, self-describing) from nmbackend EOS.
3. Reader `NSSequence` with branch-aware vectorized accessors.
4. Legacy `EOSSequenceLandry` writer + `LandryCompat` reader (branch-aware fix).
5. Validate: round-trip; **twin-star/phase-transition** correctness (where the legacy
   interp fails); replay the RIFT consumer logic for drop-in confirmation.
