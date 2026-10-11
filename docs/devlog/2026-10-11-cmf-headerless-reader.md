# Dev log — 2026-10-11 — CMF CSV reader dropped the first data row

**Who:** R. O'Shaughnessy (with AI pair).  Bug fix + doc correction.

## Bug
`eos.compose.read_cmf_csv` / `load_cmf` defaulted to `skip_header=1`.  MUSES CMF++
v1.1.0 output CSVs are **intentionally headerless** (`clean_output.py`,
`to_csv(header=False)`), so the default silently discarded the first data row.  On the
default C4 run (`CMF_output_stable.csv`, 399 rows x 21 cols) that row is the
mu_B = 922 MeV nuclear-matter onset, n_B = 0.1535 fm^-3.  The 2026-06-02 validation
used a synthetic CSV written *with* a header, which is why it passed.

## Fix
- `skip_header=None` default; `_count_header_lines` skips leading non-numeric lines
  (0 for real CMF output, 1 for a headed export).  An int still forces a fixed skip.
- Docstring now records the full 21-column layout (checked against
  `api/OpenAPI_Specifications_CMF.yaml`, schema `CMF_output_all_EoS_variables`) and
  the 11-column Lepton layout; cols 5/8/9 coincide so one mapping serves both.
- Tests (+6): verbatim 3-row headerless sample keeps the onset row and shows the old
  default dropping it; header auto-detect equals the headerless read; end-to-end on
  the real table when present (skipped otherwise).
- `docs/muses-integration.md`: layout table, headerless note, and the physics of
  col 5 (n_B incl. the Polyakov-loop dp/dmu_B term) vs col 18 (without it) — differ on
  161/399 rows of the default table; col 5 is the one consistent with (eps, p).

## Adversarial review (1 agent): 3 medium, 3 low -- all fixed
- UTF-8 BOM made the first row "non-numeric" again (re-introducing the bug on an
  Excel round-trip): open with `utf-8-sig` in both the detector and genfromtxt.
- A pandas re-export header `0,1,...,20` parsed as a data row and injected a bogus
  EOS point: an all-integer `range(ncol)` row is now treated as a header.
- `CMF_output_for_Lepton_baryons.csv` (the file the doc recommends) has 3 metastable
  p < 0 rows; the filter never required p > 0, so they were clamped to p = 1e-12 and
  produced tied h.  Filter now requires p > 0; real-file test added for Lepton too.
- Empty / all-text / too-narrow files crashed with opaque IndexErrors: ValueError
  naming the file and columns.  Real-file tests use `NMB_CMF_OUTPUT_DIR` /
  `NMB_CMF_REQUIRED` (repo convention, cf. NSCOOL_REQUIRED) instead of one home path.
- Doc wording "agree wherever Phi = 0" qualified to printed precision (4 hadronic rows
  have |Phi| ~ 1e-22).

## Integration check on the real tables (genesis-gpu env, sporcsubmit)
`load_cmf` -> `build_family` (40 models) on the actual v1.1.0 default-run output, new
default vs the old `skip_header=1`:

| table | rows kept new/old | M_max new/old [Msun] | R(1.4) new/old [km] | Lambda(1.4) new/old |
|---|---|---|---|---|
| CMF_output_stable.csv (399x21) | 388 / 387 | 1.9348 / 1.9334 | 11.221 / 11.491 | 399 / 405 |
| CMF_output_for_Lepton_baryons.csv (389x11) | 386 / 385 | 1.9937 / 1.9938 | 11.221 / 11.490 | 400 / 406 |

In both tables the dropped mu_B = 922 row is the lowest-density physical core point
(after the p > 0 filter removes the Lepton file's three metastable p < 0 rows), so
losing it moved the crust-core join from n_B = 0.1535 to 0.1616 fm^-3 and shifted
R(1.4) by ~270 m -- a real, not cosmetic, error for anything that consumed the old
reader.  (These are CMF's grand-canonical mu_S = -50, mu_Q = 0 grids, not a
beta-equilibrium NS line; the numbers are an ingestion check, not a CMF NS result.
Tables: `~/q2c_deps/cmf-v1.1.0-runs/C4_muS-50_muQ0/`.)
`scripts/build_eos_sequence.py --ingest cmf:...` writes a valid NSSequence from the
stable table.

## Downstream
Q2C pins NMB at its `NMB_PIN`; bump it in a separate Q2C change.
