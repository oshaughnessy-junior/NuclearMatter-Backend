# Executed protocol and source epochs

Working directory `/Users/rossma/Genesis_LANL`; Python
`/Users/rossma/miniconda3/envs/junior_tools/bin/python`.
All scientific commands set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`.

Original epoch (candidate18da0...):

- `python .../spectral_incident_convergence/batch.py EC`
- `python .../spectral_incident_convergence/batch.py PC`
- `python .../spectral_incident_convergence/postprocess.py EC|PC`

Each batch dispatches exact declared orders4/8/16/24/32,4nodes per separate
`nodes.py CHANNEL ORDER FIRSTINDEX` child, timeout119s. Source hashes bracket
each physical call. Original node jobs/pins/LOCK/results/partial checkpoint and
manifest remain unchanged. No beta/public-agreement call was made in this epoch.

Successor epoch (candidate dcfbafe8..., formal DIFFERENT guard-only identity):

- Wrote SUCCESSOR_SOURCE_ADMISSION.json and SUCCESSOR_SOURCE_PIN.json BEFORE calls.
- `python .../spectral_incident_convergence/batch_successor.py NEUTRON_DECAY`
- `python .../spectral_incident_convergence/postprocess_successor.py EC|PC|NEUTRON_DECAY`
- `python .../spectral_incident_convergence/batch_public_successor.py`

The beta batch uses separate `nodes_successor.py NEUTRON_DECAY ORDER FIRSTINDEX`
children and `raw_successor/`. Public comparisons are six separate children
`public_agreement_successor.py CHANNEL empty|LTE`, each fresh actualAPI order4
fine profile with timeout119s. EC/PCreference node execution remains ORIGINAL;
beta/public execution is REPAIRED. New postprocessing creates distinct successor
reports and preserves exact per-row source grouping rather than overwriting old
fingerprints/results.

Analysis/admission (no physical coefficient calculation):

- `python .../spectral_incident_convergence/check_admission.py`: valid COPY and12negative COPY probes.
- `python .../spectral_incident_convergence/analyze_successor.py`:880PASS/29retained coarseFAIL, allfinesttargetspass.
- Source verification: all122original checkpoint paths unchanged;211moment/arithmetic identities pass; current kernel/review successor hashes and original frozen source snapshot mapping reverified.

StrictJSON raw contains actual node logs/support/representation and complete
rules. CANONICAL_FINITE_RANGE_RESULTS.json is a derived evidence bundle, not a
runtime coefficient table or cache. No physics cache/fit/surrogate, installs,
new dependencies, candidate edits, Git/production/paper/plot work occurred.
Wholeenergy/tails and DIFFERENT study review remain OPEN.
