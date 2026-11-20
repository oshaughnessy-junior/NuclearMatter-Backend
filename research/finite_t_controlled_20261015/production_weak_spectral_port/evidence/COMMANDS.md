# Exact command pattern and execution scope

Working directory `/Users/rossma/Genesis_LANL`.
Python `/Users/rossma/miniconda3/envs/junior_tools/bin/python`.
Every scientific command used `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1`.
Candidate imports used the explicit owned `test_mirror/src` path; an initial
pytest missing that path failed collection and is preserved.

- `python -m pytest -q .../test_mirror/tests/test_weak_reference_current.py .../test_weak_reference_moments.py` with explicit owned `PYTHONPATH`:40PASS before spectra interpretation.
- `python evidence/batch_selected.py`: serial separate processes `run_selected.py NAME CHANNEL E base|fine`, each timeout119s,34actual selected computations. Initial EC2base is additionally retained.
- `python evidence/run_differential.py`:4actual original-shell points.
- `python evidence/independent_phase.py`: alternate original longitudinal-delta/transverse-momentum rules and literal spin sums at2fresh points.
- `python evidence/isolated_controls.py`:11selected isolated control variations (each physical call≤120s; whole script ~24s).
- `python evidence/batch_pilot.py`:18actual finite-range reconstructions,2/4nodes,3channels,3common radiation closures, each separate process timeout119s.
- `python -m pytest -q .../test_mirror/tests --junitxml=.../evidence/final_tests.xml` with explicit ownedPYTHONPATH:86PASS,20.841s (40unchangedStageA+13cold+33new).
- `python evidence/guard_probe.py` then separate `guard_probe_repaired.py`: invalid-bath admission before physical calls.
- `python evidence/record_guard_probe.py` then separate `record_guard_repaired.py`: incomplete/forged result qualification admission; initial files unchanged.
- `python evidence/validate_saved_pilots.py`:18valid saved payloads identical;108negative contract probes. No fresh physical calls.
- `python evidence/assess.py`:250selected numerical and saved-node arithmetic checks. Initial wide-glob reporting failure/source preserved; corrected explicit18path enumeration performs no physical calls.
- `python evidence/static_api_audit.py`: Python3.9syntax and real package origins; minruntime/fullCI NOT executed.

`LOCK.json` was written before any candidate validation. `ADDITIONAL_CONTROL_LOCK.json`
was written before isolated/range calls. No science cache, installs, remote writes,
Git operations, dependencies or actual production-checkout edits occurred.
