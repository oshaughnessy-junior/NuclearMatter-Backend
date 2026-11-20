# Executed commands

Working directory /Users/rossma/Genesis_LANL. All writes are inside the new
production_baryon_port/ leaf. Fresh quota/account/window checks at03:09:57,
03:13:37,03:15:39 and03:18:35UTC all match the same account/reset and25% used.
No reset,install,production,paper,git,branch or PR operation.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_baryon_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_baryon_port/evidence/forward_gate.py
```

48 PASS before launching response tests. Actual same-mass accepted particle
EOS and all forward inputs/results are in forward_gate.json and stdout/stderr.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_baryon_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_baryon_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_baryon_port/staged/tests/test_finite_t_baryons.py outputs/nmb-finite-t-20261015/production_baryon_port/test_mirror/tests/test_lepton_derivatives.py outputs/nmb-finite-t-20261015/production_baryon_port/test_mirror/tests/test_units.py --junitxml=outputs/nmb-finite-t-20261015/production_baryon_port/evidence/tests_final.xml
```

49 PASS in4.88s=36candidate+13 unchanged cold. The preceding32-case command
omitted the two cold-test paths, used tests_initial.xml and preceded addition
of three dimensionful cases and one mu-offset rejection. It passed in2.11s.
Both raw results/stderr/JUnit are retained, with no numerical failure/loosened
gate. The final staged source is unchanged between these runs and forward gate.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_baryon_port/test_mirror/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_baryon_port/staged/examples/finite_t_baryons.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_baryon_port/test_mirror/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_baryon_port/evidence/verify.py
/Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/production_baryon_port/evidence/freeze.py
```

Example finite JSON,empty stderr;71 exact baseline blobs and real imports
verify, candidate overlay matches, Python3.9 grammar parses. main_recheck.json
records fresh mainffb5cc from owning GitHub read API. No source checkout or
existing lepton candidate/mirror was written. Each experiment<120s. This does
not supply full-suite/CI/packaging/minimum-version execution or different
independent acceptance.
