# Quadrature repair commands and raw records

Working directory `/Users/rossma/Genesis_LANL`. Writes are confined to this
owned candidate directory. Original commands/evidence remain unchanged in
`pre_quadrature_repair/`. Existing junior_tools Python and exact source
mirror are used; nothing was installed.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_lepton_port/staged/tests/test_finite_t_leptons.py --junitxml=outputs/nmb-finite-t-20261015/production_lepton_port/evidence/repair_frozen_tests.xml
```

Final:81 PASS in6.88s; `repair_frozen_tests.stdout/.stderr`, JUnit.
Earlier retained run `repair_initial_tests.*`: original60 PASS in6.97s.
`repair_tests.*`:79 PASS/2 oracle FAIL, square-underflow at z600; corrected
only oracle algebra. `repair_final_tests.*`:81 PASS in6.45s before stable
free-energy identity; the frozen run above includes that final source edit.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/repair_momentum_probe.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/repair_reproduce_initial.py
```

Final momentum213 PASS,0 FAIL in `repair_momentum_results.json` and
`repair_frozen_momentum.stdout/.stderr`. The initial203 PASS/10 unresolved
reference FAIL (one degenerate muon point,128→192 nodes) are preserved in
`initial_repair_momentum_*`; final384→512 node refinement resolves them.
The second command records a fresh unique import of the original module,
all four reproduced failures, and repaired values/hashes in
`repair_initial_four_reproduction.json/.stdout/.stderr`.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_lepton_derivatives.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_units.py --junitxml=outputs/nmb-finite-t-20261015/production_lepton_port/evidence/repair_frozen_cold.xml
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/staged/examples/finite_t_leptons.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/verify_mirror.py
```

Final cold13 PASS; `repair_frozen_cold.stdout/.stderr` and JUnit. Example
finite JSON, empty stderr in `repair_example.json/.stderr`. Mirror71 exact
pinned package blobs and candidate overlay PASS. Read-only real checkout
checks inside verify_mirror.py use GIT_OPTIONAL_LOCKS=0 and do not write it.
Fresh main GitHub fetch_commit(main) returns ffb5cc...; metadata is recorded
in `repair_main_recheck.json`. Final freeze checks grammar, snapshot/review
SHA preservation and every staged/mirror identity; see `repair_preservation.json`.

No test tolerance/default was loosened. All experiments completed below120s.
Original failed independent acceptance remains unchanged; author results do
not replace the required different independent repair review or full CI.
