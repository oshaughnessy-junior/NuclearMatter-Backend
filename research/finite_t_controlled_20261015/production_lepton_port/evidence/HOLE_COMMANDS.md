# Bounded hole repair commands

From /Users/rossma/Genesis_LANL, all writes inside this owned candidate leaf:

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_lepton_port/staged/tests/test_finite_t_leptons.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_lepton_derivatives.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_units.py --junitxml=outputs/nmb-finite-t-20261015/production_lepton_port/evidence/hole_tests.xml
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/hole_preservation_probe.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/verify_mirror.py
/Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/hole_freeze.py
```

98 tests PASS in8.73s;35 selected preservation/functional checks PASS. Raw
hole_tests.stdout/.stderr/JUnit and hole_preservation.stdout/.stderr/results
are retained. No new failure occurred during this narrow repair. Old failed
functional attempts remain in pre_hole_bound_repair/ and frozen reviews.
Every experiment<120s; no installations/dependency/API/default/tolerance
changes, production checkout modifications, branch, PR or commit.
