# Exact normalized-hole follow-up

All commands run from /Users/rossma/Genesis_LANL, with writes confined to this
owned candidate leaf and no installs. Fresh pre-batch clock/quota02:07:26UTC
verified account82f4cfd9-9e20-47cf-9997-761328d89f60,codex10080min reset1792652325,
24% used; no reset. Each experiment is below120s.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_lepton_port/staged/tests/test_finite_t_leptons.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_lepton_derivatives.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_units.py --junitxml=outputs/nmb-finite-t-20261015/production_lepton_port/evidence/unit_hole_tests.xml
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/unit_hole_probe.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/verify_mirror.py
/Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/unit_hole_freeze.py
```

103 tests PASS in9.11s and42 preservation/locked-point checks PASS, with
empty stderr. Exact initial four muon exceptions are replayed from the saved
candidate; original reviewer104 PASS/6 FAIL records remain unchanged. No
new failure occurred in this repair. Source/state preservation and exact
restored tiny-hole values are separately recorded. This author replay of the
reviewer's frozen locked points does not replace a DIFFERENT review of the
new source. No broader grid, source moment algorithm, production, PR or git
change was made. Supported/minimum-version, pin-environment and full CI/
packaging gates remain open.
