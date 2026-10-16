# Minimum-compatible log-occupation follow-up

Working directory `/Users/rossma/Genesis_LANL`; all writes stay in this owned
candidate directory. `pre_scipy17_repair/` preserves75 files and
`INDEPENDENT_QUADRATURE_REVIEW_PIN.json` pins all15 files of the separate
665 PASS/1 STATIC COMPATIBILITY FAIL review. Earlier snapshots and failures
are unchanged.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_lepton_port/staged/tests/test_finite_t_leptons.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_lepton_derivatives.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_units.py --junitxml=outputs/nmb-finite-t-20261015/production_lepton_port/evidence/compat_tests.xml
```

95 PASS in8.77s, zero errors/skips: previous81 candidate,1 new helper,13
unchanged cold JIT/Hessian/unit tests. Raw `compat_tests.stdout/.stderr`.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/compat_import_probe.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/compat_momentum_probe.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/compat_thermal_guard_diagnostic.py
```

Import/algebra/preservation probe259 PASS, including2 explicitly labeled
matched thermal rejections (not physical forward passes). This isolated
missing-export simulation does not execute SciPy1.7. All successful selected
state/thermal outputs are bitwise identical to the preceding candidate.
The first probe version stopped at the old candidate's thermal guard before
finishing, with traceback preserved as `initial_compat_import*`; final script
records both old/new outcomes explicitly rather than suppressing this domain
limitation. No candidate numerical change was made to repair that guard.

Momentum/root probe213 PASS; its result is in `compat_momentum_results.json`,
separate from all preceding frozen records. It reuses the author's independent
momentum-variable oracle rather than claiming a new independent review.

Thermal diagnostic records the old/new unresolved functional-domain limit
m/T600,mu/T±.53. The cold hole integral underflows with zero absolute tolerance
and reports precision failure, although the total thermal increment reconstructed
from resolved total/rest energy is ordinarily represented. State inputs,
quad status, dimensional reconstruction and independent Bessel expectation
are in `compat_thermal_guard_diagnostic.json`. No public thermal forward value
is claimed there; the domain/guard must be assessed by the different reviewer
before a draft PR.

```sh
/Users/rossma/miniconda3/envs/junior_tools_alt/bin/python -B -c 'import sys,importlib.metadata,json; print(json.dumps({"python":sys.version,"packages":{p:importlib.metadata.version(p) for p in ["numpy","scipy","jax","diffrax","pytest"]}},indent=2))'
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/verify_mirror.py
/Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/evidence/compat_freeze.py
```

Alternate versions equal the current existing environment; no installs or
NumPy<2.3/minimum-version qualification. The final checks verify71 baseline
package blobs, candidate mirror identity, all four preservation pin sets,
Python3.9 grammar, final JUnit/probe results and exactly five staged paths.
Every experiment is below120s. No dependency/default/tolerance or public API
change; no production checkout, branch, PR or commit modification.
