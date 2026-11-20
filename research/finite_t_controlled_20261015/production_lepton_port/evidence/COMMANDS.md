# Executed verification commands

Working directory is `/Users/rossma/Genesis_LANL`. All writes are inside
`outputs/nmb-finite-t-20261015/production_lepton_port/`.

The real NMB import mirror is built from a clean read-only local checkout at
`5e0208cccf386c723f10955d0c81390fa1ecb972` plus exact owning-GitHub readbacks of
all package files changed through `ffb5cc72949f2e5401251847aef4c9c7a98dde78`.
`github_source_records.json`, `local_to_head_compare.json` and
`mirror_verification.json` record this construction. Nothing was pulled,
checked out, reset, committed or written in the real checkout. Git commands
were read-only `rev-parse HEAD`, `status --porcelain --untracked-files=no` and
`ls-files --stage src`, with `GIT_OPTIONAL_LOCKS=0` and fsmonitor disabled.

The first 58-test run used the same candidate module, before adding the two
explicit current-cold-API conversion comparisons. It passed. That first
command added `-q` to the command below and omitted the junit argument; its
raw output is `test_1_stdout.txt` / `test_1_stderr.txt`. The final run is:

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_lepton_port/staged/tests/test_finite_t_leptons.py --junitxml=outputs/nmb-finite-t-20261015/production_lepton_port/evidence/test_final.xml
```

Result: **60 passed in4.29s**, exit0, stderr empty. Output is
`test_final_stdout.txt` / `test_final_stderr.txt`.

```sh
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -m pytest -c outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/pyproject.toml outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_lepton_derivatives.py outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/tests/test_units.py --junitxml=outputs/nmb-finite-t-20261015/production_lepton_port/evidence/cold_regressions.xml
```

Result: **13 passed in3.32s**, exit0, stderr empty. These are exact existing
head tests, including energy Hessians/vector independence/JIT and unit
conversions. Output is `cold_stdout.txt` / `cold_stderr.txt`.

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/production_lepton_port/evidence/verify_mirror.py
PYTHONPATH=outputs/nmb-finite-t-20261015/production_lepton_port/test_mirror/src JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_lepton_port/staged/examples/finite_t_leptons.py
```

Result: all71 unchanged-head package blobs verify, candidate overlay matches
staging, and the example emits finite named-unit JSON with empty stderr.
`import_environment.json` records actual package module origins with the
real package initializers. No namespace stubs, fake JAX, test monkeypatch or
substituted constants were used. Python3.9 grammar was checked with
`ast.parse(..., feature_version=(3,9))` for all three staged Python files;
that is syntax validation, not execution under Python3.9.

Environment: Python3.12.12, NumPy2.3.5, SciPy1.15.2, JAX0.10.1,
Diffrax0.7.2, pytest9.0.3, existing `junior_tools` environment. This satisfies
the package runtime lower bounds, while differing from the separate conda/
pixi validation environment's NumPy<2.3 choice. No minimum-version matrix,
full repository suite, admitted remote CI, wheel build/install or scientific
independent review was run. These remain explicit gates.

No numerical test failed and no test tolerance was relaxed. Two read-only
discovery commands produced path diagnostics: the initial optional worktree
shell glob had no matches; later two relative `cat` paths were evaluated
from the read-only source checkout rather than Genesis. Neither command
executed scientific code or changed files. The diagnostics are preserved in
`discovery_diagnostics.txt`; all actual test/import/example results above
remain separate.
