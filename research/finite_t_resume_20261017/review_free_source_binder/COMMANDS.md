# Exact review commands

From /Users/rossma/Genesis_LANL, scientific interpreter
/Users/rossma/miniconda3/envs/junior_tools/bin/python, with
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1:

- eos_probe.py (original momentum roots first, actual candidate then responses)
- source_probe.py empty
- guard_probe.py (synthetic guards, no physical-rate calls)
- source_probe.py direct
- source_probe.py LTE
- package_probe.py (stdlib source/zip checks, no rates)
- python -I wheel_probe.py (fourth source group, then13cold/unit tests)

All scripts are in this review directory; no author driver was executed. Source
jobs use signal.alarm(89), one source group per process and no parallel source
jobs. The first package inventory and first wheel parsing attempts failed in
reviewer harnesses; their exact scripts/result/error are retained under initial_*.
No physical call occurred in the failed wheel parsing attempt. Corrected package
scope excludes only explicitly hash-ledgered generated build/cache/egg-info paths.
The built wheel is unpacked, not installed. Actual COLD_TESTS.xml records13tests.
No scientific calls should be repeated merely to regenerate final metadata.
