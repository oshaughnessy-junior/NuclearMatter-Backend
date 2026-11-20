# Exact bounded execution commands

All scientific child processes set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1, PYTHONDONTWRITEBYTECODE=1 and use /Users/rossma/miniconda3/envs/junior_tools/bin/python. Each process is bounded separately, serial physical work; no installs. Scientific runtime package path is production_weak_port/test_mirror/src. No generic study module is imported.

1. `python -m pytest -q test_mirror/tests/test_weak_reference_current.py --junitxml=evidence/current.xml` (9 original current-only gates, before thermal moments).
2. `python evidence/collect.py CASE CHANNEL [reference]`, CASE0/1/2;CHANNEL EC/PC/NEUTRON_DECAY. Current candidate records exist for all9combinations. Fresh original reference records exist for cases1/2 and all3channels. Each complete file retains actual controls/state/current and positive F/B species moments; candidatefiles also fingerprint actual five numerical modules.
3. `python evidence/anchors.py`: actual absolute cross sections and beta/proper-time/Doppler checks.
4. `python evidence/tail.py CHANNEL`: isolated actual tail controls at case2.
5. `python evidence/assess.py`: stdlib read-only saved comparisons/ledgers and sourcehash checks, no ratecalls.
6. `PYTHONPATH=test_mirror/src python -m pytest -q test_mirror/tests/test_weak_reference_current.py test_mirror/tests/test_weak_reference_moments.py test_mirror/tests/test_lepton_derivatives.py test_mirror/tests/test_units.py --junitxml=evidence/final_tests.xml`: final53tests.
7. `python evidence/verify.py`: actual realpackage origin, baseline71gitblobs, exactoverlay11files, row-source identity, staticminimumAPI and allfrozenresearchpins.
8. `PYTHONPATH=test_mirror/src python test_mirror/examples/free_weak_reference.py`: fresh actual threechannel example.

Commands shown relative to the ownedleaf for readability; actual executions used repository-relative paths as retained in stdout/stderr/XML. Executing collectors again would overwrite existing evidence and is not an admission shortcut; independent reviewers should write only their owned directories. No evidence record is a physical uncertainty estimate. Independent review/packaging/CI/minimumruntime remain separate.
