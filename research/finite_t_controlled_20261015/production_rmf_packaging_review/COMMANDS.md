All commands executed from /Users/rossma/Genesis_LANL except the owned build cwd below. No installation, download, environment mutation, git operation or physical grid.

```
/Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_rmf_packaging_review/prepare.py
```

Build cwd: outputs/nmb-finite-t-20261015/production_rmf_packaging_review/package_copy

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -c 'import setuptools.build_meta as b; print(b.build_wheel("../wheels"))' > ../build_stdout.txt 2> ../build_stderr.txt
```

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/production_rmf_packaging_review/check_wheel.py > outputs/nmb-finite-t-20261015/production_rmf_packaging_review/import_stdout.txt 2> outputs/nmb-finite-t-20261015/production_rmf_packaging_review/import_stderr.txt
```

The wheel check ran three times, each about10seconds and below120seconds. Initial and second audit-predicate failures were preserved; only the owned audit fixture changed, as explained in initial_fixture_failure.txt. No numerical tolerance or candidate changed.

```
/Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/production_rmf_packaging_review/static_audit.py > outputs/nmb-finite-t-20261015/production_rmf_packaging_review/static_stdout.txt 2> outputs/nmb-finite-t-20261015/production_rmf_packaging_review/static_stderr.txt
```

Fresh pre-batch quota04:59:42UTC: exact account82f4cfd9-9e20-47cf-9997-761328d89f60, codex10080min27%used, reset1792652325. No resets. No physical-model source was imported outside the wheel. Example is read from the owned exact nine-file overlay and imports the actual wheel package under Python-I.
