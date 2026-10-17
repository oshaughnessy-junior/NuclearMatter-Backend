Executed from /Users/rossma/Genesis_LANL except owned build cwd. Initial pre-acceptance preparation is preserved verbatim under pre_numerical_acceptance/ and was never accepted as runtime evidence. REFRESH_PROOF.json pins that whole initial preparation and restricts repaired overlays to _weak_quadrature.py plus its moment test, with helper AST identical outside SignedQuantity.__post_init__.

```
/Users/rossma/miniconda3/bin/python /tmp/weak_packaging_refresh.py
/Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/production_weak_packaging_review/prepare.py > outputs/nmb-finite-t-20261015/production_weak_packaging_review/preparation_stdout.txt 2> outputs/nmb-finite-t-20261015/production_weak_packaging_review/preparation_stderr.txt
```

Build cwd: outputs/nmb-finite-t-20261015/production_weak_packaging_review/package_copy

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -c 'import setuptools.build_meta as b; print(b.build_wheel("../wheels"))' > ../build_stdout.txt 2> ../build_stderr.txt
```

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/production_weak_packaging_review/check_wheel.py > outputs/nmb-finite-t-20261015/production_weak_packaging_review/import_stdout.txt 2> outputs/nmb-finite-t-20261015/production_weak_packaging_review/import_stderr.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/production_weak_packaging_review/run_cold.py > outputs/nmb-finite-t-20261015/production_weak_packaging_review/cold_stdout.txt 2> outputs/nmb-finite-t-20261015/production_weak_packaging_review/cold_stderr.txt
```

Actual wheel check3.06s; cold tests3.49s, all processes under120s. Three physical calls are exactly the existing base-control empty-radiation example; invalid guards reject before integration. No new model/phase/spectral/EOS experiment, installs/downloads/CI/git/PR/source fixes.

Fresh quota06:23:56UTC: exactaccount82f4cfd9-9e20-47cf-9997-761328d89f60,codex10080min28%used,reset1792652325. Root numerical acceptance explicitly preceded the refresh/build/imports. No resets.
