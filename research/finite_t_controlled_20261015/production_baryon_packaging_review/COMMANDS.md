Executed from /Users/rossma/Genesis_LANL, except direct build in owned package_copy:

```
/Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_baryon_packaging_review/prepare.py
```

```
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -c 'import setuptools.build_meta as b; print(b.build_wheel("../wheels"))' > ../build_stdout.txt 2> ../build_stderr.txt
```

```
OPENBLAS_NUM_THREADS=1 JAX_ENABLE_X64=true /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/production_baryon_packaging_review/check_wheel.py > outputs/nmb-finite-t-20261015/production_baryon_packaging_review/import_stdout.txt 2> outputs/nmb-finite-t-20261015/production_baryon_packaging_review/import_stderr.txt
```

Last command first ran with the preserved initial fixture, then with the explained package-data correction. All executions<120s. No installs/network/source fixes. Exact quota03:26:06UTC25%, baseline account/window/reset matched; deadline10:27:28UTC; noreset.
