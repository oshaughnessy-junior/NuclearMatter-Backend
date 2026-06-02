# Deploying a validation environment

The core package needs only `numpy scipy jax diffrax` and runs its full test suite
(incl. the in-process SciPy radius-domain cross-check) anywhere.  The **external**
references — RePrimAnd and Zhu's δ-TOV — need a heavier environment that this
Cowork sandbox can't build (C++/Boost/GSL, no conda, tight disk), so deploy one of
the following on your workstation and run the cross-checks there.

## Option A — pixi (mirrors RIFT's setup)

```bash
cd NuclearMatter-Backend/env
pixi install
pixi run tests        # 20 core tests (diffrax vs SciPy reference incl.)
pixi run validate     # scripts/validate_against_reprimand.py
```

## Option B — conda/mamba

```bash
mamba env create -f NuclearMatter-Backend/env/environment.yml
conda activate nmbackend-validate
pip install -e NuclearMatter-Backend
cd NuclearMatter-Backend
PYTHONPATH=src pytest
PYTHONPATH=src python scripts/validate_against_reprimand.py
```

## RePrimAnd (PyPI distribution `reprimand`, import module `pyreprimand`)

The PyPI **distribution is named `reprimand`** (not `pyreprimand`); the importable
**module is `pyreprimand`**.  Using the wrong name is why `pip install pyreprimand`
fails with "No matching distribution found".

1. **Linux (easiest)** — a binary wheel exists, so just:
   ```bash
   pip install reprimand
   python -c "import pyreprimand; print(pyreprimand.__file__)"
   ```
   The `env/` files already list `reprimand` for the pip/PyPI step.

2. **macOS / HPC (no wheel)** — build from source (the `env/*` files provide the
   build deps: meson/ninja/boost-cpp/gsl/hdf5/pybind11/compiler):
   ```bash
   git clone https://github.com/wokast/RePrimAnd && cd RePrimAnd
   # C++ library first (in a conda env, point meson at it):
   BOOST_ROOT=$CONDA_PREFIX meson setup --prefix=$CONDA_PREFIX mbuild --buildtype=release
   ninja -C mbuild && ninja -C mbuild install
   # then the Python bindings:
   cd bindings/python && pip install .
   python -c "import pyreprimand; print('ok')"
   ```
   HDF5 tip from the README: if meson/hdf5 misbehave in conda, prefix
   `LD_LIBRARY_PATH` with `$CONDA_PREFIX/lib`.  Reference: RePrimAnd README
   (https://github.com/wokast/RePrimAnd), docs (https://wokast.github.io/RePrimAnd/),
   Zenodo 10617510 (release 1.6/1.7).

3. **Reuse what you already have.**  If `junior_tools` (or your RIFT env) already
   provides a working `pyreprimand`, just run the validation script there.  *(I
   couldn't see `junior_tools` from the sandbox; point me at it / paste
   `pydoc pyreprimand` and I'll match the reference adapter to your exact version.)*

> **API drift:** RePrimAnd's Python API changed across releases and **RIFT's
> `EOSManager.EOSReprimand` targets an old one**.  `src/nmbackend/reference/
> reprimand.py` mirrors that old interface with the version-sensitive calls marked
> `# VERIFY`.  After installing, run the validation script; if it errors, adjust
> those lines to your installed version (`help(pyreprimand.make_eos_barotr_spline)`)
> — or hand me the output of `pydoc pyreprimand` and I'll update the adapter.

## Zhu δ-TOV (sensitivity cross-check)

To corroborate the **AD sensitivities** (`dΛ/dparam`) against an independent
perturbation solver:

```bash
git clone https://github.com/zzhu-astro/delta_tt    # Z. Zhu's C++ δ-TOV
# build per its README (RK4 TOV + perturbation; ctypes Python wrapper)
```
Then compare `jax.jacfwd` of `solve_structure(...).Lambda` to `delta_tt`'s
`get_star_delta_Lambda` on a matched RMF EOS.  (Motivated-by until licensed; see
`ATTRIBUTION.md`.)

## What runs where

| Check | Sandbox (here) | Your env |
|---|---|---|
| Core unit/physics/AD tests | ✅ `pytest` | ✅ |
| diffrax vs SciPy radius-domain reference | ✅ (M,R,Λ to <0.1%) | ✅ |
| RePrimAnd M/R/Λ cross-check | ❌ (no build) | ✅ via `scripts/validate_against_reprimand.py` |
| Zhu δ-TOV sensitivity cross-check | ❌ | ✅ |
