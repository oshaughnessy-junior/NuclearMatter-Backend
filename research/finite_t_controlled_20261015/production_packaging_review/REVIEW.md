# Independent packaging and scope audit

**PASS: wheel build, exact module inclusion and real wheel imports succeed in the existing environment.** There are19 passing wheel/import/example/scope gates and no remaining packaging blocker. This is not a clean dependency installation, minimum-runtime test or full repository CI result. No source/author checkout, dependency, git branch/PR or paper artifact was changed.

Candidate source baeef78b174ecc0bcc1ab6016c839a8e8dc6a2ba915541d7fd994d994f01e719 and manifest5e9a2793bc0de9c2e2a01c78061bd7a76f093b260cb8a7564971aae18a1bf282 were checked before preparation and remain unchanged. Exactly the five authored paths are overlaid in an owned build copy: README, finite-temperature docs, example, module and tests. All71 baseline package source/data blobs are verified against preserved owning-GitHub head records for ffb5cc72949f2e5401251847aef4c9c7a98dde78. The head pyproject blob is independently checked. LICENSE comes from a read-only git show of the pinned clean local base; the preserved complete base-to-head comparison has no LICENSE change. PREPARATION.json records that reconstruction. This is the pinned package/metadata plus candidate, not a modified real checkout or fake namespace package.

## Build and package discovery

The project already declares setuptools>=61 and wheel with setuptools.build_meta. Existing setuptools82.0.1 and wheel0.47.0 are available; the optional build frontend is absent. Calling the existing backend directly avoids installing a frontend or using build isolation. No installer or dependency fetch is invoked. Its src package discovery includes nmbackend.reference, with actual package initializers and the new finite_t_leptons.py. No extra package-discovery/export change is needed for the documented direct module import.

The build produces nmbackend-0.0.1-py3-none-any.whl, SHA2568baeb76730f8c0d3d144232529c14f572fb849751c6abc406b6b1a7e7341494c. The module inside the ZIP has exactly the accepted source bytes. Wheel metadata retains numpy>=1.21 and scipy>=1.7 and the pre-existing JAX/Diffrax dependencies. Source docs/example/test files remain repository artifacts; this wheel audit does not assert those non-package paths are installed.

A fresh Python-I process prepends only the actual wheel file to sys.path. There are no source-path aliases, fabricated modules, site patches or installation. It imports the real nmbackend initializer, constants, reference module and existing cold nmbackend.eos.leptons.lepton_sector from the wheel. File origins are asserted inside that exact ZIP. A fresh signed net-number root, finite positive thermal increment, immutable state and zero-T rejection pass. The existing cold function runs with positive outputs and its original charge closure. The source example runs against wheel imports and emits finite JSON with distinct electron/muon chemical potentials and explicit units/signs. These tests exercise package-relative imports in addition to file inclusion.

## Documentation and scope

The module docstring, README, API note and example consistently describe scalar CPU NumPy/SciPy ideal lepton-pair components, not JAX AD/JIT/vmap or a complete matter EOS. Separate flavor chemical potentials/net-number inputs impose neither neutrality nor weak/inter-flavor equilibrium. Net number is negative leptons minus positive antiparticles, electric charge−e*n_net; zero net flavor retains pairs. T, mass and total mu are MeV; number/entropy/Cv are fm^-3 (kB=1); rest-inclusive energy/pressure/free energy are MeV fm^-3. Number responses carry inverse MeV and dmu/dT is dimensionless.

The cold API distinction is explicit: existing cold mu is fm^-1 and energy fm^-4, converted by HBARC. Its inter-flavor equilibrium differs from the new independent components, and its source/API is unchanged. Thermal energy is the same-flavor fixed-net-number increment, includes pair/rest accounting consistently and requires no extra Q or rest source. Invalid inputs, unresolved quadrature and susceptibility underflow remain rejected. Component rounding policy and lack of relative subnormal/uniform-domain certification are stated. The example correctly labels this ideal CPU component model and independent number inputs.

## Evidence, warnings and open gates

The first audit result had18 PASS and one **audit-fixture** failure: it incorrectly required every state field to be builtin Python float. The API promises CPU scalar NumPy/SciPy fields; legitimate NumPy floating scalars satisfy that scope. Exact initial script/report/output and initial_fixture_failure.json are preserved. The corrected scalar-real/non-array/non-JAX requirement passes, without changing candidate code or a scientific gate. Build/import/example success was already present before that correction.

The backend emits an inherited project.license TOML-table deprecation warning and expected disabled-bytecode warnings. They do not block this build and are not changes from this five-file candidate. No unrelated metadata fix is made.

Runtime is existing Python3.12/NumPy2.3.5/SciPy1.15.2/JAX0.10.1. NumPy2.3.5 is outside the separately pinned validation environment's<2.3 range, even though the project metadata itself only declares a lower NumPy bound. No supported/minimum-version or broad-platform claim follows. Clean installation/dependency resolution, minimum/supported-version execution, admitted full repository CI and packaging on other platforms remain open. This audit closes only the selected wheel construction/inclusion/current-environment import gate. No full EOS, weak kernel, thermal evolution, empirical uncertainty or derivative qualification is added.

## Exact bounded commands

From Genesis:

```
/Users/rossma/miniconda3/envs/junior_tools/bin/python -B outputs/nmb-finite-t-20261015/production_packaging_review/prepare.py
```

From the owned production_packaging_review/package_copy:

```
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -B -c 'from setuptools.build_meta import build_wheel; print(build_wheel("../wheels"))'
```

Fresh import from Genesis:

```
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/production_packaging_review/check_wheel.py
```

Build/import raw outputs, wheel metadata, source reconstruction and example JSON are retained in this owned review directory. The wheel and build copy are local audit artifacts and must not be added to the papers repository or five-path production change. The prepare step deliberately refuses overwriting the owned copy. No action was taken on the root's publication metadata; the transient move mentioned by the root did not affect any candidate hash observed by this audit.
