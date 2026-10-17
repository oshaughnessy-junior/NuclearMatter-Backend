# Scoped finite-temperature lepton candidate

**Author-frozen staged candidate; different independent review required.** The
five proposed files are under `staged/`, preserving their exact NMB paths.
The base is current main `ffb5cc72949f2e5401251847aef4c9c7a98dde78`, rechecked
before authoring and at final freeze. No actual checkout/source, remote branch,
PR, commit, environment or existing campaign leaf was modified.

The candidate adds an installed, explicitly imported CPU ideal relativistic
lepton/antilepton state utility in `nmbackend.reference.finite_t_leptons`.
It ports the accepted reference's actual adaptive FD integrals, stable
pair-symmetric number subtraction, susceptibility moments, scaled number root
and thermal increment. It uses exact existing NMB HBARC/M_E/M_MU constants.
No frozen campaign file, review acceptance, weak rate, table or surrogate is a
runtime dependency. Existing cold APIs, exports, EOSTable, cooling, JAX and
Hessian behavior are untouched.

Inputs are temperature and total chemical potential/mass in MeV, or a signed
net lepton NUMBER density in fm−3. Each electron/muon flavor is independent;
net number counts negative leptons minus positive antiparticles, so electric
charge is −e*n_net. Neither neutrality nor inter-flavor weak equilibrium is
imposed. Pair chemical potentials are opposite within a flavor. Total energy
includes both populations' rest energy; no extra reaction-Q/rest source is
added. Named fields, fixed-net-number responses, units and exclusions are in
the module/doc. This is a CPU reference, with no claim for JAX AD/JIT/vmap,
interacting matter, a broad cold rectangle or a complete cooling inventory.

Production admission is stricter than the prototype's float conversion:
finite real non-Boolean scalars, T>0, m>=0, finite rtol/tail bounds, finite
quadrature/responses, positive resolved susceptibility and 1e−9 relative
net-number reconstruction are enforced. Exactly zero retains thermal pairs;
no positive density floor is applied. Documented susceptibility underflow
raises instead of inventing an absent pair gas. Ordinary underflow of an
individual antiparticle population is retained and explicitly does not imply
physical absence or relative tiny-population accuracy.

## Executed checks

The final candidate test suite has **60 PASS, zero failures/errors/skips** in
4.29 s. Tests include exact massless formulae, an independent scalar
infinite-momentum oracle for electrons/muons/both signs, massive cold and
Sommerfeld limits, Maxwell–Boltzmann behavior, signed/net-zero/tiny-number
roots, fixed-net-number/Maxwell response checks after forward gates, selected
tail/tolerance refinement, strict input/underflow checks, immutable state
behavior and direct comparisons with the current cold lepton API after the
explicit fm→MeV unit conversion. The earlier 58-test run passed before the
two direct cold-interface checks were added; both raw records remain.

Exact unchanged current cold Hessian/vector/JIT and units tests separately
have **13 PASS** in3.32 s. The example runs with finite JSON and empty stderr.
All experiments took under120s. No tolerance was relaxed and no numerical
failure was hidden. Read-only discovery diagnostics are retained separately
and did not execute scientific tests.

The real installed NMB checkout found through the existing `junior_tools`
environment was clean but older (`5e0208cccf386c723f10955d0c81390fa1ecb972`).
It was read only. The owned test mirror combines its unchanged tracked files
with exact owning-GitHub contents for every package source changed through
ffb5cc. **All71 baseline package Git blobs verify**, then the candidate module
is overlaid separately. Actual existing package initializers and imports are
preserved;24 imported package modules were verified inside the owned mirror.
No namespace stubs or fake JAX were used. The broad mirror is test evidence,
not part of the five-file candidate diff.

Environment: Python3.12.12, NumPy2.3.5, SciPy1.15.2, JAX0.10.1,
Diffrax0.7.2, pytest9.0.3, from an existing environment; no installs. This
satisfies pyproject's runtime lower bounds but differs from the conda/pixi
validation environment's NumPy<2.3 constraint. The staged Python files pass
Python3.9 grammar parsing and avoid newer conveniences such as dataclass
slots and np.trapezoid. That is not execution on package minimum versions.

## Remaining gates

1. A **different independent scientific/numerical/API review** of the frozen
   candidate and its test evidence before any integration claim.
2. Parent-owned draft PR publication of exactly the five manifested staged
   paths, preserving current main/canary updates. No test mirror/evidence
   path belongs in the production PR.
3. Normal admitted repository CI/full-suite and supported-version checks,
   including minimum-version compatibility or an explicit dependency-policy
   decision. No remote CI, wheel/build/install or complete repository suite
   was run here. Existing package imports/cold tests were exercised, not
   replaced by such a claim.

`CANDIDATE_MANIFEST.json` hashes precisely the five proposed files.
`baseline/README.md` and `evidence/README.diff` preserve the narrow README
change. `evidence/COMMANDS.md`, JUnit, raw stdout/stderr, environment/import
record and source readbacks contain the exact reproduction/provenance.
`CHECKPOINT.json` hashes the frozen candidate/evidence; root campaign
publication and final independent disposition remain separate.
