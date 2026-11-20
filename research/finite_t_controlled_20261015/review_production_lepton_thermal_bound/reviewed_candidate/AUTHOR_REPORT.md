# Finite-temperature lepton candidate: bounded thermal repair

**Author-frozen; different independent review required before a draft PR.**
Exactly five NMB paths remain under staged/, based on pinned main
ffb5cc72949f2e5401251847aef4c9c7a98dde78. No production checkout, remote branch,
PR, commit, environment or dependency was modified.

## Latest repair and proof

The separate compatibility review accepted140 checks but retained2 FUNCTIONAL
FAILURES: thermal_energy_density rejected returned electron states at
T=.0008516649166666665 MeV,m=.51099895 MeV,mu=±.0004513824058333333 MeV
(m/T600,mu/T±.53). The total thermal increment is normally represented near
2.41148076418e−274 MeV fm^-3. The failure came from a positive cold-hole
component too small to represent, not cancellation or underflow of total heat.
That review and every prior failure remain unchanged.

This source change only adds a conservative dimensional upper bound for that
positive hole and skips it when its entire physical contribution provably
rounds to zero. Monotonic geometry and vacancy<=1 give

H <= T^4/[pi² HC³] * sqrt[xc(xc+2v)] * (v+xc) * xc²/2.

The bound has MeV fm^-3 units. frexp bounds every positive factor by an exact
power of two; integer/half-integer binary-log arithmetic produces a strict
upper bound. Omission requires that bound be at or below2^-1075, half the
smallest float64 subnormal. There is no approximate-log threshold comparison,
arbitrary density/T floor, total-heat zeroing, blind quadrature-failure
suppression or changed tolerance. Otherwise the existing hole integration
and unresolved-status guards are unchanged. Other thermal components and all
state_mu/state_density algorithms are unchanged. The complete proof is
`evidence/HOLE_BOUND_PROOF.md`.

## Executed verification

**98 tests PASS**, zero errors/skips, in8.73s:85 candidate tests (the preceding82
plus two exact failure regressions and one bound/retained-hole control), and
13 unchanged cold derivative/JIT/Hessian/unit tests. The old81 candidate
forward/response/limit checks remain included. New exact failure regressions
compare total thermal heat against independent scaled-Bessel moments and
resolved bulk-minus-cold-rest reconstruction with the existing3e−10 gate.
Near-threshold dyadic controls test both sides of the rounding gate; an
independent exact polynomial integral checks a positive represented hole;
unresolved represented quadrature still raises.

**35 selected preservation/regression checks PASS.** Sixteen old/new states
have every returned field bitwise identical. Fourteen previously successful
thermal increments and three additional retained-hole controls are bitwise
identical. The two original exceptions are reproduced from a fresh unique
import of the saved candidate, while the new function completes and agrees
with Bessel without loosening its gate. Old exceptions and current physical
values/bounds are recorded separately in hole_preservation_results.json.
This bounded task reruns no new physics grid; earlier independent moment
acceptances remain provenance, not additional current check counts.

All experiments are below120s in the existing Python3.12.12/NumPy2.3.5/
SciPy1.15.2/JAX0.10.1 environment. Exact commands and raw stdout/stderr/JUnit
are in evidence/HOLE_COMMANDS.md. The71 exact baseline package blobs,
candidate mirror identity and Python3.9 grammar are verified. Real package
initializers/imports are used, with no namespace stubs or installs.

## Preserved history and unchanged scope

Before this edit, the complete five-file candidate, current reports/manifests
and evidence were copied into pre_hole_bound_repair/ and pinned. All previous
snapshots remain: the original quadrature candidate, compatibility candidate,
all raw failures and review copies. Independently frozen reviews still record
207 PASS/4 FORWARD FAIL,665 PASS/1 STATIC COMPATIBILITY FAIL, and140 PASS/
2 FUNCTIONAL FAIL. Preservation pin sets verify their bytes unchanged.
The prior absolute-floor repair and minimum-compatible logaddexp helper are
retained; no scipy.special.log_expit dependency has been restored.

Public signatures/defaults, NMB M_E/M_MU/HBARC, independent flavor policy,
finite-real/non-Boolean admission and root-residual guard are unchanged.
Net number counts negatively charged leptons minus positive antiparticles;
electric charge is−e*n_net. No neutrality or beta equilibrium is imposed.
Energy is rest-inclusive; no additional mass/Q term belongs in its accounting.
Existing cold APIs, exports, EOSTable, cooling, JAX and Hessian code are
untouched. This remains a scalar CPU reference, not a complete EOS or AD port.

No public log-density API, relative-subnormal guarantee, uniform extreme
rectangle, empirical uncertainty or broad thermal-domain qualification is
claimed. Different independent bound/functional-domain review remains the
next gate. Minimum/supported-version execution, NumPy<2.3 validation-pin
execution, admitted full repository CI and packaging remain open. Both
available existing environments have identical versions; no minimum-version
runtime claim follows from syntax/static API checks. Only after review may
the parent publish the exact five manifested paths in a scoped draft PR;
mirrors, snapshots and evidence are not production files.
