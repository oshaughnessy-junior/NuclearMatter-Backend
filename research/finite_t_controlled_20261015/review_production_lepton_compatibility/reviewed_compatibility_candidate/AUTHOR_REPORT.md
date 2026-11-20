# Repaired finite-temperature lepton candidate

**Author-frozen numerical repair; a different independent recheck is required.**
The five proposed NMB paths remain under `staged/`, based on current main
`ffb5cc72949f2e5401251847aef4c9c7a98dde78` (rechecked during repair). No actual
production checkout, branch, PR, commit, dependency or external environment
was changed. Canary workflow changes are preserved.

## Minimum-compatible log-occupation follow-up

A separate independent review accepted665 numerical/provenance checks and
reported1 STATIC COMPATIBILITY FAIL: scipy.special.log_expit was introduced
in1.8, while NMB declares scipy>=1.7. That review remains unchanged
(SHA5c5b3fafa8227191107cbaf4887fd1f9bf22b232a7f390da207c90aaf159a9b4).
Before this follow-up, all five current files, reports/manifests and evidence
were copied to `pre_scipy17_repair/` (75 pinned files), and all15 separate
review files were pinned. The previous snapshots also remain unchanged.

The only numerical-source edit removes that newer SciPy import and defines
private `_log_expit(x)=-np.logaddexp(0,-x)`. Public signatures, defaults,
physics, constants and dependencies are unchanged. Added helper tests cover
both signs, signed zero, tiny arguments, dilute tails and ±1e300 against
independent stable math.log1p formulas. **82 candidate tests plus13 unchanged
cold tests PASS (95 total)**. The bounded independent momentum/root rerun
has213 PASS. An isolated missing-export simulation reproduces the old import
failure and imports the repaired module successfully, without fake package
initializers or environment changes. It is not execution under SciPy1.7.
Additional algebra/numerical-preservation checks give259 PASS, including two
explicit matched rejection checks rather than physical forward acceptances.
All successfully compared state/thermal values are bitwise unchanged.

A new diagnostic encountered a pre-existing stable-thermal-increment guard
rejection at electron m/T=600,mu/T=±.53. Both preserved and new modules raise
`ArithmeticError: Unresolved finite-temperature lepton quadrature`; their
ordinary state fields agree bitwise (and the prior independent review's
new ±.53 dilute all-field Bessel/variance checks had maximum error9.86e−14).
At T=.0008516649166666665 MeV,mu=±.0004513824058333333 MeV,m=.51099895 MeV,
the cold kinetic surface x_c=2.4879181442607967e−174 makes the positive hole
integral unrepresentable: its estimated log magnitude is−990.73, below the
float64 subnormal log limit−744.44. quad_vec returns value/error0 but status1,
"Target precision not reached", with zero absolute tolerance; the candidate
then rejects conservatively. This is not a cancellation of the resolved total
thermal energy. That energy can be independently reconstructed by subtracting
the normal cold rest term m*abs(n_net) from the returned total energy; the
cold kinetic correction is unrepresented at this point. The diagnostic records
that reconstruction and the separate Bessel thermal expectation. No fallback
value is returned by the public thermal function. The original diagnostic failure and both
matched-rejection records are preserved. No thermal forward acceptance is
claimed at these two additional points. This compatibility-only change does
not fix that separate numerical limitation; any broader thermal-increment
claim must resolve it or explicitly exclude those states.

Both existing junior_tools and junior_tools_alt environments report the same
Python3.12.12, NumPy2.3.5, SciPy1.15.2, JAX0.10.1, Diffrax0.7.2 and pytest9.0.3.
Thus the alternate environment provides no new minimum-version execution or
NumPy<2.3 validation-pin qualification. The static missing import is repaired,
but a different independent compatibility/numerical recheck remains required.
Exact follow-up commands are in `evidence/COMPAT_COMMANDS.md`.

## Preserved failure and quadrature repair scope

The initial candidate, original manifests/report and all original evidence
were copied before edits to `pre_quadrature_repair/`; its31 files are pinned
by `PRE_QUADRATURE_REPAIR_PIN.json`. The initial independent review remains
**207 PASS,4 FORWARD FAIL**, unchanged in the sibling review directory; its20
files are pinned by `INITIAL_INDEPENDENT_REVIEW_PIN.json`. These historical
failures have not been relabeled as passes. A fresh unique import of the old
module reproduces all four density failures at electron m/T=40,100,300,600:
relative errors1.64083e−4,8.85698e−4,8.96872e−4,8.99699e−4.

The old dimensionful quadrature used an absolute floor that dominated these
small, ordinarily represented populations. The repair integrates dimensionless
kinetic energy x=(E−m)/T. Each particle/antiparticle population has its own
occupation scale; energy and response components use dimensionless scales.
Adaptive quadrature now has zero absolute tolerance. Logarithmic combination
of occupation and dimensional factors restores ordinary outputs without
premature amplitude underflow. This is the same FD integral, with no density
floor, fit, surrogate, classical fallback, changed defaults or rejected dilute
states. The thermodynamic free energy uses mu*n_net−P, avoiding subtraction
of rest-dominated e and T*s. The stable thermal increment sums independently
scaled positive holes, particle excitations and antiparticle energy.

The original scalar public API, LeptonState fields, NMB M_E/M_MU/HBARC,
units, rest-inclusive energy and independent flavor policy remain unchanged.
Net number counts negative leptons minus positive antiparticles; electric
charge is −e*n_net. No neutrality or beta equilibrium is imposed. Existing
cold APIs/exports, EOSTable, cooling, JAX and Hessian implementations are
unchanged. This remains an explicitly imported CPU reference.

## Quadrature repair checks preceding compatibility follow-up

* **81 candidate tests PASS**, zero errors/skips, including the original60.
  Added12 scaled-Bessel cases test every thermodynamic/response field at
  m/T=40,100,300,600 and mu/T=0,+0.4,−0.4, including tail60→90 and
  rtol2e−12→5e−13. The four old symmetric density failures now agree with
  the Bessel result to1.43e−14 or better. Their individual densities range
  from3.88e−29 through3.99e−274 fm−3, all normal float64 populations.
* Added9 independent massless analytic cases at T=1e−30,1e−9,1e6 MeV and
  mu/T=0,+0.37,−0.37. Closed forms check all thermodynamic/response fields;
  separate scalar original-momentum integrals check both populations, and
  the analytic cold subtraction checks the thermal increment. Comparisons
  have zero absolute assertion tolerance.
* **213 independent momentum/root checks PASS**: original-momentum Gauss
  quadrature384→512 with tail80→100 covers all11 output fields at19 states:
  the12 dilute cases, four different warm/degenerate e/muon states, and
  three simultaneous dimensionful T/mass scalings (m/T=.13,mu/T=.37).
  Maximum candidate/reference relative discrepancy9.68e−13; maximum
  reference-order/tail discrepancy1.14e−12; maximum source tail/rtol change
  3.45e−15. Four additional physical massive roots at ±3e−29 fm−3 pass
  reconstruction and independent pair-susceptibility linear limits.
* **13 unchanged cold derivative/JIT/Hessian/unit tests PASS** after the
  final source change. The example runs with finite JSON and empty stderr.
  All71 pinned baseline package Git blobs and the separate candidate overlay
  verify. Real package initializers/imports are used in fresh processes.
  The three candidate Python files still parse with Python3.9 grammar.

The repair evidence retains two diagnosed probe issues. The first new Bessel
run had79 PASS/2 FAIL because its oracle squared an approximately1e−274
moment before dividing, underflowing the oracle covariance. Algebraic
reassociation first*(first/total) fixed the oracle without tolerance changes.
The first momentum run had203 PASS/10 FAIL from unresolved128→192 Gauss
nodes at the single degenerate muon state;384→512 resolves it. Their scripts,
raw results and failures remain. Neither issue was hidden or used to loosen
a gate. Final raw commands/results are in `evidence/REPAIR_COMMANDS.md`.

## Numerical and integration limits

Internally scaled quantities do not create a public log-density API or a
relative-precision certificate for subnormal/unrepresented outputs. An
ordinary antiparticle output can still underflow to zero without meaning
physical absence. Unresolved susceptibility is rejected. The relative
quadrature control is not a uniform error bound for cancellation-conditioned
derived quantities, and no uniform extreme-domain rectangle is claimed.
The bounded checks qualify selected ideal-gas forward/thermodynamic response
benchmarks, not differentiation through adaptive quadrature, physical
uncertainty, a complete matter EOS or cooling inventory.

Tests used the existing Python3.12.12/NumPy2.3.5/SciPy1.15.2/JAX0.10.1/
Diffrax0.7.2 environment, with no installs. The package's supported/minimum
version execution, conda/pixi NumPy<2.3 environment, packaging and full
repository/admitted CI remain open. Syntax checks are not minimum-version
execution. A **different independent scientific/numerical/API recheck** of
this frozen repair is required before any integration claim. Only after that
may the parent publish exactly the five manifested paths in a scoped draft
PR, preserving current main; mirrors, snapshots and campaign evidence are
not production files.
