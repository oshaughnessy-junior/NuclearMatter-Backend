# Repaired finite-temperature lepton candidate

**Author-frozen numerical repair; a different independent recheck is required.**
The five proposed NMB paths remain under `staged/`, based on current main
`ffb5cc72949f2e5401251847aef4c9c7a98dde78` (rechecked during repair). No actual
production checkout, branch, PR, commit, dependency or external environment
was changed. Canary workflow changes are preserved.

## Preserved failure and repair scope

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

## Executed checks

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
