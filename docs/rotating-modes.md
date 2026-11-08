# Rotating-mode framework and implementation ladder

Motivated by [Gao & Shibata, arXiv:2611.03986v1](https://arxiv.org/html/2611.03986v1).
This package supplies a working eigenproblem kernel, an RNS input contract and
Newtonian rotating-fluid benchmarks. **It does not yet assemble or solve the
paper's coupled full-GR rotating-star mode problem.**

## Implemented

`nmbackend.modes.Pencil` represents K0 + sigma K1 + sigma² K2. `solve` uses a
generalized companion eigenproblem, so a singular K2 is permitted and its
infinite roots are reported. Indeterminate 0/0 roots and failed polynomial
left/right residuals are rejected. The dense kernel has an explicit dimension
limit of 256; this is a qualification kernel, not a full-scale GR solver.

`select_simple` and `directional_derivative` use physical left/right
polynomial eigenvectors. `modes.autodiff.simple_root` supplies the implicit
JAX JVP, which JAX transposes for reverse mode. It is local to a fixed simple
branch and rejects repeated or ill-conditioned roots. It does not differentiate
sorting, an exceptional point, background generation or missing GR matrices.
Second derivatives through the host callback are unsupported.

Branch continuation uses eigenfunction overlap, an explicitly positive norm
metric and frequency continuity. Unresolved matches fail. Near degeneracy,
subspace continuation and branch-specific physical interpretation remain work.

The convention is exp(i*(sigma*t + m*phi)). Positive Im(sigma) damps;
negative Im(sigma) grows. omega_ro = Re(sigma) + m*Omega. E-folding time is
1/abs(Im(sigma)), without a stray 2*pi. Neutral-pattern diagnostics distinguish
inertial and corotating stationarity; canonical-energy signs and actual
radiative growth are separate quantities. `exterior_contour` checks that
Im(sigma*exp(i*angle)) < 0 for outgoing exp(-i*sigma*r). It is a contour helper,
not an assembled outgoing GR boundary condition.

`rns_polytropic_background` accepts the conventional RNS export and supplies
the exact barotropic dp/de. Generic `ModeBackground` requires an explicitly
specified perturbative sound speed and closure provenance. Frozen composition
must be provided separately; finite-reaction response cannot be replaced by
a scalar constant. An unqualified learned equilibrium is not accepted by this
RNS adapter. Preparing input exchange does not certify mode-level accuracy.

## Executable physical benchmarks

1. **Newtonian slow-rotation l=m r-mode.** Numerical Gauss integration
   assembles the toroidal mass/Coriolis projection rather than inserting its
   frequency. The result recovers omega_ro/Omega = 2m/[l(l+1)]. No relativistic,
   finite-spin, viscous or radiative correction is included. The radial
   leading-order r^m profile factors out of this projection; it supplies no
   EOS-dependent neutron-star spectrum.
2. **Finite-rotation Newtonian Maclaurin m=2 bar modes.** Numerical ellipsoidal
   potential integrals assemble the classical virial polynomial. The spherical
   limit, free-surface branches and conservative dynamical growth are checked.
   The neutral-pattern and dynamical thresholds recover beta about 0.1375 and
   0.2738. Radiation and viscosity are absent; no secular growth rate is inferred
   from a neutral pattern. This homogeneous incompressible benchmark is not a
   causal neutron-star EOS or the paper's relativistic strange-star sequence.

The Maclaurin benchmark follows the classical equations recorded in
[Baiotti et al., Eqs. 7.1–7.3](https://arxiv.org/pdf/astro-ph/0609473) and
[Braviner & Ogilvie, Sect. 5](https://arxiv.org/html/1404.2461). Their time
conventions are converted explicitly. The latter's harmonic potential degree
and the r-mode's axial-vector harmonic degree should not be conflated.

```
PYTHONPATH=src JAX_ENABLE_X64=true python -m pytest tests/test_modes.py -ra
PYTHONPATH=src python examples/rotating_modes_reference.py modes.json \
  --rns-csv reference/run/rns_rotating.csv
```

The optional RNS argument exercises geometry/closure exchange, not a computed
relativistic spectrum. Physical tests and the algebraic avoided-crossing
examples have distinct roles. The two-state formula in the new paper is useful
for checking root/rate exchange, not a substitute for stellar perturbations.
No new mandatory dependency was added.

## What remains for the Gao–Shibata calculation

The paper specifies the second-order Einstein-fluid action, Lorenz gauge,
spectral Ritz–Galerkin quadratic problem, free-surface conditions and exterior
complex scaling. Its detailed action formulation is deferred to a companion
paper listed as in preparation. We can independently derive it using the
standard perturbation equations and variational identities; upstream software
is not a prerequisite. The current implementation does not contain that
thirteen-field physical matrix assembler.

The required path is:

1. **Background admission.** Freeze EOS representation, density convention,
   baryon mass, spin, units, composition closure and accuracy receipts. Validate
   metric/matter derivatives, surface and exterior multipoles under resolution
   refinement. The existing RNS output is useful infrastructure; mass/radius
   agreement alone is insufficient for small imaginary mode frequencies.
   The present PINO pilot remains below the required accuracy.
2. **Spherical full-GR physical action and sparse assembly.** Implement ten
   metric and three displacement components, the baryon-conserving density
   variation, gauge-fixing and surface contributions. Recover established
   spherical fluid/spacetime references and test gauge/constraint residuals.
   Use regular bases at the centre/axis and match surface traction and metric
   perturbations. Retain self-bound density-jump terms for MIT-type matter.
   A Cowling rung is separately labelled and does not establish full-GR parity.
3. **Outgoing exterior.** Construct a justified analytic continuation of the
   vacuum background (e.g. a qualified multipole representation). Interpolating
   real RNS grid values at complex coordinates does not define it. Assemble
   the complex Jacobian and derivative transformations. Physical roots must
   converge under exterior radius, scaling angle and domain-order changes.
4. **Uniformly rotating continuation.** Continue selected f/r/inertial branches
   with angular-sector coupling, surface-fitted domains and mode identities.
   Use fixed-baryon-mass and fixed-J sequences where the physical test requires
   them. Preserve avoided crossings, conjugate sectors and canonical energies.
5. **Physical stability and sensitivities.** Distinguish zero inertial frequency,
   zero corotating frequency, conservative mode collision and the axisymmetric
   neutral operator. The last preserves each fluid element's specific angular
   momentum; an equilibrium turning point is not that operator. Qualify
   complex derivatives with fresh constraint-resolved forward solves only
   after the corresponding spectrum passes its forward gates. At a multiple
   root, differentiate a resolved subspace rather than a scalar label.

Reference targets from the paper require matched background and EOS smoothing,
not just the same EOS name: its APR4 nonrotating star has baryon mass 2 solar
masses, gravitational mass 1.743, radius 11.11 km and sigma/(2*pi) about
(+/-2.196 + 0.001171 i) kHz. These are published targets, not our results.
The mode engine must recover radiative damping independently; a fitted time
or frequency relation does not close this gate. Differential rotation, hot
matter, pairing/entrainment, nonlinear saturation and g-mode composition
physics are further extensions.

## Compute and proposal

Use infra-atlas for placement. Small dense qualification problems run on a
few cardassia cores. EOS/spin/m/parity and convergence cases can be distributed
through OSPool; larger sparse assembly and solves belong in scheduled SPORC
or other admitted resources. Re-measure memory and runtime before scaling.
Thirteen fields on a 64x16 grid already give about 13,000 coefficients; three
dense complex matrices alone take roughly 8.5 GB before linearization.
Do not extrapolate the 512 MB RNS job profile to that calculation. A future
PETSc/SLEPc route would be optional and needs explicit dependency/environment
pins; none is introduced here.

The proposal can adopt this action/eigenproblem architecture and credit the
working numerical foundation and Newtonian benchmarks. Rapidly rotating
full-GR spectra, damping rates and EOS sensitivities remain proposed validated
work. The planned contribution is connecting those modes to shared matter
inputs and multimessenger inference, with physical and numerical discrepancy;
this paper is prior art for the full-GR mode formulation.
