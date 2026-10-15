# Recoil and Pauli blocking: a finite-T density-transition response

Research checkpoint, 15 October 2026. This directory owns an independent reference building block; it edits no earlier prototype, GM1 implementation or production NMB source. Its purpose is to replace static-target phase space with a controlled dynamical response before constructing any optical kernel.

**Delivered model:** a spin-summed scalar/density transition between two free nonrelativistic quasiparticle species with a common kinetic mass, arbitrary prescribed species offsets, and independent chemical potentials. It retains exact thermal phase-space support, nucleon recoil and nucleon Pauli blocking **within that model**. It is not a physical relativistic vector/axial weak response, a GM1-consistent opacity or a full Urca rate.

## Literature anchor and the boundary of reuse

[Reddy, Prakash and Lattimer, Phys. Rev. D 58, 013009 (1998)](https://doi.org/10.1103/PhysRevD.58.013009), [arXiv:astro-ph/9710115](https://arxiv.org/pdf/astro-ph/9710115), Eqs.16–22 derive the nonrelativistic isospin density structure function at arbitrary degeneracy. Its delta-function phase-space normalization and equal-mass analytic reduction provide the reference for this checkpoint. No external implementation was executed or copied. The more general [Roberts and Reddy, arXiv:1612.02764](https://arxiv.org/abs/1612.02764) treats physical charged currents, relativistic kinematics and additional weak-current terms; those capabilities are not established by the scalar calculation here. Sources inspected on 15 October 2026.

## Definition, units and normalization

Use MeV for momenta, energies, chemical potentials and `T=kBT`, fm for lengths. Let

```
epsilon_a(p) = O_a + p²/(2M)
epsilon_b(p) = O_b + p²/(2M)
f_i(p) = [1 + exp((epsilon_i(p)−mu_i)/T)]⁻¹
```

`O_i` is a specified total rest-energy plus vector offset. It is not inferred from the kinetic mass. `mu_i` includes the same energy zero as `O_i`. Both species have the same two spin states. Matrix elements are set to one in spin space: each initial spin transitions to its corresponding final spin, giving an overall factor two, rather than four unrelated spin combinations.

We define positive energy transfer **into the nucleons**, with `p_b=p_a+q`, and

```
S_ab(q,omega) = 2 int d³p/(2 pi hbar c)³
                   × 2 pi delta(omega+epsilon_a(p)−epsilon_b(p+q))
                   × f_a(p) [1−f_b(p+q)].
```

The delta function is an energy delta and therefore has units MeV⁻¹. `S` has units fm⁻³ MeV⁻¹. No `hbar⁻¹`, weak coupling, lepton tensor, angular weak factor or neutrino density of states has yet been inserted. Consequently **S is neither an inverse mean free path nor a rate in s⁻¹**. The explicit `2 pi` belongs to the Fourier convention; integrated sum rules below use `domega/(2 pi)`.

## Reduction derived for this ansatz

Define `Delta=O_b−O_a`, `w=omega−Delta`, and kinetic chemical potentials `nu_i=mu_i−O_i`. Orient q along z. The energy delta fixes

```
p_z = M w/q − q/2,
E_min = p_z²/(2M),
z = (omega+mu_a−mu_b)/T = (w+nu_a−nu_b)/T.
```

Its Jacobian is `M/q`. Integrating the transverse azimuth and writing the initial kinetic energy as E gives

```
S_ab = M²/(pi q (hbar c)³)
       × int_(E_min)^infinity dE
          f[(E−nu_a)/T] {1−f[(E+w−nu_b)/T]}.
```

Here `f(x)=1/(1+exp(x))`. With `a=(nu_a−E_min)/T`, the remaining exact integral is

```
I = T [softplus(a)−softplus(a−z)] / [1−exp(−z)].
```

This expression includes Pauli blocking in the final species, independently of its abundance. The equal-kinetic-mass restriction is essential: substituting two different masses into this expression is invalid. The offsets can differ without invalidating the reduction. Density-dependent offsets used as frozen quasiparticle inputs do not establish thermodynamic consistency, residual interactions or conserving vertices.

## Equilibrium and support

Reversing the species and transferring `−omega` gives

```
S_ba(q,−omega) = exp[−(omega+mu_a−mu_b)/T] S_ab(q,omega).
```

This KMS/detailed-balance relation includes chemical imbalance and is not restricted to beta-equilibrated matter. It is obtained directly from the FD occupation/vacancy ratio and a change of integration variable; it does not introduce neutrino chemical equilibrium by assumption.

For finite positive T and q, the unbounded quadratic dispersion admits a nonzero thermal response at every finite real omega. Negative energy transfer is allowed. Large `E_min` suppresses the response exponentially; it must not be replaced by the cold support test. The log interface retains such tails even when the ordinary float64 value underflows.

At zero temperature the integral reduces to the interval length

```
I(T=0) = max[0, nu_a−max(E_min, nu_b−w)].
```

This gives exact recoil and Pauli support for the declared free nonrelativistic model. For `q=0`, the response is proportional to `delta(omega−Delta)` and is distributional. The scalar evaluator rejects q=0, rather than assigning it zero or smoothing its delta peak.

**Relativistic caution:** all-real-omega support is a property of an unbounded nonrelativistic dispersion. Relativistic equal-mass particle-hole kinematics impose additional restrictions, including the spacelike kinetic-transfer condition. Contributions requiring initial/final momentum comparable with M are outside the physical nonrelativistic domain. The module does not convert these formal tails into relativistic predictions. No antinucleon excitations are included.

## Numerical stability without physical broadening

`free_density_response.py` implements both `response()` and `log_response()`. The analytic numerator and denominator vanish together at z=0. We evaluate the removable limit, with a second-order expansion only for `|z|<10⁻⁵`; this is a numerical identity treatment, not threshold smoothing. A sign reversal evaluates negative z without exponentially large intermediate values. Stable softplus differences and a logarithmic tail branch avoid cancellation/underflow. The omitted relative correction in the tail branch is below `exp(−40)`.

The common energy zero can be shifted simultaneously in both offsets and both chemical potentials without changing any result. Extremely large shared rest energies can nevertheless limit the floating point precision of their subtraction; the checkpoint tests ordinary nuclear scales rather than arbitrary magnitudes.

## Evidence actually generated

Execute

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/response_reference/check_response.py
```

`results.json` records the measured errors, chemical potentials, offsets, momentum/energy transfers, UTC time, software versions and hashes of both Python source files. The check suite covers:

- Analytical S against an independently integrated transverse-momentum FD phase space, including degenerate and very dilute states, negative transfer and near-zero z.
- Chemical KMS reciprocity by explicit species reversal.
- Dilute Maxwell–Boltzmann response and zero-T Pauli/recoil interval limits.
- Exponentially small finite-T tails where the cold response is exactly forbidden.
- Direct two-dimensional momentum/angle static sums against `int S domega/(2 pi)`, including unequal species composition and offsets, with angular refinement.
- The neutral equal-species f-sum `int omega S domega/(2 pi) = n q²/(2M)`, using an independently integrated spin-two density n.
- Common energy-zero invariance and explicit invalid-transfer guards.

The static and f-sums check normalization, recoil and occupied-state counting beyond comparing two forms of the same one-dimensional integral. They remain internal free-gas checks. This checkpoint does not claim a different physical interaction model has independently reproduced the results. The tested neutral sums integrate omega over ±200 MeV; charged sums use ±400 MeV. Agreement with the direct momentum integrals bounds omitted tails at those tested states, rather than universally qualifying these cutoffs.

## Integration gate and next physics steps

The next step is an independently reviewed lepton-tensor contraction and optical prefactor. That work must derive the sign mapping from incident/outgoing lepton kinematics, preserve final lepton blocking, and prove number/energy moments and detailed balance before producing opacity values. A scalar response alone cannot be multiplied by an arbitrary `1+3gA²` factor and called a relativistic weak kernel.

A useful controlled extension is a fully specified leading-order nonrelativistic allowed weak-current model, with its vector/axial angular factors and declared accuracy. A separate relativistic response must retain its own phase-space measure, on-shell Jacobians, current matrix elements and kinematic support. GM1 effective masses and vector shifts can then be passed only into a compatible dispersion and vertex model. No such production integration was attempted in this batch.
