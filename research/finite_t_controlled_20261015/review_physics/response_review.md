# Independent physics review: finite-temperature density-transition response

15 October 2026. Reviewed `response_reference/free_density_response.py`, technical note, test source and recorded 71-check results. Reviewer owns only `review_physics/` and changed no author files. Budget/account/window matched at review start (3% used).

**Accept this as a finite-temperature free common-kinetic-mass, spin-diagonal density-transition response reference. No blocking equation or normalization error found.** This acceptance does not supply a weak opacity, full weak tensor, interacting/RMF response or collisional nucleon width. Recoil/thermal phase space and nucleon Pauli blocking within the stated dispersion are substantive improvements over static targets.

## Independent normalization and response audit

The stated definition uses a spin sum **2**, a momentum measure `(2pi hbarc)^−3`, and an energy delta with explicit Fourier factor **2pi**. Spin diagonal unit matrix elements permit two initial-to-corresponding-final transitions; multiplying by four would double count. Orienting transfer q along z gives delta Jacobian `M/q`; transverse azimuth gives `2pi`; changing `p_perp dp_perp` to initial kinetic energy gives another `M`. Their product is

`2*(2pi)*(2pi)*M²/[q*(2pi)³*(hbarc)³] = M²/[pi*q*(hbarc)³]`.

This is the implemented prefactor. Remaining `dE` gives S units **fm^-3 MeV^-1**, and frequency integrals use `domega/(2pi)`. Neither a rate-unit `1/hbar` nor a weak coupling or neutrino phase-space factor belongs here yet.

For energy into the baryons, `omega=epsilon_b(p+q)−epsilon_a(p)`. With `Delta=O_b−O_a`, `w=omega−Delta`, the fixed initial longitudinal momentum is `p_z=Mw/q−q/2`; hence `Emin=p_z²/(2M)`. The equal-kinetic-mass assumption is essential. Prescribed offsets change w but do not alter this reduction. The FD vacancy argument yields `z=(omega+mu_a−mu_b)/T`, including total chemical potentials on the same energy zero as the offsets. All signs agree with the implementation.

The elementary transverse integral and small-z removable limit are correct. The documented second-order small-z expression has the right coefficients. Its numerical treatment smooths no physical threshold. Log tails preserve finite-T responses outside cold support; ordinary `response()` may legitimately underflow there.

## Reciprocity, support and independent probes

Explicit reversal gives `S_ba(q,−omega)=exp[−(omega+mu_a−mu_b)/T] S_ab(q,omega)`. This is grand-ensemble reciprocity for the two specified FD gases, including chemically imbalanced states. It does not assert neutrino chemical equilibrium. Offsets and total chemical potentials must move together under an energy-zero shift.

At T=0, the occupied-and-unblocked initial interval is `max[0,nu_a−max(Emin,nu_b−w)]`, exactly as coded. At finite T every finite q>0,omega has formal nonzero support for the **unbounded quadratic** dispersion. This is not the support of a relativistic response; tails requiring p~M or larger are outside the physical nonrelativistic approximation. q=0 is distributional and is correctly rejected by the scalar function. A future angular weak contraction must handle or integrate this endpoint without replacing the distributional limit by a pointwise zero.

`response_density_probes.py` writes `response_review_results.json` and adds two targeted independent checks:

- **81 100-digit comparisons** of the elementary integral near the z=0, |z|=1e-5 and dilute a=−40 numerical branches, with signed z. Largest absolute log error **1.55e-11**. This validates branch handling at selected adversarial values, not all representable inputs.
- A **charged equal-time commutator sum rule**, absent from the original suite: `integral [S_ab(q,omega)−S_ba(q,omega)] domega/(2pi)=n_a−n_b`. Independent spin-two momentum-density integrals and frequency response integration agree within **4.45e-16 relative** at the tested unequal-offset/unequal-composition state. The Pauli cross terms cancel after changing momentum variables, so this tests species counting and the explicit 2pi convention beyond one selected channel integral.

Both pass. The author's 71 checks also cover transverse phase space, neutral f-sums, selected static sums, KMS, cold/MB limits and tails. Their numerical agreement is internal free-gas qualification, not independent nuclear interaction physics. Cold boundaries and extreme floating-point transfers still require one-sided/scaled tests when broadening the domain.

## Requirements before any opacity or tensor claim

Next derive a named **leading nonrelativistic allowed-current** model. For unpolarized free equal-mass gases, the spin-density axial correlators share the same phase-space function under the declared noninteracting spin symmetry, but charge/axial matrix elements and their lepton angular contraction must be derived separately. The familiar angle-integrated static `1+3gA²` factor is insufficient as a generic differential contraction. Keep finite charged-lepton mass/velocity where required.

For each channel specify which lepton energies/momenta generate `(q,omega)` into the baryons, including inverse reactions; include final charged-lepton blocking and helicity/phase-space measures; then derive the s^-1 or cm^-1 normalization. Require positive collision coefficients, KMS with the correct neutrino chemical potential, direct multidimensional phase-space checks, and the dilute heavy-mass limit against the existing static kernel. When integrating nearly elastic q→0 or narrow response peaks, convergence must resolve the distributional limit.

Finite recoil with zero-width quasiparticles does **not** establish collision-induced spectral broadening or modified-Urca physics. Do not add that label or combine inclusive and exclusive channels without a stated matching policy. Interacting effective masses/offsets are frozen inputs here: density-dependent thermodynamic consistency, residual interactions and conserving vertices remain separate gates. A relativistic GM1 contraction needs its own compatible dispersion, on-shell measure and current tensor.

The primary reference [Reddy, Prakash and Lattimer](https://arxiv.org/pdf/astro-ph/9710115) was inspected for the density-response/weak-current distinction. Concrete acceptance here follows the audited definitions and independent probes; no external implementation was installed or executed.
