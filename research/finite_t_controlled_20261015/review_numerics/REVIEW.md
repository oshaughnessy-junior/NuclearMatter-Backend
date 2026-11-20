# Independent numerical review — first reference slices

Reviewed 2026-10-15, starting 11:00 UTC. No author files, production source or Git state modified. Review code/results live only in this directory. This is an independent numerical review of the declared equations, not an independent weak-interaction physics calculation.

## Disposition

**Accept the selected benchmark states as research references. Restrict coupling to an explicitly tested state range and enforce residual/input gates. Do not accept a domain-wide qualification.**

The original 68 thermodynamic checks and 44 weak-rate checks reproduce in a different installed environment (Python 3.13, NumPy 2.3.5, SciPy 1.15.3); copies write only to this review directory. The exact homogeneous slab also reproduces its PASS result, with moderate-depth conservation relative error 3.67047e-5 and opacity derivative error 1.02993e-6. `reviewed_source_hashes.json` pins the inspected files.

Independent scalar momentum-space quadrature agrees with lepton net density, total energy and pressure at four additional states: electron `(T,mu)=(1,2),(10,0)` MeV and muon `(0.01,110),(20,-100)` MeV. This changes variable, quadrature shape and component scaling relative to the author's kinetic-energy `quad_vec` calculation. The independent analytical charge-odd massless capture number-moment identity passes at `(T,mu)=(1,0),(0.01,100),(10,-100)` MeV. The selected cold dilute massive-gas Cv/n limits also pass down to T=1e-8 MeV; this isolated limit is not evidence for all cold states.

## Reproducible findings

1. **Unchecked density inversion residual can return a physically different conserved charge.** Fixed absolute Brent tolerance `xtol=1e-13` is inappropriate as the root chemical potential approaches zero. Electron gas `state_density(10,1e-18)` returns net density `9.07967193e-19 fm^-3`, a 9.2% error, and `state_density(1,1e-24)` returns zero. These targets are small but finite; a stable forward occupation subtraction does not cure a root solver's absolute stopping criterion. Massless variants are saved in `review_results.json`, including a 100% error at positive target density. A caller must reject an unmet net-density residual before using the result for charge neutrality, or the author must change and requalify inversion. No blanket positive-density support should be claimed yet.

2. **Weak public numerical entry points do not reject nonfinite inputs.** `spectral(valid_state,nan,channel)` returns NaNs; infinite energy produces NaN/Inf. `target_moments(nan,0,channel)`, NaN mu, and infinite tail similarly return NaNs without an exception. `State` itself validates its fields correctly, but direct moment/spectrum users can bypass that protection. A coupling boundary must validate all inputs and outputs until these interfaces are hardened. This matters because a nonlinear solve can probe invalid states and apparent convergence comparisons involving NaNs can silently fail if written poorly.

3. **Extremely suppressed positive rates silently underflow to zero.** Examples are electron capture `(T,mu)=(0.001,0)` MeV and positron capture `(0.01,8)` MeV, with common logarithmic suppression about -1293 and -851 respectively. This is already acknowledged in the author's note and does not block ordinary floating-point source calculations in the selected domain. It does block claims about relative tiny-rate accuracy, exact kinetic-equilibrium ratios or derivatives in that extreme domain without a log-rate representation and an underflow flag. Do not interpret such zeros as kinematic forbidden channels.

## Accepted scope and limits of independence

The lepton paper's selected states and the weak note's five `(T,mu)` points are accepted as pointwise benchmarks. They are not a filled parameter rectangle. The weak selected finite-mass checks cover T=0.03–10 MeV and mu=-2–8 MeV, plus explicitly synthetic analytical limits. Thermodynamics independently passes the supplemental points above; its broad cold-limit qualification remains pending.

The author's weak adaptive integrator shares the same physical equations and constants. Its agreement with Gaussian quadrature is useful numerical evidence, not independent evidence for physical omissions, spin/helicity conventions or omitted recoil. Detailed balance checks validate algebra within the common-wavelength static-target ansatz. An independently sourced normalization and later recoil/degeneracy comparisons remain separate gates.

The slab benchmark solves prescribed homogeneous coefficients exactly. Its source integral is independently discretized, but its energy check at each fixed energy is exactly energy times its number check. It does not independently test broadband energy accounting, matter energy loss, cross-group redistribution or microphysics. The stated scope already makes this distinction; future integration should retain it.

## Gates for the next coupled slice

Use a small warm dilute nucleon element before extending the domain. Lepton charge-neutrality spot checks at fixed net density 1e-6 fm^-3 and T=0.1,1,3,10 MeV invert to relative residual below 1e-14 in this environment. **This is a starting set, not a qualification of every intermediate composition**; verify the entire proposed Ye root bracket and evolution trajectory.

- Reject nonfinite or negative T/densities and unphysical Ye before entry; reject nonfinite numerical outputs afterward.
- Record requested and reconstructed net density, with a scale-aware residual gate and a distinct exactly-zero branch. Verify both signs where the generic lepton interface advertises them.
- At every new representative trajectory state refine weak quadrature order and tail independently; check absolute error when a physically irrelevant channel is exponentially suppressed rather than dividing by its underflowed rate.
- For fixed T and nB, compare the transparent composition stationary root of `R_pc+R_nd-R_ec` with time evolution and step-size refinement. Check the rate-balance residual directly, and do not substitute `mu_nu=0` as transparent equilibrium.
- In the spectral slab integrate all three supported reactions on their exact supports. Compare volume-integrated matter number/energy sources with the corresponding boundary number/energy fluxes, using several genuine energy groups and both neutrino species.
- Before thermal evolution, derive total matter energy including nucleon rest energies and lepton/pair energy. Enforce the composition-energy term and radiation balance; adding a separate Q source would double count rest-mass conversion.
- Fixed-bath neutrino equilibrium must annihilate every channel collision term pointwise. A closed box must additionally conserve total electron lepton number and matter+radiation energy under refinement.

No fixes to author code are proposed or applied here. These findings define acceptance gates for the orchestrator and authors.
