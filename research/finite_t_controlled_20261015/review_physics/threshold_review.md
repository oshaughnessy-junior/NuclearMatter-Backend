# Independent physics review: controlled thermal threshold example

15 October 2026. Coordinated freeze with the author before hashing/executing: `urca_threshold/threshold_study.py` SHA-256 **d658eceaf83574971940540fdc7f413cec47df6ba7ce0f955d7c1f5a9b787d1a**. Reviewed its code, cold support calculation, frozen research note and qualified/preserved-failure evidence. Reviewer changed only new `review_physics/threshold_*` files. Account/window matched at review start (6% used).

**Accept the nine core selected spectral points and their narrowly stated thermal-opening interpretation in the prescribed local-offset free NR model.** The finite-energy cold absorption onset is independently confirmed. **The T=.01 MeV extension remains excluded** because its forward quadrature gates fail; this review does not repair or relax those gates. No total direct-Urca emissivity, physical neutron-star beta-equilibrium sequence, interacting GM1 rate, collision width or universal suppression fit is established.

## Local offsets and thermodynamic scope

The common kinetic mass is 939 MeV, charge neutrality gives np=ne,net and independent nn,np are fixed at each selected composition. The code defines a **constant species offset separately for each local model**,

`On−Op=EFp+mu_e,cold−EFn`.

This makes the T=0 chemical affinity vanish by construction. Holding those offsets fixed at all temperatures is thermodynamically coherent: they add constant `On*nn+Op*np` to free/total energy and the same constants to chemical potentials, with no extra entropy/heat-capacity term. The FD factory and dynamical response receive identical offsets and total chemical potentials. Finite-T affinity is then allowed to change; it is not silently reset to zero.

The Y=.10,1/9,.12 cases have **different Hamiltonian offsets**, approximately 54.45,56.84,58.66 MeV. These are not vacuum neutron-proton Q values or self-consistent nuclear interaction potentials. The composition plot is a family of local controlled models, not a composition derivative or threshold scan through one global density-functional EOS. If one instead makes these offsets composition-dependent in a single potential, its derivatives require the additional offset terms; the current factory's chemical potentials would not automatically provide them. No such reinterpretation is accepted.

Uniform free nucleons are a deliberately restricted species/phase ansatz at nB=.01 fm^-3. The quoted pF/M values diagnose the NR expansion but are not an error bound on a realistic low-density nuclear mixture. The external finite-T GM1 forward gate remains open.

## Cold zero-neutrino triangle

At T=0 beta equilibrium, vanishing neutrino momentum and electron-only neutrality, all participating particles approach their Fermi surfaces. Momentum conservation requires `pFn<=pFp+pFe`; spin-two number counting and ne=np imply pFe=pFp. Therefore `(1−Y)^(1/3)<=2Y^(1/3)` gives **Y>=1/9**. This is a limiting geometric statement, not a finite incident-energy absorption veto.

For finite-energy absorption, Pauli blocking and energy conservation impose an excitation budget

`E_nu = delta_n_hole+delta_p_particle+delta_e_particle`,

with all terms nonnegative. The neutron momentum decreases with its hole energy, whereas final proton/electron momenta increase with their excitation energies. In the selected low-velocity state, the most efficient way to remove the momentum deficit at onset assigns the available energy to the slow proton. Favorable collinear momenta then give

`pFn−Ecrit = pFe+sqrt(pFp²+2M Ecrit)`.

Its small positive root is `B/[A+sqrt(A²−B)]`, with `A=M+pFn−pFe`, `B=(pFn−pFe)²−pFp²`. This applies to the below-triangle state; for already allowed zero-neutrino triangles the physical onset is zero rather than a negative root.

At Y=.10 the onset is **0.3089189799573 MeV**. At the exact threshold only boundary/zero-measure cold configurations appear; finite phase-space support opens above it. A finite E_nu can thus activate a cold absorption process even below the nominal Y=1/9 zero-neutrino triangle.

## Independent cold kinematic witness

`threshold_probes.py` does **not** call the author's cold-response interval optimizer to determine the onset. It directly allocates the incident energy among the three cold excitation costs, builds the four physical momentum lengths and minimizes their polygon-closure deficit. Four lengths can satisfy the vector equation iff the largest does not exceed the sum of the others. A separate root solve finds where the optimized deficit vanishes.

All **11 probes pass**, recorded in `threshold_probe_results.json`. The independently optimized onset is **0.308918979957265 MeV**, agreeing with the analytical expression at **2.00e-15 relative**. The optimizer independently chooses zero neutron-hole/electron-particle excitation at the onset, confirming the assumed proton-energy allocation. It finds cold support absent at E_nu=.09 and .30 MeV, present at .9 MeV, and changes support across `Ecrit±1e-5`. Above/on the triangle, infinitesimal positive incident energy is allowed in the tested geometry. Source hashes are recorded in the probe payload.

This independent geometric test complements the author's exact cold scalar-response interval optimization. Agreement checks the cold on-shell/Pauli constraints rather than comparing two quadratures of the same warm integrand.

## What the finite-T spectra establish

The response uses FD occupation/vacancy factors, finite electron mass and exact energy/momentum conservation for the specified quadratic nucleon dispersion. There is no numerical sigmoid, inserted threshold width or nucleon collision lifetime. Positive finite-T response can arise from thermally available states that the sharp cold occupations prohibit. The allowed-current vertex remains the previously qualified static leading-current ansatz; retaining recoil phase space is not a complete current 1/M calculation.

At Y=.10, T=.03, the sampled E_nu=3T=.09 MeV is **below** the independently established cold onset. The qualified finite-T log chi is −.672538..., i.e. chi≈.5104 s^-1, so this point is an unambiguous thermal opening within this model. T=.1 with E=.30 is also below onset. The T=.3 point uses E=.9 and is already cold-allowed; its larger response cannot be described entirely as thermal opening of a cold-forbidden channel.

Along the sequence both E_nu and T vary. Consequently chi/T² is a useful diagnostic, not a uniquely isolated thermal suppression function. The plot does not provide fixed-energy temperature derivatives or a validated universal fit. Inverse-channel KMS with the measured finite-T affinity is correct but is not transparent kinetic equilibrium, which requires the full independent directional reaction inventory.

Only electron capture and inverse nue absorption are sampled. No antineutrino/neutron-decay total is evaluated in this leaf; no neutrino energy integration, T^6 emissivity law, composition relaxation or total weak heating/loss source follows. Adding previously available crossed kernels elsewhere does not retroactively add their moments to these nine points.

## Qualified core and preserved colder failure

The core has nine points and 48 passing author gates, including independent momentum-density reconstruction and a separate scaled adaptive optical integration with a different energy/angle coordinate. These tests support the stated forward values within the shared physical model. The offsets are unchanged across each Y's qualified temperatures, also explicitly checked by the new probe.

The colder Y=.10,T=.01,E=.03 case fails both optical order refinement and the independently integrated forward comparison: about .6623% and .0360%, respectively, versus the unchanged 3e-6 gate. Its successful number inversion/KMS does not compensate. The failed script/payload remain preserved and the optional CLI extension remains a failing experiment. No reviewed value or figure should include that point as a qualified rate until a new cold integration scheme passes independent forward checks.

## Next scientific gate

Resolve the cold optical phase-space saddle/edge integration separately, preserving log tails and exact support. Then add the crossed antineutrino inventory with the same fixed local offsets and validate number/energy moments before any evolving material example. For astrophysical threshold predictions, replace the local family with a single consistent interaction EOS, compatible dispersion/current response and explicit phase/species scope; finite collision widths and modified-Urca matching remain additional physical calculations. The present selected spectra are a useful controlled proof of thermal occupation opening, not those broader results.
