# Independent review: pinned GM1 equation/fixture forward contract

**ACCEPTED for the selected same-equation numerical and convention audit. The external GM1 forward gate remains OPEN.** No native O2scl numerical result has been obtained. The authored momentum reduction, existing internal implementation, and this review's rapidity reduction solve the same declared Hartree equations; their agreement is not an external-code oracle or an estimate of physical-model uncertainty.

Reviewed frozen author manifest SHA256 `e07eb96cdd60b4eda2fdfdb674685cca535a840632ba22a23704921bb20811b9`. All manifest entries and the tracked internal dependencies match before and after the probe. Command: `/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_external_gm1_forward/probe.py`. Result: **64 PASS, zero failures**, 0.472 s. Checks include provenance and disclosure checks as well as forward physics identities; this count is not 64 independent physical benchmarks.

## Source and convention audit

The archived primary-source snapshots are pinned to O2scl revision `d68532ff31e59dac65844c3d387e1388cb71a770`: [RMF implementation](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/eos/eos_had_rmf.cpp), [RMF declarations](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/eos/eos_had_rmf.h), and [relativistic pair thermodynamics](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/part/fermion_rel.h). Re-fetch attempts during this review returned cache-miss errors; the equation audit uses the preserved, hash-verified local source bytes, not a newly fetched or executed upstream distribution.

In fm units let M=939/hbarc, Phi=g_sigma sigma, W=g_omega omega, R=g_rho rho. The source defines g_i=m_i c_i, so c_i=sqrt(C_i), with C=(11.785,7.148,4.410) fm². Its scalar source is spin-two per species, its total densities are net particle minus antiparticle, and its scalar/energy/pressure/entropy integrals add thermal particles and antiparticles. The latter are thermal excitations, not a Dirac vacuum sea. The nonlinear scalar term uses the common bare mass, not the effective mass: U=Phi²/(2C_sigma)+b M Phi³/3+c Phi⁴/4, b=.002947,c=-.001070. The scalar equation is dU/dPhi=n_s. The source's scalar trace identity (epsilon_gas-3P_gas)/m*=n_s is consistent with pair addition.

The rho halves match: W=C_omega nB; R=C_rho(np-nn)/2; mu_n=nu_n+W-R/2 and mu_p=nu_p+W+R/2. Vector energy and pressure each contain W²/(2C_omega)+R²/(2C_rho), hence rho energy C_rho(np-nn)²/8 and vector chemical work twice vector energy. Scalar field energy enters epsilon positively and pressure negatively. Rest-inclusive total chemical potentials and epsilon give T s=epsilon+P-mu_n nn-mu_p np. These factors and signs were independently checked in the new state.

The declared fixture is a common-939 MeV, uniform nuclear-only, no-sea Hartree local branch with independent nn,np, retained thermal antiparticles, no leptons or photons, and no imposed beta equilibrium. All extra O2scl couplings are zero and zm_mode is false. With those choices, separate meson masses cancel from the homogeneous field equations once the coupling ratios are fixed. They do not certify a finite-nucleus parameterization. The negative scalar quartic and local root choice do not establish global phase stability or coexistence.

## New heldout forward gate

I used p=m*sinh(y), E=m*cosh(y), adaptive rapidity quadrature with exact FD particle/antiparticle occupations, and an independent simultaneous three-variable root for Phi,nu_n,nu_p. All integrations and roots were in MeV units with explicit hbarc factors. The root started from independently constructed approximate free-gas quantities, rather than an authored converged field. Density and scalar residuals are at most 2.99e-15 after normalization. Tail/refinement controls changed from 55T/2e-11 to 85T/2e-13.

At **nB=.19 fm^-3, T=14 MeV, Yp=.22**, the independent results are:

| Quantity | Value | Units |
|---|---:|---|
| Phi | 329.617640180789 | MeV |
| m* | 609.382359819211 | MeV |
| mu_n | 978.162096382293 | MeV, rest-inclusive |
| mu_p | 883.994982073520 | MeV, rest-inclusive |
| epsilon | 179.247482346854 | MeV fm^-3 |
| P | 5.48511913250325 | MeV fm^-3 |
| s | .201284896059143 | fm^-3, kB=1 |
| f | 176.429493802026 | MeV fm^-3 |
| antiparticle density | 2.60902754385682e-41 | fm^-3 |

All twelve tested forward quantities agree with freshly evaluated authored momentum and internal kinetic-energy implementations within the unchanged 2e-9 relative gate; the largest observed relative discrepancy is 7.20e-14 in the extremely small antiparticle density. Raw results separately retain each implementation and the tail refinement. This is a forward gate only; derivatives were not tested.

## Native fixture and remaining gate

The C++ fixture's masses, spin degeneracies, rest-mass flags, zero extra interactions, input densities, T/hbarc conversion, public `calc_temp_e`, `get_fields`, and literal `calc_eq_temp_p` residual calls logically match the pinned declarations and equations. It does not feed the internal answer back as native initial fields. The preserved compile attempt fails on missing `o2scl/eos_had_rmf.h`; the fixture execution field is null and no installed library/image has been substituted. Syntax/link/runtime behavior and native solver convergence therefore remain unverified.

Closing the external gate requires an actual build against the pinned source, recorded compiler/library/environment hashes, native successful return status and independently refined fermion/root controls, followed by comparison of nn,np, scalar/vector residuals, fields, m*, rest-inclusive chemical potentials, epsilon,P,s,f. An internal expected row or this review's same-equation values cannot satisfy that requirement. No full NMB package parity, weak-rate, interacting response, phase-stability, or external-physics certification is granted here.
