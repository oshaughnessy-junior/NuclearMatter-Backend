# Off-pole external-leg weak OPE algebra: a deliberately incomplete current

This leaf implements four external-leg insertions of a one-body charged weak current around the accepted static Born OPE operator, with coherent final exchange. **199 selected algebra/control checks pass. This is not a complete modified-Urca current, finite-T shell integral, absolute rate or collision-width prescription.** In particular, the finite-momentum vector divergence is NONZERO, and a matched interaction current is still required. No output is promoted to a physical opacity or cooling coefficient.

## Conventions and declared model

Natural units, momenta/energies MeV, common nucleon mass M=939 MeV, degenerate neutron/proton rest masses and zero species potentials. External nucleons have epsilon(p)=M+p²/(2M). The four common rest masses cancel from the two-baryon energy equation; chemical potentials do not belong in these free propagator denominators. This is not a GM1 quasiparticle insertion. No EOS, density, occupations, thermal width, pairing or lepton integration enters.

The strong source is the **unchanged** `ope_nucleon_amplitudes/ope.py`: V(k)=−f²/m_pi² (sigma1.k)(sigma2.k) tau1.tau2/(k²+m_pi²), f=1.01, m_pi=140 MeV. It is static, unregulated full OPE, including its contact-distribution component. Independent OPE review qualified its conventions, not realistic NN accuracy. One-body states obey `<p'|p>=(2pi)^3 delta3`; spinors are normalized two-component states. The matrix basis is `(s1,s2) tensor (t1,t2)`, with isospin (n,p), tau3=(+1,−1), tau_minus=[[0,0],[1,0]]. It has unit n->p matrix element, not twice that value.

Write R_i=tau_i_minus (or its literal Hermitian adjoint for raising). The stripped one-body components are rho_i=gV R_i, jV_i=(p_before+p_after)/(2M) gV R_i and jA_i=−gA sigma_i R_i, with gV=1, gA=1.26 by default. The minus sign specifies V−A spatial components. Axial charge, magnetic current, form factors, induced pseudoscalar and relativistic corrections are NOT supplied. Including convection only tests the free NR vector continuity algebra; it does not complete a systematically accurate recoil expansion of the weak current. Raising is a different isospin insertion, **not** a completed crossed lepton reaction or computed inverse coefficient.

## External-leg derivation and signs

All weak four-momentum K=(omega,Q) is outgoing: p1+p2=p3+p4+Q and epsilon1+epsilon2=epsilon3+epsilon4+omega. The function checks both equations. Define q1=p3−p1 and q2=p4−p2. When radiation is on line1, the spectator's strong transfer is q2, so its strong matrix is V2=V(q2); for radiation on line2 use V1=V(q1). Evenness of static OPE removes the sign convention for the transferred pion, but **does not** make q1 and q2 equal at finite Q.

The resolvent is G(E,k)=[E−epsilon(k)]^−1. For incoming emission on i=1,2 and outgoing emission on f=i+2,

```
D_in,i  = epsilon(pi)−omega−epsilon(pi−Q)
        = −omega+pi.Q/M−Q²/(2M)
D_out,f = epsilon(pf)+omega−epsilon(pf+Q)
        = +omega−pf.Q/M−Q²/(2M).

B^mu = V2 J_in,1^mu/D_in,1 + J_out,1^mu V2/D_out,3
     + V1 J_in,2^mu/D_in,2 + J_out,2^mu V1/D_out,4.
```

This follows directly from `V G0 J + J G0 V` acting on the incoming ket. No denominator is replaced by a lepton chemical potential. The time order fixes the operator order: isospin exchange in V and the charged insertion generally do not commute. The API rejects any `|D|<=pole_gap`; its default 0.05 MeV is an **explicit excluded numerical domain**, not a physical width, smoothing scale, pole subtraction, imaginary self-energy or completeness claim. Changing that exclusion is not a physical uncertainty estimate. On-shell spectator poles, overlapping DU diagrams and finite widths remain outside this reference.

The labelled final antisymmetrized matrix is A(12;34)=B(12;34)−P B(12;43), with P swapping both spin and isospin output labels. There are four direct terms and their four exchanged terms, combined before squaring. It also obeys initial exchange antisymmetry; this was checked rather than assumed. There is no sqrt2, initial spin average, isospin average or event factorial in A. The saved channel diagnostics are complete spin **sums** for a specified labelled n/p channel. Any eventual many-body MU integral must additionally use its labelled event symmetry factor (one identical nucleon pair gives 1/2 in the declared branch inventory) without counting exchange interference a second time.

V has units MeV^-2, G MeV^-1, and all stripped one-body components are dimensionless; A has units MeV^-3 and its squared spin sum MeV^-6. GF Vud/sqrt2, the lepton current, covariant external-state conversion and all phase measures are absent. Neither these squared matrices nor their angular averages are absolute MU matrix elements in a pre-existing covariant convention.

## What the current tests prove, and the missing interaction current

At Q=0, V1=V2=V and D_in=−omega, D_out=+omega. Therefore A^mu=[J_total^mu,A_strong]/omega. Isospin scalar OPE commutes with total charged isospin, giving A_V0=0 in this isospin-symmetric long-wavelength model. The charged spin/axial generator does not commute with the tensor interaction; all three selected axial commutators are nonzero. This is an algebraic limit, not a claim that physical finite-q charged-current vector contributions vanish.

For finite Q, the included convection terms give the following denominator-free identity, separately before final exchange:

```
omega B_V0 − Q.B_V = gV ([R1,V2]+[R2,V1]) = Delta_V.
```

It follows because omega*rho−Q.j_in=−D_in*rho and omega*rho−Q.j_out=+D_out*rho. Antisymmetrize the RHS just as B. At the two finite-Q test points, the maximum absolute matrix entries of this divergence are 2.37646518899e−6 and 3.49980906839e−6 MeV^-2. **It does not vanish.** The missing interaction current must cancel this defect in the same Hamiltonian/energy-transfer convention. A manufactured longitudinal term proportional to Q*Delta_V/Q² would fix one equation while leaving the transverse current arbitrary; no such substitute is implemented.

The next construction must gauge the same derivative pion-nucleon interaction and potential reduction. Pion-in-flight and contact/seagull vector contributions need matched signs, pion charge flow and any regulator currents. A consistent axial calculation also needs the associated pion-production/absorption operators and PCAC constraints; a nonzero axial commutator is not a PCAC verification. Delta excitations, short-range terms and associated LECs require an explicit power counting rather than automatically adding every term from another model. Do not insert a phenomenological current and claim it is uniquely determined by static V.

## Primary-source audit

[Shternin, Baldo & Haensel (2018), Eqs3–5 and Fig1](https://arxiv.org/pdf/1807.06569) supply a leading NR V−A vertex, before/after external-leg propagators and a warning that the in-medium spectrum matters. Their strong G-matrix, Fermi-surface treatment and omitted intermediate-state diagrams are not copied into this free Born reference. Our finite-Q formulas above are independently derived time-ordered algebra.

[Suleiman, Oertel & Mancini (2023), SecII and appendices](https://arxiv.org/html/2308.09819v2) motivate retaining diagram interference and auditing vector approximations. Their displayed Eq23 numerator ambiguity remains unresolved; this leaf inherits the independently sourced strong OPE convention, not a silently corrected Eq23 or its polarization normalization.

[Ruiz Simo et al. (2016), Eqs12–31](https://arxiv.org/pdf/1604.08423) explicitly distinguish seagull, pion-in-flight, pion-pole and Delta terms in their relativistic two-body current. Their operator is not added wholesale to this static potential model. [Dmitrašinović & Sato (1998)](https://arxiv.org/abs/nucl-th/9806103) establish the need for associated two-body axial and pion operators for nuclear PCAC. These sources specify missing consistency work, not completed coefficients here. Primary renderings checked online 2026-10-15; no byte-pinned PDF archive is claimed.

## Measurements, preserved failure and independence boundary

`python check_algebra.py` performs 199 selected checks in 0.067 s. A second construction fills individual strong and weak spin/isospin matrix elements by explicit indices and uses differences of independently computed NR kinetic energies, rather than importing strong matrices, Kronecker assembly or expanded propagator formulas. Agreement tests operator ordering and signs; **both constructions share the declared Born model and do not independently validate nuclear physics**. Additional gates test both exchanges, forbidden charge blocks, soft vector/axial limits, finite-Q divergence, MeV dimensional scaling and input rejection.

The two finite-Q points have (p_CM,omega,Q)=(100,5,(2,−1,3)) and (135,8,(−3,4,2)) MeV; raw final directions, all four denominators and complex nn->np matrices are saved. External momenta/M are at most about0.15 at these selected states. No numerical rectangle, density or thermal domain is qualified.

An adversarial complex-input regression initially FAILED: `amplitude` cast complex momenta to float before the lower-level validator. `initial_external_legs.py`, `initial_check_algebra.py` and `initial_results.json` preserve that 198/199 result and source. The wrapper now rejects complex momenta before casting. The unchanged gate passes in the final199/199 run; finite/overflow invariant, non-Boolean raising, pole and conservation controls are retained. This repair changes input handling, not the operator.

Next implementable rungs: (1) independently review these conventions; (2) derive matched vector interaction currents and test cancellation of this **measured** finite-Q defect, including exchange; (3) establish axial/PCAC input and its short-range uncertainty; (4) introduce species-dependent quasiparticle resolvents with an explicit energy/rest convention; (5) contract the completed current with independently checked lepton tensors and compare off-pole absolute few-body benchmarks before any thermal shell integral. Width resummation and DU/MU matching require a separate non-overlapping approximation. No width, smoothing fit, total thermal MU rate, interacting-EOS result or realistic uncertainty band is supplied by this leaf.
