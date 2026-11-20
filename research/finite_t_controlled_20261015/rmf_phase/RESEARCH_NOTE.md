# Two-charge local stability and low-temperature holdouts for uniform GM1

15 October 2026. This bounded study extends the accepted **local** finite-T GM1 reference with the missing two-species susceptibility/stability diagnostic. All new code and results are confined to `rmf_phase/`; the frozen implementation in `rmf_reference/` is imported without modification. The source remains NMB snapshot `25e68182f726e5bd9c0b372c9a667536940fbd07`, with provenance and matching cold calibration in the preceding checkpoint.

The central result is concrete: **positive compressibility at fixed proton fraction can miss an unstable coupled neutron/proton fluctuation**. Four of the 72 sampled nuclear-only states exhibit that failure. A useful application gate must inspect the full chemical Hessian on the physically chosen composition constraints.

## Reproduce and inspect

```bash
python outputs/nmb-finite-t-20261015/rmf_phase/run_stability_study.py
python outputs/nmb-finite-t-20261015/rmf_phase/validate_response.py
python outputs/nmb-finite-t-20261015/rmf_phase/plot_stability.py
```

Measured results: **270/270** combined Hessian/map/holdout checks and **24/24** separate ideal-quasiparticle response derivative checks pass. The full study takes about 83 seconds on the current host. `validation_results.json` contains actual matrices, raw finite-difference/Richardson values, stability labels and approximation errors. `response_validation.json` records the separate distribution-derivative checks. The PNG and PDF show **sampled nodes only**, with no interpolation, binodal or purported certified boundary.

No production files, frozen reference files, dependencies or git state were changed. Matplotlib was already available. Its cache is redirected to the authorized temporary directory by the plotting script.

## Physical question and independent variables

The reference is uniform, zero-field, nuclear-only GM1 matter at fixed temperature, with independently specified neutron and proton **net** number densities. The local nuclear free-energy density is `f(T,nn,np)` after self-consistently solving the scalar field and the two number constraints. The matrix of interest is

```
H_ij = dmu_i/dn_j|T = d²f/(dn_i dn_j)|T,
i,j in (neutron,proton).
```

Returned units are MeV fm³. A positive definite H means that the specified uniform branch is locally convex against these two conserved-density fluctuations. A negative eigenvalue identifies a bulk spinodal direction; its eigenvector specifies the neutron/proton density combination. Local convexity does **not** prove that the homogeneous state is the global free-energy minimum.

The nucleonic constraint is deliberate. Leptons, Coulomb fields, gradients/surface tension, nuclear composition, pairing and weak equilibration are excluded from this map. These omissions matter for deciding which modes physically exist in a charged stellar fluid. For example, if local charge neutrality forces an ideal electron density to follow `np`, the free energy gains `fe(T,np)` and the pp entry gains the positive lepton inverse susceptibility `dmu_e/dne`. Coulomb and finite-wavelength effects require their own response/geometry. The nuclear-only spinodal must not be relabelled as the neutron-star crust boundary or a neutrino-opacity prediction.

## Implicit Hessian from the scalar and number equations

Use the frozen reference's internal fm units in this derivation: energies/momenta, masses, kinetic chemical potentials and temperature are in fm^-1; density is in fm^-3. `m=M-Phi`, `nu_i=mu_i-V_i`, and `Vn=W-R/2`, `Vp=W+R/2`. Let `f_i` and `fbar_i` be the particle and antiparticle Fermi occupations, `w_i=f_i(1-f_i)`, `wbar_i=fbar_i(1-fbar_i)`, and `E=sqrt(p²+m²)`. For each species define

```
chi_i = dn_i/dnu_i|m = integral p²(w_i+wbar_i) dp /(pi² T)
b_i = drho_s_i/dnu_i|m
    = integral p²(m/E)(w_i-wbar_i) dp /(pi² T)
q_i = drho_s_i/dm|nu_i
    = integral p²[(p²/E³)(f_i+fbar_i)
                 -(m/E)²(w_i+wbar_i)/T] dp /pi².
```

Differentiating the number integrals gives `dn_i/dm|nu_i=-b_i`, which is also checked independently by finite differences. Therefore, along fixed species density,

```
a_i = drho_s_i/dn_i|m = b_i/chi_i
dnu_i/dm|n_i = a_i
drhos_i/dm|n_i = q_i+b_i²/chi_i .
```

The fixed-density scalar equation is `G=U'(Phi)-sum_i rho_s_i(M-Phi,n_i)=0`, so

```
D = dG/dPhi|nn,np
  = U''(Phi)+sum_i [q_i+b_i²/chi_i]
dPhi/dn_j = a_j/D
H_ij = delta_ij/chi_i - a_i a_j/D + dV_i/dn_j .
```

For the source's density-independent GM1 couplings and no omega-rho coupling,

```
dV/dn = [[C_omega+C_rho/4, C_omega-C_rho/4],
         [C_omega-C_rho/4, C_omega+C_rho/4]].
```

The first term in H is the positive species kinetic response, the second is the attractive scalar relaxation contribution, and the third is the vector/isovector contribution. The raw matrix in fm² is multiplied by `HBARC` to return MeV fm³. `D` has units fm^-2. A positive D checks local canonical scalar curvature, but the remaining coupled-density matrix can still be indefinite. This is exactly why scalar-solver convergence and local scalar stability alone are insufficient.

The implementation checks quadrature status and estimated error. A near-singular scalar denominator is rejected. `classify` reports an unqualified result if supplied a materially nonsymmetric Hessian, rather than treating numerical asymmetry as physical. It also rejects a nonpositive supplied scalar-curvature diagnostic as an unstable scalar branch. The displayed eigenvalues are computed from the symmetric part after reporting the raw antisymmetry.

These formulas apply to this constant-coupling GM1 ansatz. Density-dependent RMF couplings, omega-rho cross couplings or additional fields require the full corresponding field Jacobian and rearrangement terms. This module does not silently generalize them.

## Mechanical and diffusive projections

Write `n=nn+np`, `Y=np/n`, `v=(1-Y,Y)` and `u=(-1,1)`. Let `A=vᵀHv`, `B=vᵀHu`, `C=uᵀHu`. Then

```
dP/dn|T,Y = n A
d(mu_p-mu_n)/dY|T,n = n C
d(mu_p-mu_n)/dY|T,P = n(C-B²/A) = n det(H)/A .
```

The fixed-pressure slope is undefined at `A=0` and is not used as a substitute for the full Hessian in that limit. When `A>0`, its positivity is the complementary diffusive stability requirement. Looking only at the fixed-n composition curvature `C` can also miss the coupled instability: a positive curvature along each chosen coordinate direction does not guarantee convexity along a mixed direction.

At eight selected states, three finite-difference step sizes `(0.002,0.001,0.0005)` times each species density generate raw chemical Hessians and Richardson extrapolants. These are compared with the implicit response, with a norm gate of `2e-6` relative to the matrix scale; the ladder and raw symmetry each have their own gate. Independent second differences of the **free energy**, using larger steps to control rest-energy subtraction, agree within a `1e-4` matrix-scale gate. Separate fixed-Y pressure derivatives and compensated fixed-pressure composition perturbations verify the projected formulas. Twenty-four additional checks differentiate the gas number/scalar integrals directly at fixed `nu` or mass.

## Measured map and useful counterexamples

The modest grid is

```
nB = (0.02,0.04,0.06,0.08,0.12,0.16,0.24,0.32) fm^-3
T  = (1,10,20) MeV
Yp = (0.1,0.3,0.5).
```

Of its 72 nodes, **52 have a locally positive Hessian and 20 are spinodal**. The marginal-eigenvalue threshold is 1e-7 times max(1,the matrix spectral norm), in the returned units. No sampled node lands inside that numerical threshold; it is not a physical model-uncertainty bound. At T=20 MeV all these particular sampled nodes are locally convex; that fact does not locate a critical temperature, establish global stability or exclude an instability between nodes.

Four states have positive fixed-Y mechanical slope despite a negative Hessian eigenvalue:

| nB [fm^-3] | T [MeV] | Yp | smallest eigenvalue [MeV fm³] | dP/dn at fixed Yp [MeV] | fixed-P diffusive slope [MeV] |
|---|---:|---:|---:|---:|---:|
| 0.06 | 1 | 0.1 | -280.089890 | +4.101856 | -263.924609 |
| 0.08 | 1 | 0.1 | -95.709813 | +13.406203 | -47.251247 |
| 0.02 | 10 | 0.3 | -26.405586 | +0.121663 | -172.109330 |
| 0.08 | 10 | 0.3 | -49.270863 | +1.518451 | -210.199832 |

The last state is included in the full independent FD/free-energy ladder, so the qualitative finding is directly checked beyond the analytic response formula. The figure marks these four points with black circles. The susceptibility eigenvector can be used to choose a controlled perturbation direction; it is not a growth rate without a kinetic/transport model.

For subsequent small response experiments, useful **locally stable candidate states** with independently checked derivatives are:

| nB [fm^-3] | T [MeV] | Yp | smallest Hessian eigenvalue [MeV fm³] |
|---|---:|---:|---:|
| 0.12 | 5 | 0.1 | +287.097172 |
| 0.16 | 10 | 0.1 | +616.378740 |
| 0.24 | 20 | 0.3 | +894.321871 |
| 0.32 | 30 | 0.1 | +1078.605710 |

These have comfortable numerical separation from the local spinodal. Their designation remains nuclear-only and local; it does not replace a physical EOS validity model or a global phase comparison. A next homogeneous weak-response calculation should start with the already used `(0.16,10,0.1)` or `(0.32,30,0.1)` point, with consistent mean-field dispersions and explicit lepton/radiation closure.

## Blocked low-T approximation holdouts

The preceding reference supplied the leading Sommerfeld free-energy approximation `delta_f=-gamma(n,Y)T²/2`, with energy, entropy, heat capacity and pressure derived consistently. This study probes **24 new states** with densities `(0.11,0.21,0.45,0.65) fm^-3`, fractions `(0.07,0.23,0.41)`, and temperatures `(0.05,0.2) MeV`. All density and fraction nodes differ from the three original asymptotic sample states. No new coefficients are fitted.

Measured maximum relative errors against detailed finite-T calculations on those holdouts are:

| Quantity | Maximum relative error |
|---|---:|
| Thermal energy increment | 1.82846e-4 |
| Thermal free-energy increment | 6.08912e-5 |
| Thermal pressure increment | 1.94661e-4 |
| Entropy density | 1.21865e-4 |
| Thermal index `1+delta_P/delta_e` | 5.43752e-6 |

Each fit state also passes a pressure check from a **separately differenced approximate free energy**, verifying `delta_P=n*d(delta_f)/dn-delta_f` at fixed Y. That is the derivative convention needed by the compact representation; a fit to thermal pressure alone need not obey it.

These are measured selected holdout errors, not an analytic error bound over all temperatures allowed by the original conservative `T<0.03 min(E*F_i-mD)` guard. Near zero increments, full-energy subtraction will itself become unreliable; the current T values keep the increments resolvable. No automatic broad-domain fit certification or extension to a phase transition is claimed.

## What a global phase calculation must do next

A binodal/coexistence construction was deliberately not fabricated by flattening `P(n)` at fixed Yp. In a two-charge nuclear system, the gas and liquid generally have different compositions. At fixed T, both phases must satisfy

```
mu_n^L=mu_n^G, mu_p^L=mu_p^G, P^L=P^G
nn_bar=alpha nn_L+(1-alpha)nn_G
np_bar=alpha np_L+(1-alpha)np_G.
```

The mixture free energy must beat the uniform candidate, both phases need appropriate local stability, and a solver must exclude the ever-present trivial identical-phase solution. Continuation/multistart and a common tangent plane in the two density variables provide useful independent checks. The conservation/neutrality constraints must be stated before solving: local charge neutrality, global charge neutrality, trapped lepton number or a prescribed local proton fraction are different physical problems. Surface/Coulomb or nuclear-cluster physics may invalidate a simple macroscopic two-phase bulk ansatz for the intended stellar application.

The useful next milestone is therefore either a clearly labelled **bulk two-charge GM1 Gibbs reference** at a few states, or the physically motivated lepton/finite-wavelength stability extension needed by the first application. It should not be sold as an arbitrary-composition stellar EOS until those choices and the external finite-T comparison are qualified.

## Primary context

- [Müller & Serot 1995, *Phase transitions in warm, asymmetric nuclear matter*](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.52.2072): finite-T RMF with two conserved charges; the distinction between coupled chemical and purely mechanical instability is directly relevant here.
- [Ducoin, Chomaz & Gulminelli, *Isospin-dependent clusterization of Neutron-Star Matter*](https://arxiv.org/abs/nucl-th/0612044): free-energy curvature and the effect of electrons/beta-equilibrium in a separate mean-field model. It motivates the stellar-matter caveat, not a numerical GM1 comparison.
- [Hempel, Pagliara & Schaffner-Bielich 2009, *Conditions for phase equilibrium in supernovae, protoneutron, and neutron stars*](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.80.125014): conservation and local/global neutrality choices determine the applicable equilibrium construction.
