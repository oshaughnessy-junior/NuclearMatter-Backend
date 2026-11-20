# Independent review: dynamical density response and GM1 local stability

15 October 2026. `probe_response_phase.py` executes this review without modifying author files. Final evidence is `response_phase_review.json`, with pinned source hashes. Original failed reviewer integration gates remain in `response_phase_initial_failures.json`.

## Disposition

**Accept both slices within their declared research ansatz and selected pointwise numerical tests.** No author-code numerical blocker was found. This review independently executes **74 passing checks**, supplementing the authors' archived 71 response checks, 270 phase/holdout checks and 24 gas-response checks. It does not independently rerun every author check or certify a continuous density/temperature domain.

The response is a spin-two scalar transition response for a common-mass, nonrelativistic free quasiparticle model. It is not yet an opacity or physical relativistic weak tensor. The phase calculation classifies **local nuclear-only uniform-matter stability**, with no phase coexistence, Coulomb, leptons or composition construction.

## Dynamical response evidence

The response reduction, delta-function Jacobian, spin factor and Fourier convention are consistent: spin-two phase space and `2pi delta(omega+epsilon_a-epsilon_b)` give `M²/[pi q (hbar c)³]` multiplying the one-dimensional kinetic-energy integral. Energy transfer is into the nucleons; both prescribed offsets and total chemical potentials must be retained. The chemical KMS exponent is therefore `(omega+mu_a-mu_b)/T`, not its kinetic-chemical-potential part alone.

This review compares the elementary reduced integral with independent scaled scalar FD-product quadrature at 35 `(a,z)` points: a=-100,-5,0,10,1000 and z=-100,-1e-5,-1e-8,0,1e-8,1e-5,100. All pass. These cases cross the small-z treatment boundary, include strong degeneracy and test strongly suppressed responses without exponentiating away their common logarithmic scale. The small-z series coefficients and negative-z reciprocity are mathematically consistent with the exact integral.

At neutral states `(T,nu)=(3,30),(0.01,50)` MeV and q=1e-4,0.1 MeV, integrate using omega=q*v. This resolves the narrowing small-q distribution directly. Independent number densities and susceptibilities are integrated in momentum space. Checks include:

- KMS at positive, negative and zero omega;
- the f-sum using KMS folding, `int_0^infinity omega S(omega)[1-exp(-omega/T)] domega/(2pi)=nq²/(2M)`, avoiding subtraction of large opposite-sign integrals;
- the q→0 static compressibility limit `int S domega/(2pi)→T dn/dnu`, at q=1e-4 MeV.

All pass, including the strongly degenerate state with nu/T=5000. An independent shift of the final species' energy zero, accompanied by the matching chemical-potential and omega shift, leaves the response unchanged. This supplements common-zero invariance and explicitly checks the transition-offset sign.

### Preserved reviewer-quadrature failure

The first small-q outer integrals split only at the Fermi velocity. Two gates failed: a static sum by 6.96e-5 and an f-sum by 3.13e-5. A long interval ending at vF missed the narrow deficit immediately below its endpoint, despite the adaptive integration's apparently satisfactory absolute tolerance. Explicit panels at `vF±q/(2M)+k T/(MvF)` with k=-40,-8,0,8,40 resolve both recoil-shifted thermal edges. Independent density/susceptibility quadrature receives equivalent Fermi panels. The same unchanged response function then passes all gates.

This failure belongs to this review's outer integrator, not the author's analytic response. It demonstrates a requirement for future optical kernels: rescale narrow support, panel every relevant Fermi/thermal edge and verify integrated sum rules. A higher nominal integration tolerance alone is not independent convergence evidence.

## Local stability evidence

The implicit Hessian follows canonical differentiation of the scalar and number constraints:

```
chi_i = dn_i/dnu_i
a_i = (drho_s_i/dnu_i)/chi_i
D = U'' + sum_i [drho_s_i/dm|nu + (drho_s_i/dnu_i)²/chi_i]
H_ij = delta_ij/chi_i - a_i a_j/D + vector_ij .
```

With fm internal units, D is fm^-2, dPhi/dn_i=a_i/D is fm², and H before multiplication by HBARC is fm². The returned chemical Hessian is consequently MeV fm³. The vector matrix is `C_omega` in every entry plus `C_rho/4` times the neutron/proton isovector matrix. Its signs match the existing GM1 vector fields. Pair terms and the fixed-density correction to scalar curvature have the correct signs.

Independent tests at `(nB,T,Yp)=(0.08,10,0.3),(0.1,12,0.2),(0.4,40,0.2)` compare the analytic Hessian with separately solved central species-density chemical-potential differences and total free-energy second differences. They also compare dPhi/dn_i, unsymmetrized chemical Hessian reciprocity, and canonical scalar curvature from independently density-inverted kinetic/scalar free energy at perturbed Phi. Every gate passes; the largest relative difference is 7.98e-7 for the free-energy Hessian at `(0.1,12,0.2)`.

The mixed instability counterexample independently reproduces at `(0.08,10,0.3)`:

```
dP/dnB|T,Yp = +1.51845128 MeV
minimum species-Hessian eigenvalue = -49.2708630 MeV fm³.
```

Thus a positive frozen-Y pressure slope does not establish local stability when both conserved species may fluctuate. The projection onto fixed-pressure composition fluctuations agrees algebraically with `nB(C-B²/A)` for the stated basis. The implementation checks Hessian asymmetry and nonpositive scalar curvature before assigning an accepted local label.

## Scope and remaining gates

- The 72 phase sample nodes and the reviewed derivative ladders are nodewise local evidence. They do not locate a binodal, certify interpolated boundaries or establish a globally preferred phase. Near zero eigenvalues require step/refinement and branch assessment rather than a sign claimed below numerical resolution.
- GM1's negative quartic coefficient and restricted local scalar branch retain the limitations already recorded in `rmf_review.md`. A positive canonical scalar curvature is necessary for the selected local branch, not a global effective-potential minimum certificate.
- The 24 Sommerfeld holdouts have archived maximum relative errors of 1.83e-4 in thermal energy, 6.09e-5 in thermal free energy, 1.95e-4 in thermal pressure, 1.22e-4 in entropy and 5.44e-6 in the thermal index. Those are measured errors at held-out nodes, not uncertainty bounds everywhere inside the proposed T/T_F restriction. The earlier independent density-derivative review remains applicable.
- The log-response retains finite-T tails in its declared nonrelativistic model; q=0 is distributional and excluded. Nonrelativistic all-real-omega support is not relativistic support. There is no antinucleon excitation or physical vector/axial optical prefactor here.
- Contracting this response with a weak lepton tensor requires an independently derived prefactor, angle/energy sign convention, final-lepton blocking and optical detailed balance. GM1 dispersions cannot be substituted into the free nonrelativistic formula and declared a conserving relativistic response.

No production edits, installs or author-file changes were made.
