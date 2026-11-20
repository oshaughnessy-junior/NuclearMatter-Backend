# Independent numerical review: finite-temperature GM1 slice

15 October 2026. Reviewed the frozen `rmf_reference` implementation and source snapshots. No author files or Git state changed. `rmf_review.json` records the independently executed results and source SHA256 values; reproduce using `python3 outputs/nmb-finite-t-20261015/review_numerics/probe_rmf.py`.

## Disposition

**Accept the selected finite-T local uniform solutions and the narrowly demonstrated Sommerfeld product as research references. No numerical blocker was found at the reviewed states.** The source-convention regression and internal consistency checks do not supply a matched independent finite-temperature EOS comparison, phase equilibrium or a broad accuracy envelope. Those remain gates before production promotion.

The author reports 197 passing checks and supplies their actual outputs. This review adds **31 independently executed passing checks**, including three new finite-T states, a separate grand-potential formulation, thermal-antibaryon derivatives and an independently discretized Sommerfeld density derivative. It does not claim to independently rerun the author's 197-check driver.

## Provenance and equations

Both archived full-file Git blob identities independently match the author's pinned NMB source claims: rmft.py `1edba14f7422f37f1ae6e4eb58167dc2c2140061` and constants.py `cbd0b9009e1f333c6c447930670bcbfedaefd68e`. These checks compute SHA1 of the exact `blob <length> NUL <bytes>` representation; they are independent of AST extraction. The reviewed GM1 coefficients, common bare M=939 MeV, sigma/omega/rho normalization, vector signs, no cross coupling and unit conversion match that source convention.

The finite-T scalar condition follows stationarity of kinetic free energy plus `U(Phi)` at fixed species net densities. The density equations use particles minus antiparticles; scalar density, kinetic pressure, energy and entropy use sums. The opposite sign of the vector shift for antibaryons is carried by the kinetic chemical potential convention `nu=mu-V`: antiparticle occupation contains `E*+nu`. Thermal antibaryons and no vacuum sea are therefore mutually consistent in this specified Hartree ansatz. There is no additive T-independent vacuum contribution in the reviewed kinetic integrals.

## Independent numerical evidence

Three supplementary pointwise states were solved at `(nB [fm^-3],T [MeV],Yp)=(0.12,15,0.3),(0.4,50,0.2),(0.02,3,0.2)`. Maximum normalized field/number residuals range from 4.96e-16 to 1.26e-14. All pass stricter quadrature/tail reruns.

At each point, this review computes kinetic pressure from **scalar momentum-space integration of the grand potential**:

```
Pkin = T/pi² integral p² [log(1+exp((nu-E*)/T))
                        +log(1+exp((-nu-E*)/T))] dp .
```

This does not reuse the implementation's pressure moment, energy moment or entropy moment. Adding `-Uscalar+Uvector` agrees with returned total pressure. Independent finite differences also check pressure from Helmholtz free energy, the composition entropy Maxwell relation, and the Gibbs-Duhem relation `dP/dnB=nB[(1-Yp)dmu_n/dnB+Yp dmu_p/dnB]` at fixed T,Yp. The largest error in these supplemental checks is 2.36e-7 relative, for pressure from free energy at the low-density point.

Two explicitly pair-sensitive ideal-gas states at T=100 MeV check `n=dPkin/dnu`, `rho_s=-dPkin/dmD`, `s=dPkin/dT` and Euler consistency using the independent grand-potential integrator. Exact net-zero pair symmetry and positive thermal antiparticle density pass. These test the occupation convention; they do not validate full GM1 solutions at T=100 MeV.

The leading Sommerfeld density-of-states coefficient is dimensionally and thermodynamically correct: `gamma=sum kFi E*Fi/(3 HBARC)` for momenta in fm^-1 and T in MeV. Holding Yp fixed requires both Fermi momenta to vary with nB and the cold scalar solution to be recomputed. The implementation does this. A separate logarithmic-density finite difference with a different step reproduces `Gamma_th=2-dln(gamma)/dln(nB)` within 1e-7 at its three specified states. This verifies the derivative convention, not a global error guarantee at the proposed T/T_F cutoff.

## Phase, branch and domain boundaries

The documented spinodal example independently reproduces negative `dP/dnB|T,Yp` near nB=0.06 fm^-3,T=1 MeV,Yp=0.5. Converged number/field equations and positive local scalar curvature therefore cannot establish mechanical stability, full two-component diffusive stability or physical phase selection.

GM1 has a negative quartic scalar coefficient. Its potential is not globally bounded as unrestricted Phi tends to infinity. The numerical implementation deliberately restricts positive effective mass to `0<Phi<0.985M` and starts from the cold first upcrossing. It is a selected local effective-model branch; neither this review nor the author proves the global minimum or unique finite-T continuation. A caller-supplied initial state also does not certify branch connectivity by itself.

Selected new and original points do **not** qualify a rectangle in nB,T,Yp. Endpoints Yp=0,1 are excluded; extreme pair symmetry/diluteness can make density subtraction and the simultaneous root ill-conditioned. Solver failures should remain failures, and residual checks must not be relaxed to manufacture coverage. A future domain qualification should map residuals, branch continuity, scalar curvature and the species chemical-potential Hessian, then distinguish locally stable uniform matter from phase coexistence.

The baryon common-mass convention differs from the previously qualified static weak kernel with physical neutron-proton vacuum splitting. Do not combine these by substituting chemical potential differences into the old capture Q. A new charged-current calculation must use the matched RMF dispersions, blocking and energy/momentum constraints, with an explicit common-mass convention or a consistently revised EOS.

## Remaining acceptance work

1. Obtain an external finite-T GM1 point/table or independently implemented field solve with matched coefficients, no-sea convention, baryon content and independent composition. Forward agreement must precede derivative claims.
2. For each application trajectory, audit local phase/branch validity and conserve species net charge; include leptons/photons separately with consistent rest-energy conventions.
3. Promote the Sommerfeld product only with statewise finite-T increment checks and density-step refinement near any rapid mass variation. The proposed 0.03 kinetic-Fermi-scale restriction is not a proved uniform bound.
4. Validate EOS-consistent weak kinematics separately. Thermodynamic Maxwell identities do not determine a dynamical weak response function.

No new production or AD qualification is implied.
