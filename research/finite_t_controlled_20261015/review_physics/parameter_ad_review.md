# Independent physics review: RMF coupling sensitivities and parameter work

15 October 2026. Reviewed frozen `rmf_parameter_ad/gm1_parameter_ad.py`, research note, validation driver, 504-check payload, manifest, relevant frozen RMF field/response definitions and archived source formulas. Reviewer changed only new files in `review_physics/`; no author or production module changed. Campaign account/window matched at start (4% used).

**Accept the local first-order raw-coupling sensitivities on the three documented warm uniform branches and small perturbations. No blocking parameter-work, implicit-response or coupling-convention inconsistency found.** Acceptance remains conditional on the underlying finite-T forward RMF reference and its local branch qualifications. It does not independently validate that forward theory against another finite-T implementation, supply a physical uncertainty distribution, or qualify large parameter excursions.

## Generalized first law and pressure derivative

Let f(T,nn,np;lambda) be the reduced canonical free-energy density, after algebraically eliminating vector fields and evaluating the scalar stationary solution. Define `X_lambda=(partial f/partial lambda)_T,nn,np`. Then

```
df = −s dT + mu_n dnn + mu_p dnp + X_lambda dlambda
de = T ds + mu_n dnn + mu_p dnp + X_lambda dlambda
dP = s dT + nn dmu_n + np dmu_p − X_lambda dlambda.
```

The energy relation follows from e=f+Ts; the pressure relation follows from P=mu_n nn+mu_p np−f. The implementation has the correct **minus** parameter-work term in dP. Applying fixed-coupling Gibbs-Duhem while varying a coupling would be incorrect. The vector examples make the failure concrete: their f/P interaction terms are quadratic in density, so `sum n_i mu_i,lambda` gives twice the actual pressure parameter derivative.

These generalized forces describe the derivative of a Hamiltonian family. Ordinary physical fluid evolution keeps the nuclear couplings fixed. If a controlled model deliberately varies them in time, the external parameter-work reservoir contributes `X_lambda dot(lambda)` consistently. Comparing different EOS parameter choices in an inference calculation does not itself imply a physical time-dependent heating source.

## Explicit reduced-potential derivatives and units

For constant bare M, scalar `mD=M−Phi` and zero omega-rho cross coupling,

`f = sum fkin(T,ni,mD)+Phi²/(2Cs)+b M Phi³/3+c Phi⁴/4+Co nB²/2+Cr n3²/8`.

Scalar stationarity cancels the implicit Phi variation from the first derivative at fixed T,ni. Vector fields must first be eliminated with their own stationary/algebraic equations; freezing an independent W while changing Co is not the same partial derivative. The resulting explicit derivatives are exactly

`(-Phi²/(2Cs²), nB²/2, n3²/8, M Phi³/3, Phi⁴/4)`.

Multiplication by HBARC converts the internal fm potential to the returned MeV/fm³ convention. Raw Cs/Co/Cr have units fm²; b/c are dimensionless. This is sensitivity to these raw squared-coupling/mass combinations, not directly to g_sigma/g_omega/g_rho, meson masses, saturation observables or a calibrated posterior. Those need an explicit chain rule and declared held-fixed quantities. The fixed 939 MeV bare mass is consistently excluded from the input space.

## Implicit differentiation and fixed-state semantics

The scalar gap and number-constraint Jacobian signs are correct. Since `d mD/dPhi=−1` and `n_m=−rho_s,nu`, the number rows have +rho_s,nu in the Phi column. The scalar row has `U''+sum rho_s,m` and negative scalar susceptibilities in the chemical columns. At fixed Phi,nu,T, the explicit coupling scalar row is `(-Phi/Cs²,0,0,M Phi²,Phi³)`; number-equation explicit coupling rows vanish. These are the appropriate sources for `Fz dz=−Fx dx`.

The direct vector contributions `dW=Co dnB+nB dCo` and `dR=Cr dn3/2+n3 dCr/2` are retained in total chemical potentials. Entropy has no explicit raw-coupling dependence at frozen kinetic state; its dependence through fields and kinetic chemical potentials is included. MeV temperature input versus internal inverse-fm temperature is converted consistently. The chemical-Hessian diagnostic uses the supplied perturbed couplings rather than silently falling back to baseline vector coefficients.

Density, temperature and proton fraction are independent inputs. No beta-equilibrium composition or saturation refit is differentiated. The measured derivatives therefore answer a precise conditional question: change one coupling while holding this nuclear state coordinate fixed, then resolve the local field/number equations.

## What is independent in the source checks

The restricted AST executes the archived NMB **primal potential formulas**, freezes Phi/densities, varies one source parameter and compares with the new analytical explicit partial. This is useful independent implementation evidence for signs, rho conventions and units. The cold kinetic term is legitimate for this specific test because it has no explicit coupling dependence at fixed Phi and densities; its explicit derivative is the same as the finite-T kinetic term's zero explicit derivative. It does **not** validate finite-T occupation integrals, all implicit field derivatives, a different EOS model or an external physical calibration.

Re-solving the nonlinear finite-T state at decreasing coupling steps provides separate implementation evidence for total output derivatives, with forward residual and local stability gates before derivative comparison. The mixed Hellmann-Feynman/Maxwell identities are useful cross checks, but reuse the frozen state-only tangent and do not constitute an independently qualified Hessian or nested AD implementation. The note states these distinctions adequately.

## Additional independent parameter-work probes

Executed `parameter_ad_probes.py` with the existing junior_tools Python environment; no installs or full-suite replay. Results are in `parameter_ad_probe_results.json`. **All seven checks pass** at `(nB,T,Yp)=(0.16,10,0.1)`:

- For Cs, Co and b, independently re-solve a scalar temperature root that keeps entropy density and species densities fixed as each coupling changes. Finite differences of the resulting total energy reproduce the generalized force X, with largest scaled error **1.93e-8**. This tests parameter work on an independently formulated isentropic path rather than simply rearranging the supplied Jacobian rows.
- The same entropy-root temperatures reproduce `dT/dlambda|s,n=−s_lambda/s_T`, with largest scaled error **3.72e-8**.
- Reparameterizing the vector coupling as `Co=g²` at fixed arbitrary meson scale and re-solving pressure reproduces the chain derivative `dP/dg=2g dP/dCo` within **1.06e-12 relative**. This explicitly tests why raw-C sensitivities must not be labeled raw-g sensitivities.

These probes share the forward physical model; they establish numerical/thermodynamic consistency, not external many-body theory agreement. The recorded author 504-check suite supplies broader selected-state finite-difference, primal/branch, JAX and source-partial evidence. Small residuals should not be reinterpreted as physical-model error bars.

## Unresolved gates and safe next use

The implicit rule is first order, callback-based and conditional on a smooth locally admissible branch. Positive local chemical Hessians and scalar curvature do not establish global equilibrium or coexistence selection; the baseline negative quartic scalar coefficient retains the known global-potential limitation. A finite-difference perturbation can remain on a locally stable/metastable branch without proving that branch minimizes the global thermodynamic potential.

The host oracle reports diagnostics, while the staged interface returns NaNs on caught numerical/domain failures. Those sentinels are not a differentiable phase-selection prescription. Nested/higher-order AD remains unsupported. The physical EOS family is not recalibrated under coupling perturbations, and no causality/domain/global-stability audit for broad excursions is supplied.

Accepted next use: small local coupling sensitivities of the same uniform thermodynamic/quasiparticle model, followed by a **separately matched** weak-response calculation. The existing nonrelativistic common-mass optical response cannot become a relativistic GM1 response merely by inserting a Dirac mass and vector offsets. A compatible dispersion/current contraction, source composition/energy bookkeeping, and external finite-T forward comparison remain production gates. No full hydro, stellar-structure derivative, dynamical phase conversion or physical uncertainty inference follows from the current leaf.
