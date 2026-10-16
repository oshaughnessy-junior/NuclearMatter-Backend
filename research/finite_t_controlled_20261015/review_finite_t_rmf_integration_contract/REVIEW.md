# Independent finite-T GM1 integration-contract review

**ACCEPTED as a scoped implementation design, not an implemented or externally validated EOS.** There are **166 passing static provenance/saved-data/algebra checks and no failed gates or contract-blocking physics/API defects**. No EOS, root solver, AD, rate, native library, install or production calculation was executed by this review.

Frozen CONTRACT SHA256: `f225acd8434612b9d26df3e4e92c167f8c67fcaa0125c9ad485c719bd421d07f`; MANIFEST `2f07a52db20d979ca42f71534f40d011e69b421e240cce21be95269292a23803`; CHECKPOINT `e41f433c522d6d8a6bd5416411a91d8e02743c17cd8e4e37f418bc38273a5bca`. All seven manifest-covered author files and 45 pinned source/evidence paths remain byte-identical.

## Equations and independent algebra

The proposal consistently defines exact rounded NMB GM1, common bare M=939 MeV, Cs/Co/Cr=(11.785,7.148,4.410) fm², b=.002947, c=−.001070, fixed parameters, no extra omega-rho/vector couplings, no vacuum sea and explicit thermal antibaryons. The internal energy variables are fm^-1 and internal energy densities fm^-4. Multiplication by hbarc converts e/P/f and the two-charge chemical Hessian to their declared MeV units; Cv has identical fm^-3 dimensions for d(e_internal)/dt and d(e_MeV)/dT_MeV. A future parameter API cannot silently accept raw g, C=(g/m_meson)^2, source-specific nonlinear coefficients or a recalibrated GM1 label interchangeably.

For a single gas grand potential Omega(t,nu,m), n=−Omega_nu, rho_s=Omega_m, s=−Omega_t imply n_m=−rho_s,nu and s_nu=n_t. At fixed net number the Legendre transform yields

`rho_s,m|n = Q+B²/chi`, `rho_s,t|n = rho_s,t|nu − B*n_t/chi`.

With m=M−Phi, stationarity is U'−sum(rho_s)=0. Direct elimination gives

`D=U''+sum(Q+B²/chi)`, `Phi_t=K/D`, `K=sum(rho_s,t−B*n_t/chi)`,

`Cv=t*[sum(s_t−n_t²/chi)+K²/D]`,

`H=diag(1/chi)−aa^T/D+[[Co+Cr/4,Co−Cr/4],[Co−Cr/4,Co+Cr/4]]`.

These agree with the contract, including the **positive scalar-relaxation term in Cv** and **negative scalar contribution to H**. `audit.py` independently constructs 12 different consistent local quadratic grand-potential jets and solves the full three-constraint system by direct dense linear algebra. It checks Phi_t, Cv, full H, symmetry and vector mixed derivatives against the Schur expressions. These are algebraic sign/normalization checks, not physical gas points, EOS calculations, numerical phase validation or external-code agreement.

Thermal baryon number is the particle-minus-antiparticle net density; scalar density and kinetic energy add both. Pair Cv at fixed NET number is not the sum of two fixed-particle Cv values. The proposed private pair/mass/scalar-moment path is necessary: the current public positive-particle utility cannot safely represent all negligible antiparticle components or supply field-relaxed derivatives. Treating its underflow exception as rejection of an otherwise resolved nuclear state would be wrong. The contract explicitly prohibits that shortcut and requires separately controlled representation diagnostics.

The vector energy is Co*nB²/2+Cr*(np−nn)²/8. The single-particle vector-energy sum is twice this quantity; the proposed energy decomposition and rho halves are correct. The nonlinear b term uses the common bare M, not mD. Physical neutron/proton mass splitting, an extra vacuum Q, or a chosen weak-current mass is not part of this common-mass EOS.

## Saved forward and actual cold API audit

The three copied rows exactly match their pinned JSON pointers. Independent arithmetic checks their common-mass relation, scalar potential, vector fields/energy, kinetic-to-total chemical shifts, f=e−Ts, Euler relation and decomposed energy. This verifies saved-data provenance and dimensional conventions only. It does **not** execute a new EOS, verify their numerical quadrature or establish their Hessian stability; the saved rows' phase labels remain unassessed.

AST/source inspection verifies the actual cold GM1 literal values and RMFTEOS symmetric/composition/eos_table/landau_masses APIs. The current saved cold source is byte-identical to the original scientific snapshot. Its cold lepton/beta-equilibrium wrapper, pressure shortcut, barotropic tables, optional Lambda_v and saturation-to-coupling map are not interchangeable with independent (T,nn,np) nuclear matter. The contract correctly leaves those APIs untouched and rejects unsupported parameter extensions. The first-upcrossing cold routine's existing no-crossing argmax behavior and tiny-density clamp are source facts, not numerical patterns to inherit into the new CPU solver.

The primary O2scl source was inspected directly from pinned preserved bytes: spin2, interacting effective mass, pair_mu, rho halves, nonlinear bare-mass coefficient and field-energy signs match the reduced convention. There was no new remote-source fetch or native execution. Its source equations plus independently solved same-equation campaign reductions do not close the native external gate.

## Stability and branch admission

The full nuclear two-charge Hessian is mandatory. The preserved (.06 fm^-3,1 MeV,Yp=.1) example has positive fixed-Y mechanical projection and positive canonical D but a negative Hessian eigenvalue; this is independently verified from the frozen saved classification. It must be rejected by the physical API and may be returned only by an explicitly unadmitted diagnostic API. Positive composition curvature or positive fixed-Y compressibility cannot replace this test. Positive H and D remain LOCAL restricted nuclear criteria, not a global Gibbs/Coulomb/charge-neutral-matter phase certificate.

The cold-connected finite positive-mass branch is a named effective-model restriction because the negative quartic coefficient makes the unrestricted potential unbounded. A numerical seed must not change model/units or silently choose a competing root. The proposed crossing/residual/conditioning/branch-connection checks and explicit ambiguity/no-root/marginal errors are adequate design requirements. They still need actual independent nested-root and continuation tests in an implementation.

## Open implementation gates and clarifications

No contract repair is requested. Before accepting a candidate, the following remain real blockers to the corresponding stronger claims:

- Implement and independently test net-pair/scalar/mass/thermal moments, branch selection, two-charge H, Cv and strict numerical admission. Lock supported T/nB/Yp ranges or an explicit per-state qualification policy, tail/residual/error scales, conditioning/H/D margins and response controls **before** collecting candidate validation results. Selected saved rows alone cannot qualify an arbitrary positive-input domain.
- Keep internal derivatives explicitly in t=T/hbarc and public derivatives in T_MeV. The contract's shorthand thermodynamic differentials using T should be read after public-unit conversion; code and response tables must never differentiate an internal fm^-4 potential directly against MeV and omit hbarc. This is a nonblocking notation clarification, not a changed physical equation.
- Perform genuine independent forward/root/thermodynamic tests before derivative claims, including pair-sensitive artificial gas cases and tiny-component representation cases. Recompute full stability at exact saved/heldout states rather than inheriting a stable label from neighboring examples.
- Exercise exact installed wheel/import/example behavior, unchanged cold APIs, supported/minimum dependency versions and normal admitted CI. Current NumPy2.3.5 is outside the separate validation pin<2.3. Static feasibility with existing NumPy/SciPy does not establish runtime support.
- **Execute the matched native O2scl fixture** or retain external validation as OPEN. Missing headers/library/runtime are not repaired by a copied equation solve, cold beta/crust table, model recalibration or tolerance relaxation. External native agreement at selected points would still not establish empirical accuracy or global phases.

The nine-file additive CPU reference layout is feasible and appropriately avoids production EOS/TOV/cooling changes, research-directory runtime dependencies, implicit muon pairs, weak-vertex selection, parameter-family admission and higher-order/JAX claims. Later fixed-parameter first-law and parameter-work/weak-consumer interfaces are correctly distinguished. No merge-ready interacting backend or application evolution follows from this contract approval.

## Reproduction and preservation

Run `OPENBLAS_NUM_THREADS=1 <existing junior_tools Python> -I -B audit.py`. It completes in less than one second and imports no NMB or campaign solver module. All 166 checks and exact saved payload/algebra details are in `results.json`; `CHECKPOINT.json` records unchanged author/source hashes. Original missing-path and saved-row-label correction artifacts remain untouched. Exact baseline quota/account/window was checked at review start: 26%, reset1792652325. No source, dependency, git, PR, paper or production changes were made.
