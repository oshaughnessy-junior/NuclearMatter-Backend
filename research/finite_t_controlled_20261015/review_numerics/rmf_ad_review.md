# Independent review: first-order finite-T GM1 implicit AD

15 October 2026. Code and numerical evidence live in `probe_rmf_ad.py` and `rmf_ad_review.json`; all source hashes and the actual interpreter/JAX version are recorded. No author files, dependencies or Git state changed.

## Disposition

**Accept the first-order implicit derivative interface on the selected smooth warm local branches.** This review independently executes **55 passing checks** at two states absent from the author's AD qualification set: `(nB,T,Yp)=(0.2,15,0.35),(0.4,40,0.2)`. Both pass positive local chemical-Hessian checks before derivative interpretation. The author's 304-check report remains separate evidence; it was inspected, not rerun wholesale here.

Accepted operations are the tested float64 scalar-state primal, first JVP/VJP, jacfwd/jacrev, tested JIT combinations and first gradients of smooth downstream functions. No higher-order, coupling, phase-selection or full implicit time-integrator derivative claim follows.

## Implicit derivation and units

The 3x3 constraint matrix is correct for unknowns `(Phi,nu_n,nu_p)`. Since mD=M-Phi and `dn/dm=-drho_s/dnu`, its number rows contain **positive** scalar-column entries b_i. Its scalar row is `(U''+q_n+q_p,-b_n,-b_p)`. Explicit temperature derivatives of the scalar equation carry `-sum rho_s_T`, while number constraints carry `n_T dT-dn_target`. Solving `F_z dz=-F_x dx` therefore has the stated signs.

Internal temperature is MeV/HBARC. The input-coordinate dT vector includes 1/HBARC, while field and chemical-potential derivatives are multiplied by HBARC on return. Target-density derivatives are `(1-Yp,0,-nB)` and `(Yp,0,nB)`. Vector-field density/composition derivatives match the fixed-coupling GM1 convention.

The new analytic thermal moments are derivatives at fixed kinetic chemical potential and Dirac mass. Entropy Maxwell identities give `ds=sum(s_T dT+n_T dnu)+sum rho_s_T dPhi`; the final scalar term is positive because `s_m=-rho_s_T` and dm=-dPhi. Total thermodynamic differentials correctly include rest-energy composition terms through chemical potentials. The derived de,df,dP expressions are consistent with the primal energy convention.

The numerical least-squares implementation used for the primal may internally estimate a solver Jacobian. The returned AD rule differentiates the converged constraints using analytic distribution moments and a 3x3 linear solve; it does not differentiate or finite-difference root iterations. Reviewing those distinct roles found no hidden regularization or changed primal physics.

## Independent numerical checks

At each new warm state:

- All twelve callback primal outputs reproduce an independent direct call to the frozen state solver.
- JIT jacfwd and JIT jacrev match the analytic Jacobian.
- Separate coordinate finite-difference ladders of all twelve direct primal outputs, with Richardson extrapolation, reproduce every Jacobian row.
- Entropy density/composition derivatives satisfy Maxwell relations using independently finite-differenced total chemical potentials versus temperature.
- A nontrivial JVP/VJP dot-product identity passes.
- A JIT reverse gradient of a nonlinear downstream objective involving log pressure, entropy per baryon and squared chemical imbalance matches direct finite differences. This tests ordinary downstream differentiation without requesting a second derivative of the EOS callback.

Two additional ideal-gas thermal-moment states at T=100 MeV, including zero kinetic chemical potential, retain significant pair sensitivity. Four-point direct gas-temperature differences verify n_T,rho_s_T,s_T for all six comparisons. These are moment checks, not a qualification of full GM1 states at 100 MeV.

Invalid negative temperature, endpoint proton fraction and NaN density are rejected by strict host preflight and yield all-NaN fixed-shape values in a staged primal. Reverse-over-forward nested AD explicitly raises the expected unruled-callback error. This supplements the author's forward-over-reverse rejection test.

## Callback and qualification boundaries

`pure_callback` performs deterministic CPU root/quadrature evaluation with fixed shapes and no intended I/O. Repeated values depend on frozen module parameters and reference code; mutating the global coupling dictionary after tracing would violate that assumed frozen-model contract and is not a supported parameter mechanism. No hidden cache or application state is used.

The callback's NaN sentinel is an explicit numerical failure contract, not a substitute for a status-bearing production API. A downstream consumer must retain finite-result checks and strict preflight where diagnostics matter. The primal-only callback can evaluate a local state whose derivative conditioning is rejected by the Jacobian callback; callers must use the strict value-and-Jacobian oracle when derivative admissibility is required.

The canonical scalar curvature and 3x3 condition-number gates qualify differentiation of a selected scalar/number branch. They do not test the full species-Hessian phase stability. First-order AD remains mathematically possible inside a spinodal, and the interface does not repair that physical instability.

The returned derivative of a chemical potential supplies a physical second free-energy derivative as a separately exposed first-order output. It does not qualify applying nested AD to the free-energy callback. In particular, a stiff solver or constitutive model that differentiates EOS derivatives internally may require unsupported higher rules.

The original missing matched external finite-T GM1 forward comparison and phase/neutrality qualification remain open. This derivative review does not independently establish model accuracy, arbitrary low-T behavior, all-state conditioning or coupling gradients.
