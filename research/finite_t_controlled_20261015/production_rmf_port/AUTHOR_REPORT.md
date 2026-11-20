# Staged finite-temperature uniform GM1 CPU candidate

**Frozen candidate for DIFFERENT numerical/physics and packaging review; not merge/release qualified.** Exactly nine staged paths implement the accepted integration contract in a real owned NMB package mirror. Final exact-source validation is **45 pytest tests PASS (32 new candidate,13 unchanged cold/unit)**, plus65 separately recorded independent forward component comparisons and3 raw additional response checks. Counts are numerical/contract checks, not45 or113 independent physical models. No candidate, original production checkout, prior PR mirror, frozen research/review or dependency was modified outside this leaf. No Git, branch, PR, CI dispatch or native external runtime was used.

## Fixed public model and file scope

`nmbackend.reference.finite_t_rmf` is an explicitly imported scalar CPU NumPy/SciPy nuclear reference. Common bare M939MeV,HBARC197.3269804,Cs11.785,Co7.148,Cr4.410fm²,b.002947,c−.001070 are fixed. Net n/p targets are independent, thermal antibaryons retained, no Dirac vacuum sea. Leptons, photons, beta/neutrality restrictions, weak-current choices, crust, pairing, additional RMF families/Lambda_v and a phase coexistence construction are absent. Physical neutron/proton free masses are not substituted and no vacuumQ is added. Existing JAX cold RMFT/lepton/EOSTable/TOV/cooling code and exports remain unchanged.

The nine paths are the three reference modules, two tests, one small INTERNAL-only saved-data fixture, executable example, model/API document and narrow README addition. The installed implementation imports no campaign path, accepted-manifest registry, research state, mutable model dictionary, source cache or unmerged lepton/baryon utility. New private pair/scalar/mass/thermal moments are evaluated directly; the reviewed fixed-mass particle utility is not misused behind a substituted mD. Detailed physical quadrature/root is the default, with no surrogate or fit.

The public `state_density(T,nB,Yp)` requires a numerically resolved locally convex nuclear branch. `diagnose_density` returns the same canonical branch with explicit unadmitted spinodal/marginal labels. Returned Cv includes scalar relaxation and the full H is ordered(neutron,proton). T/fields/mu inMeV,n/s/Cv fm^-3,e/P/f MeVfm^-3,H MeVfm³. Gas energy called kinetic_energy_density includes Dirac effective rest energy; scalar/vector terms are separate. The canonical potential/energy/rho-half signs and double-counting boundary are documented. Antiparticle ordinary underflow gives None plus a finite log, not exact absence or a positive-density floor.

## Locked numerical and branch policy

`LOCK.json` SHA909be0ce9098060f4d6e4eab1cec73658ef7cdabedcb5c3dfff8b11eac8fa7c6 was written04:28:50.803455UTC before candidate evaluations. Numerical request bounds are T1–50MeV,nB.02–.4fm^-3,Yp.05–.5; these are not a continuous physical/accuracy envelope. Actual per-state admission additionally checks the three source constraints, canonical scalarD, row-scaled response conditioning, Cv,H symmetry and resolved positive minimum eigenvalue. Positive fixed-Y compressibility alone is insufficient.

Cold128-interval actual upcrossing, warm continuation<=5MeV steps and final48-interval nested fixed-density scalar verification retain the same branch/gates throughout. No missing crossing silently chooses index0 or clamps a species density. A supplied seed is validated but never bypasses recomputed cold-connected branch checks. The model's negative quartic and positive-mass search interval are explicit local restrictions, not global thermodynamic minimization. Default strict admission rejects the preserved(.06,1,.1) mechanically-positive nuclear spinodal; diagnostics return it unadmitted.

## Forward agreement before response interpretation

An independent original-momentum scalar quadrature computes log-partition pressure, number/scalar/energy/entropy and nested chemical/scalar Brent roots. It imports no candidate and never receives the candidate's converged field as its seed. All five independent roots were evaluated BEFORE candidate import in run_forward.py. Three saved regression states plus two predeclared new points match65 forward component checks, maximum relative discrepancy4.46e−13 (gate2e−9), maximum normalized candidate equation residual4.34e−15.

| nB [fm^-3] | T [MeV] | Yp | mD [MeV] | Cv [fm^-3] | minH [MeVfm³] | canonicalD [fm^-2] |
|---|---:|---:|---:|---:|---:|---:|
|.16|10|.1|652.158479454|.116645715700|616.378739935|.123978846535|
|.24|20|.3|546.351971809|.253057370884|894.321871101|.140406018330|
|.19|14|.22|609.382359819|.174427426787|778.047593553|.129815379890|
|.18|7|.17|622.027930982|.0976190940111|689.562550611|.128051417456|
|.28|25|.35|501.992538476|.318628507325|887.736305542|.149813094142|

The last two coordinates were locked fresh before candidate numerical results. Row-scaled Fz condition numbers19.26–47.62 are deliberately not the unscaled metric in older research AD reports. Saved fixture rows are source-labelled INTERNAL/same-equation regression data, never native O2scl output. This is implementation agreement within the fixed Hartree model, not external calibration or empirical uncertainty.

## Response, branch and package checks

All45 final pytest tests pass in bounded partitions23/2/20, taking28.55/81.44/37.47s. They cover independent pair-sensitive scalar/mass/thermal moments; fixed NET pair Cv through separately re-solved chemical roots; full field-relaxed Cv and2x2H against three re-solved FD step sizes/Richardson at both fresh states; H from independent potential second differences; free-energy/pressure/entropy derivatives; scalar-relaxation positivity; selected low-T Sommerfeld Cv; tail/tolerance changes; ordinary antiparticle underflow at a represented cold bulk state; no/ambiguous/nonfinite scalar crossings; injected nonlinear failure; strict spinodal exclusion; invalid input/control/seed/scalar representation; and13 unchanged cold-lepton/unit regressions. Each partition<120s; no installed dependency or huge grid.

A separate raw response holdout(.19,14,.22) records full plus/minus states at two steps. Richardson relative errors are6.37e−11(Cv),2.06e−11(Phi_T),5.51e−11(−s=f_T), against3e−6. This adds observable raw ladder evidence, not a new physical model. Thermodynamic response fields are not JAX/AD/Hessian-through-callback claims.

Real package verification checks71 baseline Git blobs, nine exact mirrored overlays, all actual nmbackend module origins under the owned mirror, immutable state/controls, six Python files parsed with Python3.9 grammar, actual example output and narrow README diff. The example computes two strict admitted states and one explicitly unadmitted spinodal. No namespace stubs, eager export changes or research runtime imports are used. Wheel/build/install qualification is left to the separately commissioned packaging reviewer, not inferred from import success.

## Preserved initial failures and corrections

1. Initial full-moment bracket scanning was correct at the five warm forward states but took~19s/state; its65PASS source/raw identity is preserved in pre_projection_optimization. At the valid T1 spinodal diagnostic, attempting the number bracket at nu0 asked the full positive pair gas to represent completely underflowed bulk moments and raised ArithmeticError. The original failure is retained. The final root uses exact NET nu0 symmetry only for the bracket and a lightweight same-equation number/scalar projection for the scan. Final full moments and all48 intervals/continuation/physical gates are unchanged. It never returns a zero positive gas state. Current forward runs take~1.8–2.3s/state, and all five FULL returned states are bitwise identical to the initial warm outputs. This is a selected cost measurement, not performance certification.
2. A separate adversarial scalar test found `float(10**1000)` leaked OverflowError instead of the promised ValueError; invalid finite-seed fields also needed the same scalar validation. Initial source/tests/XML/traceback are retained in pre_scalar_guard_repair and initial_scalar_guard_failure.*. Repair changes ONLY `_real` and `_seed`; all physical definitions are AST-identical, with five complete forward states bitwise unchanged. Final exact-source45tests and65forward checks were rerun after repair. No numerical tolerance, physical model or prelocked admission was relaxed.

Earlier successful test partitions and the added pair-Cv/large-scalar test expansions remain saved with their source identities. They are not double-counted as extra qualification. CURRENT_EXECUTION_SOURCE_PIN is a historical pre-guard-repair record; GUARD_REPAIR_PROOF and final verification/manifest identify the actually accepted-for-review candidate. Full before/after snapshots and failures remain auditable.

## Actual repository/runtime and remaining gates

Fresh owning-GitHub reads at immutable mainffb verify README and absence(404) of AGENTS at eight relevant root/source/reference/test/data/docs/example paths. This is bounded instruction absence, not a permanent guarantee. The source base is mainffb5cc72949f2e5401251847aef4c9c7a98dde78; cold scientific source remains25e68182f726e5bd9c0b372c9a667536940fbd07 unchanged. Package/source/review preservation hashes are recorded. No actual checkout or existing PR51/52 mirror was edited.

Executed Python3.12.12,NumPy2.3.5,SciPy1.15.2,JAX.10.1,pytest9.0.3. NumPy2.3.5 is outside the separate validation pin<2.3. Static Python3.9 grammar and use of existing NumPy/SciPy APIs do NOT execute minimum/supported runtimes. Full normal admitted CI, clean installation/dependency resolution and supported-version runs remain OPEN.

**Native external GM1 forward gate remains OPEN**: no compiled O2scl fixture or other independent software was executed. The model/couplings/sea/species/phase must match exactly when that gate is attempted; a same-equation reimplementation, cold beta/crust table, refitted GM1 variant or shifted energy zero does not close it. No global two-charge coexistence, crust composition, in-medium current, weak source, transport/evolution, higher-order AD or physical uncertainty family is qualified by this candidate. Root may propose only the separately reviewed scoped draft after numerical and packaging review; no merge/release is authorized here.
