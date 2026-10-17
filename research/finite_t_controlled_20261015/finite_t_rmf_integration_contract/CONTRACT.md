# Next installable finite-temperature RMF reference: scoped integration contract

**Recommendation: prepare a separately imported scalar CPU reference for the exact rounded NMB GM1 uniform nuclear model, with stationarity-aware responses and mandatory local two-charge stability admission. Do not extend the cold RMF/TOV/cooling APIs or promote the existing research solver by copying it unchanged.** Implementation can be scoped now; this audit writes no solver or production source. Independent implementation review, supported-runtime/CI and the matched external-software forward gate remain open. A draft reference candidate is not a merge-ready interacting matter backend.

This audit reads frozen source and accepted campaign evidence. It runs no EOS, AD, rate or native-library calculation. Three saved forward rows are copied with their JSON locations/hashes into `SAVED_FORWARD_AUDIT.json`; they remain internal/same-equation evidence. `SOURCE_PIN.json` records45 inspected source/evidence paths. No frozen bytes changed.

## 1. Actual package surface and instructions

The inspected real package mirror is NMB `ffb5cc72949f2e5401251847aef4c9c7a98dde78`; the original scientific snapshot is `25e68182f726e5bd9c0b372c9a667536940fbd07`. Cold `src/nmbackend/eos/rmft.py` is byte-identical between the saved current package and that scientific snapshot, SHA2564a717efc9ae79895876e2529c2d2f0325cd5adfb36251bf3c60ea1eac0cc377c, Git blob1edba14f7422f37f1ae6e4eb58167dc2c2140061. This is a pinned-source audit, not a claim that live remote main was refreshed in this batch; a later branch must recheck head and preserve unrelated workflow/other contributors' changes.

The saved package README describes a JAX EOS/TOV inference toolkit and directs architectural/provenance readers to architecture/attribution docs. The earlier owning-repository audit records404 for AGENTS at root and relevant source/reference/test/docs/example paths, and no applicable local ancestor file at its checked locations. That is bounded absence evidence, not permission to assume future instructions absent. Genesis instructions govern this new research-only audit. Re-read applicable instructions at the actual implementation checkout.

Actual cold APIs include `RMFTEOS.symmetric(n_B,params)`, `composition(params)`, `eos_table(params)`, `landau_masses(params,comp)`, module-level `landau_masses`, `symmetry_energy`, `saturation_properties` and `couplings_from_saturation`. The defaults use common M939MeV,Cs11.785,Co7.148,Cr4.410fm²,b.002947,c−.001070. The cold model also supports optional omega–rho coupling Lambda_v and maps empirical saturation inputs to couplings. Those extensions are NOT in the accepted finite-T GM1 model and must reject in the first thermal slice rather than being partially ported.

Cold field/gas internals use energies/momenta fm^-1,densities fm^-3,energies fm^-4; public nuclear outputs convert by HBARC to MeV/fm³. Cold `lepton_sector` separately takes charge and imposes e/muon chemical equilibrium; composition is cold beta-equilibrium on a grid. `EOSTable` is barotropic, with crust/tail processing; it cannot represent independent T,Yp. Existing lepton/cooling JAX/Hessian/source-parity contracts stay unchanged. In particular the cold private pressure shortcut `mu_n*nB−e` with leptons assumes beta equilibrium and charge neutrality; it is not the pressure formula for independent Yp. The nuclear-only cold comparison must use `mu_n*nn+mu_p*np−e` or independently calculated pressure, without the cold lepton wrapper.

The cold scalar helper selects a first grid upcrossing and bisects; the finite-T research reference explicitly errors when its own cold scan finds no crossing, then uses a warm three-constraint least-squares solve. The production proposal needs explicit crossing existence, scaled residual and branch checks; neither a selected grid index nor nonlinear solver success alone is an admission rule. This is an integration design boundary, not a source-fix request for the cold implementation.

## 2. First model and state contract

First public model: EXACT rounded GM1 coefficients above, common bare M939MeV, uniform spin-two n/p Hartree matter, thermal baryon/antibaryon occupations retained, zero Dirac-vacuum sea, zero extra vector/cross couplings. Inputs `(T,nB,Yp)` specify independent positive NET species densities nn=(1−Yp)nB,np=Yp*nB. No beta equilibrium, local/global charge-neutrality rule, leptons, photons, clusters, nuclei, pairing, neutrinos or crust is implicit. Exact endpoints Yp0/1 remain unsupported until zero-species/pair branches are separately qualified.

Keep nuclear output separate from the accepted pair-lepton modules. A later wrapper may impose Yp=Ye+Ymu and choose which lepton flavors are PRESENT. Zero net muon density is not absence of muon pairs. Any lepton-stabilized/charge-neutral Hessian must state its constrained variables separately; it is not the nuclear Hessian with a new label.

Public quantities: T,fields,mD,kinetic and total mu in MeV; n,scalar density,s,Cv in fm^-3,kB1; e,P,f in MeVfm^-3; H=dmu_i/dn_j in MeVfm³. Positive mD is a Dirac mass; density-defined Landau diagnostics are sqrt(kFi²+mD²), not a universal finite-T transport mass. Output `kinetic_mu` means nu including mD, before vector shifts, not mu−M and not the scalar field. Existing free CPU baryon defaults939.565421/938.272088MeV, campaign free939.56542052/938.27208816 and common GM1939 are three explicit conventions. Do not silently mix them. Retain HBARC197.3269804MeVfm.

## 3. Potential, stationarity and double-counting contract

These equations are in INTERNAL fm units: M=939/HBARC,m=M−Phi,t=T/HBARC,nu_i=mu_i−Vi, all energies/fields fm^-1. Let n3=np−nn,

```
U(Phi)=Phi²/(2Cs)+b*M*Phi³/3+c*Phi⁴/4
W=Co*nB; R=Cr*n3/2
Vn=W−R/2; Vp=W+R/2
Vfield=W²/(2Co)+R²/(2Cr)=Co*nB²/2+Cr*n3²/8.
```

The common M in these internal expressions is939/HBARC. Do not insert mD into the nonlinear b*M coefficient. With E*=sqrt(p²+m²),f_i=expit((nu_i−E*)/t),fbar_i=expit((−nu_i−E*)/t), spin-two measure p²dp/pi²,

```
n_i=∫(f_i−fbar_i)
rhos_i=∫(m/E*)*(f_i+fbar_i)
ekin_i=∫E* (f_i+fbar_i)
Pkin_i=∫p²/(3E*)*(f_i+fbar_i)
s_i=∫[h(f_i)+h(fbar_i)], h(f)=−f ln f−(1−f)ln(1−f).
```

No vacuum or arbitrary energy offset is added. At fixed T,nn,np the reduced canonical potential is

`f=Σ fkin_i(T,n_i,m)+U(Phi)+Vfield`,

and scalar stationarity gives `U'(Phi)=rhos_n+rhos_p`. Solve this plus BOTH net number constraints for Phi,nu_n,nu_p. Return independent kinetic/field decompositions,

```
e=Σekin+U+Vfield; P=ΣPkin−U+Vfield
s=Σs_i; f=e−t*s
mu_n=nu_n+Vn; mu_p=nu_p+Vp.
```

Convert e,P,f by HBARC; s and number/scalar densities stay fm^-3. Pressure is independently integrated, not defined by Euler: e+P=T*s+mu_n*nn+mu_p*np is a gate. The quasiparticle vector-energy sum Vn*nn+Vp*np equals2Vfield. Therefore summing single-particle total energies and then adding U+Vfield double counts vector energy; the equivalent correct expression is `equasiparticles+U−Vfield`.

At fixed couplings the envelope theorem yields df=−s dT+Σmu_i dn_i. It does NOT imply that an ideal fixed-mass Cv is the interacting Cv: second derivatives include field relaxation. For density-dependent couplings, additional rearrangement terms are required; the proposed fixed-coupling formulas do not supply them. Importing DDRMF parameters, a state-dependent external effective mass, nonzero Lambda_v or another GM1-labelled calibration must reject until the corresponding potential/Jacobian and forward gates are derived.

## 4. What the reviewed particle utility can and cannot supply

`reference.finite_t_baryons` is a useful independently reviewed FIXED-mass particle-gas primitive, with positive particle density, actual FD moments, direct kinetic energy, susceptibility and fixed-particle Cv. It is not an RMF solver or net baryon pair gas.

At fixed m, particle/antiparticle bulk moments could be formed from mu=+nu and−nu evaluations, subtracting NUMBER and adding energy/P/s. But the public particle utility rejects unrepresented positive number/susceptibility; an exponentially tiny antiparticle contribution can underflow while the RMF state is perfectly representable. Do not call both public particle roots and then mistake an exception/zero for physical exclusion. The first RMF slice needs an explicit private pair-moment path retaining particle/antiparticle bookkeeping, component normalization/log diagnostics and rigorous representation policy. It must add the missing scalar-density and mass/temperature response integrals, not infer them from a rounded rest-dominated energy subtraction. Independent direct scalar integration is required even if `(ekin−3Pkin)/m` is also checked.

Nor is pair Cv at FIXED NET number the sum of particle and antiparticle Cv at independently fixed populations. At fixed m, `Cv_pair=t*(s_t−n_t²/chi)` uses the common pair chemical potential and net constraint. In the interacting case there is an additional scalar-relaxation term. Pair-sensitive artificial ideal-gas states should test these distinctions even when the selected dense GM1 examples have tiny antiparticle abundances.

Reuse stable quadrature/occupation/root techniques and compatible internal moment code after new tests; preserve the existing public utility unchanged. Simply substituting mD(T,n,Y) into ParticleState and returning its fixed-mass fields would silently give wrong thermodynamic derivatives and misidentify particle density as net. A finite-T RMF port is therefore a new solver/response slice, not an adapter around ParticleState alone.

## 5. Analytic local response and stability required by admission

Use internal fm units and distinguish response B_i from scalar coupling b:

```
chi_i=∂n_i/∂nu_i|t,m >0
B_i=∂rhos_i/∂nu_i|t,m =−∂n_i/∂m|t,nu
Q_i=∂rhos_i/∂m|t,nu
D=U''+Σ(Q_i+B_i²/chi_i); a_i=B_i/chi_i.
```

The canonical scalar denominator D has fm^-2 units. The three-constraint Jacobian in z=(Phi,nu_n,nu_p) is

```
Fz=[[U''+Qn+Qp,−Bn,−Bp],
    [Bn,chi_n,0],
    [Bp,0,chi_p]].
```

Solve Fz dz=−Fx dx with analytic distribution moments; do not differentiate adaptive panel choices or least-squares iterations. A root search may use a numerical search Jacobian without turning the returned physical derivative into a finite-difference rule. Report scaled conditioning/residuals and reject singular/ill-conditioned response instead of regularizing away a spinodal.

With n_t,rhos_t,s_t evaluated at fixed m,nu and K=Σ(rhos_t−B_i*n_t/chi_i),

```
dPhi/dt|nn,np=K/D
Cv|nB,Yp=t*[Σ(s_t−n_t²/chi_i)+K²/D]
H_ij=delta_ij/chi_i−a_i*a_j/D+dVi/dn_j
(dV/dn)=[[Co+Cr/4,Co−Cr/4],[Co−Cr/4,Co+Cr/4]].
```

Cv has the same fm^-3 units whether returned as de_MeV/dT_MeV or de_internal/dt. Multiply internal H by HBARC. Entropy derivatives give de=T ds+Σmu_i dn_i,df=−s dT+Σmu_i dn_i,dP=s dT+Σn_i dmu_i at fixed parameters. These are CPU analytic response fields; they do not promise JAX gradients through the scalar API.

Admission must separately require actual root/number/scalar residuals, positive canonical D, acceptable Jacobian conditioning, positive Cv, near-symmetric H and a resolved positive minimum eigenvalue of the FULL nuclear two-charge Hessian. A solver returning a stable scalar root is insufficient. For v=(1−Yp,Yp),dP/dn|T,Y=nB*vᵀHv is only one projection. The saved counterexample(.06,1,.1) has positive dP/dn4.101856MeV and positive D.101834fm^-2 but H minimum−280.089890MeVfm³. A fixed-n composition curvature is also positive there. Default physical-state admission must reject it; a diagnostic-only call can return the explicitly labelled spinodal branch.

Even positive H is only LOCAL nuclear convexity, not a global two-charge Gibbs minimum, crust transition or matter-plus-electron/Coulomb stability. Local neutrality adding ideal electrons contributes inverse electron susceptibility to Hpp; trapped flavor constraints give other projections. Global coexistence needs matching both chemical potentials and pressure, lever rules in BOTH charges and a free-energy comparison, excluding the trivial identical-phase root. None is implemented by flattening a fixed-Y pressure curve. Negative c makes the unrestricted scalar potential globally unbounded; the finite positive-mass branch is an explicitly restricted effective-model branch.

## 6. Proposed minimal API and branch behavior

Use explicit import `nmbackend.reference.finite_t_rmf`, out of EOS/TOV hot paths. Proposed first signatures:

```
state_density(temperature_mev, baryon_density_fm3, proton_fraction,
              *, controls=SolveControls(), initial=None) -> UniformRMFState
diagnose_density(temperature_mev, baryon_density_fm3, proton_fraction,
                 *, controls=SolveControls(), initial=None) -> UniformRMFDiagnostics
```

First public implementation fixes the exact rounded GM1 parameter record. Return immutable parameter metadata; reject unknown coupling/mass/cross-coupling overrides rather than silently admitting an untested model family. A later explicit constant-coupling API can reuse the reviewed parameter-response work after its own branch/calibration gates. Do not take the cold `couplings_from_saturation` mapping as already qualified at finite T or differentiate that mapping implicitly.

`UniformRMFState` includes input state/species convention, Phi/W/R,mD,kinetic/total mu, actual nn/np and antiparticle diagnostics, e/P/f/s/Cv and kinetic/scalar/vector energy decomposition, H with neutron/proton order, D, scaled root/Jacobian/antisymmetry diagnostics, branch policy and local/global phase labels. Response-unit tables must be explicit. Diagnostic calls may expose stationary spinodal values but must mark them unadmitted and never be silently used by evolution. Default `state_density` raises a clear physical-admission error on spinodal/marginal states, and ArithmeticError for unresolved numerical response. Invalid/nonfinite/Boolean scalar inputs and unsupported zero species/T reject; cold handling remains the old API or an internal cold initializer, not a new replacement.

Branch policy: cold first actual scalar upcrossing in0<Phi<.985M, verified bracket/refined crossing and positive local canonical curvature; finite-T continuation from that branch with actual source residuals. A supplied initial state must match model identity/units and is only a numerical seed. Validate branch connection by controlled continuation/independent nested fixed-density scalar-root scans on test cases. No-root, multiple-candidate/ambiguous connection, mD boundary, scalar-conditioning or marginal-H cases must produce explicit failure/diagnostic status, not the first root returned by a multidimensional solver. Comparing roots' free energies only within a documented finite branch interval does not prove a global minimum.

Production candidate cannot require campaign directories, JSON acceptance manifests, mutable global coupling dictionaries or data files outside its installed wheel. Embed no application rate cache. Source/review provenance belongs in documentation/fixtures. Importing the new reference must not execute validation drivers or JAX compilation.

## 7. Nine-file implementation proposal

| Path | Responsibility |
|---|---|
|src/nmbackend/reference/finite_t_rmf.py|Immutable GM1 state/controls, strict APIs, branch solve, units and physical admission|
|src/nmbackend/reference/_rmf_moments.py|Private spin-two particle/antiparticle bulk/scalar/thermal/mass moments with representation diagnostics|
|src/nmbackend/reference/_rmf_response.py|Analytic constraint response,Cv,H,conditioning and local classification|
|tests/test_finite_t_rmf_forward.py|Independent momentum/grand-potential/root oracle, saved provenance and cold/units/branch regression|
|tests/test_finite_t_rmf_response.py|Mass/temperature moments, re-solved FD ladders,H/Cv/Maxwell and spinodal/invalid-input gates|
|tests/data/finite_t_gm1_reference.json|Small explicitly INTERNAL-only selected rows with exact model/source metadata; external payload only when actually executed|
|examples/finite_t_rmf.py|Two stable nuclear states plus explicit diagnostic spinodal example, named units, no neutrality/beta assumption|
|docs/finite_temperature_rmf.md|Model/equations/species/branch/units/evidence and external/phase/AD limitations|
|README.md|Short explicit-import CPU-reference link and example command|

If the CPU component PRs are still drafts, do not import modules absent from the actual branch: either declare an explicit reviewed prerequisite or keep the necessary new pair moments private to this nine-file slice. No runtime research-directory dependency is allowed.

No eager top-level exports, EOSTable extension, shared cold lepton change, cold RMFT rewrite, cooling replacement, weak kernel integration, dependency change or new CI workflow. Existing NumPy/SciPy suffice; match Python>=3.9 and declared SciPy>=1.7 APIs, with actual minimum/supported execution and normal admitted CI before calling those runtimes supported. Current research NumPy2.3.5 is outside validation pin<2.3; do not inherit environment qualification from local success.

## 8. Forward-before-response implementation gates

Predeclare test states/controls/gates before a candidate's numerical results. These are proposed candidate gates, not tests executed in this audit:

1. Exact mass/couplings/rho halves/HBARC/source-record checks; zero vacuum convention and particle/anti signs. Nuclear-only cold T→0 comparison against archived original formulas, not a beta-equilibrated total EOS or differently calibrated GM1 label.
2. Forward e/P/s/mD/nu/mu/fields/decomposition/net roots at saved(.16,10,.1),(.24,20,.3),(.19,14,.22), plus fresh predeclared heldouts.2e−9 relative component gate and2e−9 normalized scalar/number residuals are useful existing anchors; retain dimensional floors separately for near-zero quantities. Tiny antiparticle relative accuracy is not inferred from bulk agreement.
3. Independent ORIGINAL momentum grand-potential integration and nested density/scalar roots. Different coordinate, quadrature, solver seed and refined tails; do not merely call the candidate twice or reuse its converged field as the independent root. Tail60→80/100 and tolerance variation tested separately from identities. Saved fixtures alone are regression, not independent validation.
4. Differentiate original gas integrals at fixed nu,m before full RMF responses. Include pair-sensitive ideal-gas cases, scalar trace identity and mass derivative identities. Require correct pair Cv and field-relaxation term; cannot pass only in low-antiparticle limits.
5. After matching primal values, re-solve T,nB,Yp perturbations with decreasing steps/Richardson for all response outputs. Compare free-energy derivatives, entropy/Cv, pressure derivatives, cross-species Maxwell symmetry, H against separate chemical finite differences and free-energy second differences. Existing research H2e−6 matrix-scale and potential second-difference1e−4 gates are starting targets, not broad domain guarantees.
6. Default rejects saved mechanically-positive spinodals; diagnostic route reproduces signs/eigenvectors. Stable high-density examples need positive H margin as well as D/root checks. Explicit cases for no bracket, nonconvergence, ambiguous branch/seed, near singular scalar response, asymmetric H, zero/endpoint species and unrepresented moments. No solver failures converted to physical zeros.
7. Stable thermal increments/Sommerfeld limits at low T with resolved error accounting. Do not subtract O(100MeVfm^-3) full energies to certify arbitrarily tiny heat. Retain the source-defined free-energy-derived Sommerfeld product only as optional separately qualified asymptotics; default detailed physical quadrature/root remains.
8. Real package import/wheel/example and unchanged cold RMFT/lepton/unit regressions. Supported/minimum versions and repository CI with normal budget admission remain real gates, not a static AST claim.

## 9. Saved forward audit and the exact unresolved external gate

No new physical values were generated. Saved internal/equation-reduction values:

|nB,T,Yp|mD [MeV]|mu_n / mu_p [MeV]|e / P [MeVfm^-3]|s [fm^-3]|
|---|---:|---|---|---:|
|.16,10,.1|652.158479454|978.526263933 /861.822728677|151.793127078 /4.23032933803|.132651075091|
|.24,20,.3|546.351971809|996.263371238 /911.901057128|227.460230399 /11.4593230222|.294521546992|
|.19,14,.22|609.382359819|978.162096382 /883.994982074|179.247482347 /5.48511913250|.201284896059|

First two rows come from `external_gm1_forward/results.json`'s INTERNAL fields independently compared to a direct momentum reduction; third is the separately reviewed rapidity holdout. The author104 and different-review64 numerical checks remain SAME-EQUATION, not native O2scl output. The exact reduced model is grounded in pinned primary [O2scl RMF source](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/eos/eos_had_rmf.cpp) and [pair implementation](https://github.com/awsteiner/o2scl/blob/d68532ff31e59dac65844c3d387e1388cb71a770/src/part/fermion_rel.h), with bytes preserved/hash-verified here. A fresh raw-URL browse attempt failed; local pinned primary bytes, not an unread re-fetch, support this audit.

**External native forward gate remains OPEN.** The native fixture has not compiled/run: missing `o2scl/eos_had_rmf.h`; no matching installed library or recorded image was available in the bounded host inventory. A copied equation solve, CompOSE cold beta/crust GM1 table or Zenodo mirror cannot close that gate.

Concrete closeout: provide a complete independently built O2scl revisiond68532ff31e59dac65844c3d387e1388cb71a770 or a separately audited external runtime; record compiler/library/source digests, exact HBARC, solver and gas tolerances. Set BOTH bare fermion masses and nonlinear mass scale939/HBARC,spin2,inc_rest_mass,true; interacting RMF flags,cs=sqrt(11.785),cw=sqrt(7.148),cr=sqrt(4.410),b/c exact,all extra couplings zero,zm_modefalse,thermal pairs/no vacuum,no leptons/photons/beta constraint. Native meson masses/fields must be explicit; only ratios enter this uniform no-extra-coupling model. Use its density-based finite-T solver without seeding the internal converged fields. Require successful native return, density/scalar/vector residuals and branch, independently refined native numerical controls, then compare all forward fields,mD,nu,mu,e,P,s,f at selected states BEFORE derivative interpretation. Preserve any mismatch; no coupling refit, mass switch, added energy offset or tolerance relaxation to force closure. Even closure at a few points is external code agreement, not empirical calibration/uncertainty or a global phase model.

## 10. AD, current consistency and compact products are later layers

The research state and parameter implicit-AD layers have selected independent first-order reviews. They do not provide nested callback differentiation or a production JAX Hessian. First CPU reference should expose analytic response values; later AD work must rematch primal fields and derivatives on the same branch before first JVP/VJP claims. A stiff solver differentiating an EOS derivative may need additional higher-order rules. No callback is substituted behind existing cold names.

At fixed Hamiltonian parameters, normal Gibbs-Duhem holds. If a later explicit parameter API varies Cs,Co,Cr,b,c, parameter work must be included: df=−s dT+Σmu_i dn_i+Σf_theta dtheta and dP=s dT+Σn_i dmu_i−Σf_theta dtheta. In the reduced canonical potential, internal f_Cs=−Phi²/(2Cs²),f_Co=nB²/2,f_Cr=n3²/8,f_b=M Phi³/3,f_c=Phi⁴/4. Fix M and coefficients when applying ordinary fluid evolution; do not add these parameter terms as spurious density-dependent rearrangement. The old research504-check parameter validation and independent source partials are a later reuse gate, not automatic first-PR coverage.

An EOS-consistent weak consumer must retain En,p=sqrt(p²+mD²)+Un,p,nu_i=mu_i−Ui and kinetic transfer omega*=omega−(Ub−Ua). Total affinities use TOTAL chemical potentials without a second potential shift. The selected campaign Guo adapter distinguishes kinetic spinors/vertex q* from physical q² form factors and explicitly uses weak bare M939 independent of mD; it claims only a specified Hartree prescription. This separation agrees with the source current's distinction between effective dispersions and bare-mass normalization ([Guo et al.,Eqs23–28](https://arxiv.org/html/2006.12051v2)). The EOS itself must not select an in-medium weak vertex or add vacuum1.293MeV to the common-mass model. Nuclear EOS tests do not validate RPA/correlations or a conserving interacting response. Integrated GM1 capture evidence remains a separate reviewed restricted channel, not full Urca inventory.

For optional compact approximations, fit the free-energy potential with exact coordinates/units/branch and a separately fixed coefficient lock, then derive e/s/P/mu consistently. Reserve blocked fresh holdouts and test H/stability, increment errors and any near-zero cancellation; reject outside qualified state/phase domains and retain detailed actual quadrature fallback. Fitting P,e,mu independently or using a fixed thermal index can violate stationarity/Maxwell relations. Existing selected potential/affinity fits are research convenience evidence, not a replacement for detailed default physics or proof of AD fidelity. No fit is needed to implement this first slice.

## Readiness and next sequence

1. Agree on the nuclear-only exact-GM1 CPU/strict-stability contract; leave old APIs untouched.
2. Author the nine-file candidate plus preregistered independent forward/response/branch tests in an owned real-package mirror; no campaign runtime imports.
3. Different numerical/physics/package reviewers gate exact-source outputs, failures and scope; exercise admitted dependency versions/CI.
4. Execute the matched independent native forward fixture, or keep that external blocker explicit. A prepared draft candidate is not externally validated or generally production-qualified while this is open.
5. Only then propose application-specific composition/photon/weak wrappers, higher-order AD and larger-state approximations under separate acceptance. Preserve the detailed controlled-physics default throughout.
