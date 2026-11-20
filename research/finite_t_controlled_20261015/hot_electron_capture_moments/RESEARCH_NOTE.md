# Physical hot electron-capture and inverse total moments

**Author freeze for different independent review.** This computes actual free **p+e−↔n+nu_e** total number and species-resolved energy moments at the same T2MeV,nB1e−7fm^-3,Ye.3 pair-rich npe bath as the accepted hot neutron-beta and positron-capture leaves. Actual FD targets and all applicable holes are retained; no Maxwell/stationary-target replacement, saved-rate lookup or surrogate supplies the source. It is one complete reversible event channel, not a complete neutrino inventory, an opacity, a new joined source, beta equilibrium or hydro.

There are213 passing assertions across the main/anchor/guard/tail batches and **31 retained failed coarse assertions**. The selected fine rule is admitted only through its passing forward density, current, refinement and tail gates, pending independent review. Counts are related controls, not213 independent physical measurements.

## Exact common state and current compatibility

Physical masses are mn939.56542052,mp938.27208816,me.51099895069MeV. The weak-current mass is the **average bare nucleon mass**938.91875434MeV; it is not a Dirac/Landau effective mass. Exact matched total potentials are

```
mu_n=919.8287578100662 MeV
mu_p=916.8449303084382 MeV
mu_e= .17446333200826397 MeV
mu_nue,LTE=mu_p+mu_e−mu_n=−2.809364169619812 MeV.
```

The particle-only relativistic free n/p density inversions give nn7e−8,np3e−8fm^-3. Independent adaptive momentum integrations check those densities before event comparisons. Electrons and positrons are actual equilibrium pairs: nminus2.03428636784e−7,nplus1.73428636784e−7fm^-3 with nminus−nplus=np. Electron capture samples **the electron particle occupation**, not its net density; positrons affect the charge/EOS closure but are not an incoming species in this reaction. Muons, photons, antibaryon reactions, mean fields and interactions are absent; the existing EOS antibaryon diagnostics are retained in MATCHED_STATE.

Compatibility checks compare the complete state metadata exactly with `positron_capture_moments/MATCHED_STATE.json`, verify identical masses/GF/Vud/weak mass with both that and `hot_beta_moments`, and compare three fresh full-current matrices with the accepted tensor/form-factor implementation. They agree exactly. The common source primitive is `relativistic_capture_oracle/events.py`, SHA256c2c4afc2797b812c0585cdf09c598553e45c275ff362b64277b31350d49c4a76. Beta and PC use that same full tree current, not a gA1.26 static formula or a measured neutron-lifetime calibration. **No join has been performed:** identical matter/current is necessary but radiation distributions and independent source acceptance must also match. The existing integration leaf is `antinu_source_join/`; no nonexistent common-source or capture-moments module was assumed.

The adopted [Guo et al. current, Eqs24–27 and47](https://arxiv.org/html/2006.12051v2) has GF1.1663785e−11MeV^-2,Vud.974,gA1.2723,weak magnetism3.706 and physical-q² vector/axial/pseudoscalar form factors. Pion/vector/axial mass scales and all conventions are inherited from the pinned primitive. The physical current transfer is q=p_p−p_n for the **capture** ordering nu+n→p+e. For original EC use its literal Dirac-adjoint reciprocal, `ubar_n adjGamma(q) u_p` and `ubar_nu gamma(1−gamma5)u_e`. It uses u spinors for both neutrino/electron, not the PC antineutrino/positron v trace. There is no model/state-dependent normalization fit.

The accelerated evaluator is the published invariant polynomial in the same free convention, adapted openly from `review_pc_moments/reference.py` with n→p capture ordering and the **neutrino** VA/AF signs. It is a full tree polynomial, not a fitted function. Four fresh reciprocal vacuum amplitudes and a complete moving-target thermal pilot compare it to the literal spinor construction before using it for fine moments; maximum full-moment discrepancy1.78e−15. This is algebraic/current-method evidence within a shared physical model, not empirical nuclear validation. Coulomb/radiative/plasma corrections are omitted consistently with the hot beta/PC sources.

## Exact moving-target threshold and event phase space

An initial proton has Ep=sqrt(mp²+P²), chosen along bath z; the initial electron has total energy Ee and relative cosine c. Set D=(mn²−mp²−me²)/2 and A=Ep²−P²c². Within the explicitly guarded branch Ep*me<D,

```
Ee_min(P,c) = [Ep D + P c sqrt(D²−A me²)]/A.
```

It solves `(p_p+p_e)²=mn²`, not a smoothed step. Independent bracketed roots at12 fresh (P,c) points verify it. At proton rest Ee_min1.294084587783196MeV (TOTAL electron energy), distinct from an antineutrino inverse-beta incident threshold. Opposing moving protons can lower this electron threshold; the code does **not** impose a global stationary-target Ee cutoff in a hot gas. The numerical upper-tail truncation is not a physical support zero. The omitted ultrahigh-proton branch would need different threshold bookkeeping if Ep*me>=D; the retained selected grids are far below that condition, and the API rejects it rather than inventing support.

Use exact two-body CM final momentum pf=(s−mn²)/(2sqrt(s)), boost final n/nu four-vectors to the bath, and evaluate their actual occupation factors. The outgoing CM angle is relative to the incoming electron CM direction, with explicit azimuth. Kinetic proton quadrature uses Kp=T y²; electron quadrature uses **Ee=Ee_min(P,c)+T z²**, resolving the actual moving angular threshold without changing physics. Both are ordinary Jacobian transformations; no physical width is inserted.

The incoming spin2 measures after integrating global proton orientation and relative electron azimuth are P²dP/pi² and pe²dpe dc/(2pi²). The squared EC amplitude averages the two proton and two electron spin states, **1/4**. The accepted inverse nu+n capture primitive averages only its initial nucleon spin, so its reciprocal squared amplitude is divided by2 for EC. A neutrino has one active helicity; no extra factor2 is inserted.

With `dSigma/dOmega=Mavg4² pf/(64pi² s pi)` and Moller flux `pi sqrt(s)/(Ep Ee)`, the initial CM momentum pi cancels exactly. The code uses this cancelled form, including at threshold approach; it does not divide a tiny pi and later multiply it away. The two spin measures, flux and differential cross section form the **same invariant event measure** for forward and inverse occupations. A second inverse spin multiplicity is not applied on that common measure.

The total measure is converted by1/[hbar(hbar c)³] to fm^-3s^-1, with hbar6.582119569e−22MeVs and hbarc197.3269804MeVfm. Energy moments add oneMeV. There is **no additional neutrino DOS** after this already complete event integral. A spectral chi(E_nu), conditional spectrum or inverse rate per single incident neutrino is not supplied by these integrated radiation-distribution moments.

## Separate occupations, thermodynamics and ledgers

At each physical event p+e=n+nu, compute separately

```
F = fp fe (1−fn)(1−fnu)
B = (1−fp)(1−fe) fn fnu.
```

Empty radiation has fnu0 and B exactly absent, with a named metadata reason. Common-T FD cases evaluate fnu and its hole directly. No inverse moment is manufactured with exp(affinity/T). Per-event logF/logB and logsumexp preserve the separate positive moments; stable expm1 subtraction computes net moments and entropy. Finite log moments have ordinary representability flags; no positive source is floored to zero. The selected states are ordinarily representable; extreme underflow is not a qualified domain.

For the common-T FD closure, the independently checked identity is `logF−logB=(mu_p+mu_e−mu_n−mu_nu)/T`. LTE uses the value above; a separate nonequilibrium example has mu_nu=mu_eq−.6T. It is a prescribed radiation population, not transparent beta equilibrium. Distinct saved nonequilibrium radiation choices in other leaves are not silently joined.

For net event rate R=F−B, species sources are dn=+R,dp=−R,d(n_e−−n_e+)=−R,dn_nue=+R. Baryon number, charge and total electron lepton number therefore conserve. Material rest-inclusive energy change is En−Ep−Ee=−E_nu per forward event, and its inverse reverses sign. Separate positive n,p,e,nu energy moments and a direct material-loss moment are all stored; no extra neutron–proton Q is added to the EOS accounting. Entropy is sum(weight*(F−B)*log(F/B)); it is nonnegative for the admitted FD closure. Empty-radiation entropy is not assigned an artificial finite value.

The full phase tails include both baryon centers and the inverse chemical combination mu_n+mu_nu, not just the proton-density peak. The actual target/proton and electron upper limits are saved. This is a refinement-qualified selected truncation, not a uniform analytic tail/error certificate for the admitted input rectangle.

## Actual completed measurements and retained failures

The main three actual rules are `(proton,electron,angles,tail)`=(12,20,6,40T),(20,32,8,50T),(28,48,12,60T), with CM polar/azimuth orders tied to the listed angular order. They take.061,.291,2.154s respectively. The initial rule's proton density is high by1.12306e−5, failing the unchanged3e−8 gate; its total moments differ from the middle rule by about1.0e−5, failing30 unchanged3e−6 F/B moment gates. **All31 failed coarse assertions remain in results.json**, not relabelled as passes.

The middle/fine separate forward/backward number and all energy moments agree within3.74243e−11 at the unchanged3e−6 gate. Fine integrated proton density agrees to1.40e−14. A separate isolated **tail60T→80T** experiment at fixed fine orders changes moments by at most1.09e−14 and passes31 assertions; it completes in2.216s. Neither LTE nor a small energy ledger is used as a substitute for these density/refinement checks.

Selected fine values are

| Radiation scenario | Gross emission [fm^-3s^-1] | Gross absorption [fm^-3s^-1] | Net R [fm^-3s^-1] | Net Q_nue [MeVfm^-3s^-1] |
|---|---:|---:|---:|---:|
|empty|3.30886034824735e−7|0|3.30886034824735e−7|3.16038483675068e−6|
|LTE|3.27897971120008e−7|3.27897971120008e−7|1.16947e−21|9.75704e−21|
|mu_eq−.6T|3.29219397145512e−7|1.80679435981317e−7|1.48539961164195e−7|1.42242025776795e−6|

The emitted mean energy in the empty case is9.55127900283MeV. LTE residuals are **retained**, not forcibly zeroed; no relaxation time is inferred. Nonequilibrium entropy is8.91239766985e−8 kB fm^-3s^-1. Maximum fine four-momentum error is5.97e−13MeV and threshold-shell relative residual3.75e−16. All selected species/E/Le ledger assertions pass, with direct material loss available to avoid subtracting large rest-energy moment totals in subsequent controlled evolution.

Additional32/32 controls cover independent threshold roots, inverse vacuum spin/phase reciprocity, finite input/radiation closure guards and an absolute normalization limit. For physical vacuum events, `Sigma_nu+n = 2(pi_EC/pf_EC)² Sigma_EC`, as required by the different initial spin averages and two-body phase. A separately constructed **point-VA** current at fixed physical mass splitting and mathematical M1e4→1e5→1e6 approaches

```
Sigma_EC*v_e = GF² Vud²/(2pi) (1+3gA²) (Ee−[mn−mp])².
```

At Ee10MeV the final relative error is−2.24e−5, with the expected decreasing recoil trend; the1e−4 gate passes. This is explicitly a point-VA heavy-mass **normalization limit**, not a full form-factor thermal benchmark or a physical nucleon-mass change. Four separate full-current reciprocal anchors and the integrated spinor pilot cover the actual selected tree-current prescription.

## Evidence, operational failure and limits

`anchors.py`, `study.py`, `controls.py`, `tail_check.py` and the pure `electron_capture.py` are owned here. Unique function-module imports are used; no frozen validator/study is imported. Every required upstream hash remains unchanged. `INITIAL_UPSTREAM.json` and the final checkpoint preserve source identity and accepted prior-channel provenance.

The actual tail80 calculation saved `tail_raw.json` before its reporter failed serializing a NumPy Boolean. Exact initial reporter and error are preserved as `initial_tail_reporting_*`. `recover_tail_report.py` recovered only the gates from that completed raw source, with Python scalar conversion; it did not rerun or invent a numerical integral. The ordinary tail reporter is fixed for future fresh-copy reproduction. No equation or tolerance changed. The coarse31 failures remain independent of that reporting error.

The main batch has144 passing assertions and31 coarse failures; controls32, tail31 and compatibility6 pass, giving213 passing assertions across related evidence. Reproduce in a fresh owned copy to preserve frozen reports; per experiment the110s work guard remains below the campaign120s cap. No install, production/paper/git edit or heavy build occurred.

The admitted input guards (pair-rich/dilute baryons,T.2..4MeV) are **not** a numerical domain certificate. Only the selected matched bath, stated quadratures/closures and reference limits are evidenced. No mean-field insertion, correlated/modified Urca, medium-dependent mass, radiative/Coulomb correction, general radiation spectrum, opacity or full inventory is qualified. A different reviewer should independently check phase-space/current/spin normalization and one new coordinate or state before a common-bath source join or material-feedback application is accepted. The detailed physical methods remain the path for changed-state work.
