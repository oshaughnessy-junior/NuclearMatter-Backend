# Isotropic spectral νe–pair-lepton scattering box

**Authored freeze pending independent review:109 selected gates PASS; SIX earlier numerical gates FAIL and remain preserved/excluded.** This implements actual finite-grid radiation feedback with a prescribed free FD electron/positron thermostat. It is a controlled closed energy box, not a converged infinite-energy transport solution or evolving material model.

## Bath and elementary physical source

The bath is fixed T=2 MeV, net electron density3×10⁻⁸ fm⁻³, with exact pair-EOS chemical potentialµe=.174463332008264 MeV. Actual ne−=2.034286367843283×10⁻⁷ and ne+=1.734286367843283×10⁻⁷ fm⁻³ are independently reconstructed by adaptive momentum integrals before evaluating any edges. This is the lepton sector of the previously specified neutral nB10⁻⁷,Ye.3 state; nucleons do not scatter or evolve here. Both e− and e+ targets with their own FD initial/final blocking are included. No muon targets, photons, plasma correlations, running couplings, radiation flavor conversion or oscillation are included.

The actual `neutrino_lepton_scattering/scatter.py` pure function module is uniquely loaded by its explicit file spec. Its accepted source and the pair/current primitives are pinned. No upstream driver, `study.py`, validator or reviewer probe is imported. Scripts should run in a fresh interpreter because frozen function modules retain legacy internal imports; this leaf does not claim parent-process module isolation.

The kernel is the accepted D(E,F,c)=dΓ/(dF dc), per incident **one-helicity νe**, with no final neutrino blocking. It already includes the outgoing-neutrino phase measure and target spin2. It is computed separately for both e± targets and both energy directions, then physically integrated over c using z=sin(theta/2), dc=4z dz. There is no kernel fit, interpolation or detailed-balance synthesis. The primary Fermi-current/target conventions are those of [Guo et al. Eqs1–13 and AppendixA](https://arxiv.org/html/2006.12051v2), with the selected electron-flavor current from the independently accepted scattering leaf. The separate current/absolute-prefactor validation in `review_neutrino_lepton` remains load-bearing; this is a new collision-operator slice, not a new EW accuracy calculation.

Before any evolution, three selected physical angular edges were compared against a separate transverse-target-momentum reduction with analytically averaged azimuth and adaptive **q-outer** integration. It does not call the author's differential kernel or its target-energy/phi rule. Fine agreement is≤1.27×10⁻¹⁰ relative. Three additional actual eight-node edges were tested later, including a high-energy near-diagonal pair; their coarse failure and new refined pass are explicitly below.

## Discrete physical equation and conservation

For positive energy quadrature nodes Ei and weights wi on[0,Emax], use

```
rho(E)=E²/[2π²(ħc)³],  Wi=rho(Ei) wi
Kij=∫dc [D_e−(Ei,Ej,c)+D_e+(Ei,Ej,c)]
aij=Wi wj Kij
Jij=aij fi(1−fj) − aji fj(1−fi), i<j.
```

**Only wj multiplies D on the outgoing side; another Ej² DOS would double-count it.** For each unordered pair, Wi dfi/dt decreases by Jij and Wj dfj/dt increases by Jij. Both neutrino occupation/hole factors are updated at every RK stage. Direct and inverse physical Kij,Kji are evaluated independently; no averaging or normalization is used to enforce KMS.

The physical reciprocal relation is

```
Kji/Kij=(Ei/Ej)² exp[(Ej−Ei)/T]
aij/aji=exp[(Ei−Ej)/T].
```

The measured grid log residual is≤5.33×10⁻¹⁵. Thus any same-T FD occupancy is stationary under this equation to the measured quadrature accuracy; it is not set to zero by code. The selected µν=−.7 MeV FD collision residual is≈9×10⁻¹⁶ of the offdiagonal event activity, and actual0.08s evolution is unchanged at floating-point precision.

Pair accumulation cancels number exactly and separately computes radiation energy gain Σ(Ej−Ei)Jij. The thermostat receives its negative. A separate dense forward-event matrix with row/column gain/loss is compared to the paired derivative at nonequilibrium states. Diagonal self-transitions cancel identically for an isotropic occupation at a single energy and are omitted from this generator. **This does not say physical D(E,E,c)=0.** The reported offdiagonal activity is a numerical scale, not a continuum total scattering frequency or count of all physical angular events.

A stable signed expm1 subtraction computes Jij. With `h(f)=−f log f−(1−f)log(1−f)`, radiation entropy is ΣWi h(fi), bath entropy change is Qbath/T, and

```
d(Srad+Sbath)/dt = Σi<j Jij log[forward_ij/backward_ji] >=0
```

up to the directly measured KMS/quadrature residual. Microscopic pair and macroscopic entropy rates are checked separately. Bath energy is a thermostat ledger; the bath's EOS or temperature is never advanced. Scattering changes neither neutrino number nor target species and produces no composition source.

Use explicit SSPRK2/Heun stages for f and exactly the same stage weights for the separately accumulated bath heat. Positivity0<f<1 is checked at the predictor and completed update, not clipped. Energy/number ledger closure is a necessary algebraic check, not independent evidence of the kernel normalization; the separate physical-edge references provide that forward evidence.

## Selected physical evolution

Initial radiation is FD with µν=−.7 MeV and Tr=1.3 or3 MeV. Then all occupations evolve freely on the fixed grid; no effective radiation temperature is imposed afterward. The final qualified eight-node edge rule uses angle40,target-energy32,phi8,target-tail80T, with0.08s and64 RK steps. The starting coarse rule was24/24/8/tail60T.

| Initial radiation | Initial radiation ε (MeV fm⁻³) | Final radiation ε | Bath heat over0.08s |
|---|---:|---:|---:|
|Tr1.3 MeV|6.37990180218e−8|7.01648315192e−8|−6.36581349741e−9|
|Tr3 MeV|2.32031687029e−6|2.08269557788e−6|+2.37621292410e−7|

Colder radiation gains energy from the thermostat; hotter radiation deposits energy into it. The approximately10% energy changes are nontrivial. Neutrino numbers remain1.58768047848×10⁻⁸ and2.55135071006×10⁻⁷ fm⁻³ respectively. Final fine-grid number residual is≤2.05×10⁻¹⁶ relative to number; combined radiation/bath energy residual is≤1.68×10⁻¹⁵ relative to exchanged energy. Occupations remain strictly interior. Total radiation-plus-bath entropy grows in both cases.

Final fine-edge16/32/64-step studies measure weighted-occupation orders2.0167 and2.0287. The32→64 heat changes are1.22×10⁻⁵ and4.15×10⁻⁵ relative, well below the percent-level spectral changes. Entropy-production integral versus actual entropy difference is time-discretized, not an exact invariant: original fine-time64 residuals are2.38×10⁻⁵ and4.03×10⁻⁶ relative and decrease with the timestep.

The PNG/PDF shows E²f at the eight actual nodes before/after; connecting lines guide the eye and do not assert a continuous spectral solution. The caption explicitly declares the closed0–24 MeV box.

## Separate edge, spectral and tail errors — retained failures

The full eight-node weighted heat/activity change under24/24/tail60→40/32/tail80 is≤3.54×10⁻⁸, with heat changes≤4.23×10⁻⁹. However, weighted convergence does **not** qualify every pointwise edge. At actual grid edge(6,7), the coarse kernel differs from the independent q/r reference by−4.4119×10⁻⁶, FAILING the unchanged3×10⁻⁶ point gate. `grid_edge_results.json` retains it. Recomputed fine edges at the same nodes pass: maximum1.52×10⁻⁹ relative in `refined_edge_grid_results.json`. The other two actual fine-edge references agree within1.11×10⁻¹¹. The coarse failure is not erased by small weighted moment errors.

At fixed Emax24,4/6/8-node evolution gives six/eight-grid bath-heat changes2.277% for cold and1.112% for hot radiation, both FAILING the1% gate. The8→12 cold heat change1.0264% also FAILS that same gate. A new bounded12→16-node comparison gives0.2557% cold and0.1097% hot, passing the unchanged1% **selected heat-observable** gate. It does not establish convergence of every spectral shape, occupation derivative or continuum observable. These are quadrature nodes, not nested energy-cell averages; only integrated observables are compared across these grids.

To isolate the physical upper bound, the eight core nodes, weights and K entries on[0,24] are kept **bitwise fixed** while new Gaussian nodes are added on[24,36] and[36,48]. Thus the tail study changes the physical event/radiation domain without resampling the core. Four→six tail-node order initially FAILS a3×10⁻⁵ hot initial-heat gate (−4.695×10⁻⁵) and an offdiagonal-activity gate(−.003286); both remain failed in `sensitivity_results.json`. A separate six→eight-node tail rule passes the same heat gates, with maximum8.97×10⁻⁶ initial and1.42×10⁻⁶ accumulated heat changes. No accepted continuum total-scattering-activity claim follows.

This tail test matters physically. With the unchanged eight-node core, increasing the closed ceiling24→48 raises hot-radiation bath heat from2.3762×10⁻⁷ to2.8393×10⁻⁷ MeV fm⁻³, about19.5%. Even36→48 changes it about1.16%. Cold-radiation tail effects are smaller. Cross-boundary forward and reverse event fluxes are calculated separately; at initial Tr3, Emax48/tail6 they are2.10058×10⁻⁸ and9.55517×10⁻⁸ fm⁻³s⁻¹ across the24MeV boundary. A closed24MeV box explicitly omits these transitions. LTE balances such exchanges, but nonequilibrium finite-box omission need not cancel. **No infinite-energy or open-range convergence is claimed.** Tail panels temporarily increase total nodes to24; each actual experiment remains≤12.2s, well inside90/120s budgets.

The six preserved failed gates are: two6→8 heat gates, one8→12 cold heat gate, two tail4→6 gates, and one coarse actual-edge gate. Further temporal passes or detailed balance do not convert them into successes.

## Reproduction and acceptance boundary

Run in fresh processes using the existing campaign Python: `study.py`, `sensitivity.py`, `followup.py`, `grid_edge_checks.py`, `refined_edge_grid.py`, and `fine_time.py`. Each writes only this leaf. Reports retain full physical matrices, states, controls, independent reference values, trajectories, per-stage heat histories and source hashes. `partial.json` checkpoints base grids; the sensitivity report checkpoints before extensions. `make_figure.py` reads saved fine results only. The initial `fine_time.py` JSON failure on a NumPy bool is preserved; cheap16/32-step runs were actually replayed after explicit scalar conversion. Existing64-step physics remained saved throughout; nothing is inferred from the failed serialization.

The bath forward root, interior-angle support, actual direct/reverse edges, selected absolute phase-space references, finite-box conservation/entropy, positivity and time convergence are qualified authored results pending independent review. Collinear endpoints and q=0 are not evaluated; interior quadrature nodes approach their regions, with explicit refinement. Actual source guards reject unresolved support and physical numerical underflow rather than supplying floors. A selected admission box is not a uniform domain accuracy certificate.

Not implemented: material temperature/composition feedback, angular anisotropy, spatial streaming, external boundary irradiation, all neutrino flavors/antiflavors, plasma response, QED/EW uncertainty or full energy-continuum convergence. This is useful as a controlled radiation redistribution experiment alongside the isolated kernel API and detailed weak/EOS references; it is not full hydro or a production GRMHD transport backend.
