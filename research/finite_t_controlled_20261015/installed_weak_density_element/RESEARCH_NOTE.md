# Short weak-on prescribed-density free npe element

This is a research evolution on the independently accepted actual-package free-npe binding, not a new production API. The exact reviewed constant-mass particle-baryon CPU and electron-pair CPU modules are used through the frozen binding. Source kernels, full weak-current conventions, existing acceptance records and package bytes are pinned. No production file, old binding, reviewer artifact, dependency, fitted rate or saved source table is changed. Different independent review of this NEW weak-on evolution is pending.

The author record is **104 PASS / 3 preserved coarse FAIL**. Four/eight-step individual trajectories, their finer comparison and changed-state source profiles meet the unchanged gates. The two-step approximation is excluded by its failures; global `summary.json` deliberately remains FAIL rather than erasing or relabelling them. Neither selected agreement nor conservation proves a continuous-domain or long-time solution.

## Locked problem and energy variables

Before the first scientific call, `LOCK.json` was written2026-10-16T03:54:44.931707UTC, SHA55b82078ca4aeb2195ebc000704904694e543f0c50cdefd6ff6f351b3d3f0391. Initial T2MeV,nB1e−7fm^-3,Ye.3; prescribed nB=n0 exp(kt),k=.25s^-1; duration.002s. No hydro momentum, metric or density equation is solved. Muon species are absent. Baryons are physical-mass particle-only ideal FD species, electrons and positrons are a single ideal pair flavor, with explicit campaign masses and MeV/fm units from the accepted binding. Incoming νe/antinue are empty and all emitted neutrinos escape. The network contains three named CC emission/inverse-kernel families evaluated in this empty-radiation condition: EC,positron capture and neutron beta. It is not a complete neutrino inventory.

Ideal trapped LTE photons and internal rapid EM pair rethermalization are declared material closures. They do not supply full EM plasma kinetics, photon transport or a physical timescale proof of equilibration. The same closure is used at every actual EOS solve.

With rest-inclusive volume energy e, specific energy u=e/n and actual weak moment sources R=Rbeta+RPC−REC,Q=QEC+QPC+Qbeta,

`udot=P*ndot/n²−Q/n`, `Yedot=R/n`.

The accepted binding provides Cv=eT,eY=n(hp−hn+he),en=(1−Ye)hn+Ye(hp+he),hi=mui−T*mui,T|ni. Thus the independently reported temperature derivative is

`Cv*Tdot=[(e+P)/n−en]ndot−Q−eY*R/n`.

No extra neutron–proton or electron rest-mass Q is added. Kinematic material-loss and species-energy moments already include the same rest conventions. Photons affect e/P/s/Cv and the inversion but do not directly modify a weak current at a specified material state.

## Actual time integration and independent diagnostics

Explicit midpoint at2/4/8steps evolves specific u and Ye. Each stage solves the ACTUAL EOS energy inversion for T at the current n,Ye, then recomputes all three FD physical moment integrals. No rate cache/interpolation or saved moment feedback is used. Even coincident endpoint/start states are freshly evaluated. The initial six absolute positive number/energy moments must reproduce the accepted binding before interpreting a trajectory. Every run passes that forward gate.

Separate evolved accumulators store escaped νe/antinue energies and numbers per baryon, pressure work, material entropy source, and direct weak e−/e+ creation. At every accepted endpoint a separate fresh physical source call permits composite Simpson integration of Q/n,R/n and P*ndot/n², independently of midpoint stages and the evolved accumulators. This supplies a meaningful quadrature check rather than only a telescoping identity. No direct source reuse is substituted for the independent endpoint calls.

The frozen source helper previously used a fixed NB1e−7 in diagnostic denominators. This leaf therefore imports only its accepted function definitions/controls and assembles the identical accepted event integrals locally with the ACTUAL changing nB in every density diagnostic. Its legacy state factory and fixed-density assembly are not called. All imports are under unique names in fresh `-I -B` processes; no frozen driver main is executed or rewritten.

Rest-inclusive energy and lepton conservation alone are not time-accuracy certificates. The ledger is `(uf−u0)+Eescape−W`; its scale is gross nonzero escaping thermal energy or work, never u≈939MeV/b or a nearly zero net source. Independent endpoint quadrature and refinement errors are reported separately. Physical model uncertainty is not estimated by these numerical comparisons.

## Results and retained failures

Two-step run12PASS/2FAIL,8.40s: endpoint Simpson escape error3.9601e−5 and net-lepton/gross error1.0771e−4 exceed the unchanged3e−5 gate. Its conservative energy/Le/EM-pair/material-entropy checks pass, demonstrating why small ledgers do not establish sufficient time resolution. The2→4 absolute Ye difference2.25722e−6 also exceeds the predeclared2e−6 gate and is retained as the third coarse FAIL. No gates, profiles or physical equations were retuned.

Four-step and eight-step runs each14PASS,15.47s/29.51s. Four→eight endpoint differences are dT2.26975e−8MeV,dYe5.56107e−7 and escaped-energy fractional change7.15190e−6. These pass their locked gates but are selected comparisons, not a general order claim or extrapolated error certificate.

Eight-step final state:

| Quantity | Value |
|---|---:|
| T [MeV] |1.996561942538|
| Ye |.315188618257|
| nB [fm^-3] |1.000500125021e−7|
| Escaped νe energy [MeV/b] |.0646263802811|
| Escaped antinue energy [MeV/b] |.2355547613349|
| External pressure work [MeV/b] |.00721462372471|
| Escaped νe number/b |.00677229878779|
| Escaped antinue number/b |.02196091704484|

Actual R declines6.73895% and Q1.85632% over the path. Thus sources are changing materially even though T changes only.00344MeV. Changed-state base/fine/tail profiles are recomputed, with separate incoming FD density checks and a different bracketed EOS energy inversion. All18 endpoint checks pass; the maximum positive individual moment refinement is5.23e−9, well below the3e−6 source gate. Profiles include different beta rest/parent coordinates and independently extended tails; they are numerical controls on the same declared physical model.

Eight-step energy ledger is7.80e−11 relative to escaping energy; Le/gross2.96e−15; equal EM pairs/gross1.10e−13. Independent endpoint Simpson discrepancies are2.434e−6(escape),6.606e−6(netLe/gross),2.674e−7(work). These are larger and more informative about time integration than the algebraic conservative ledgers. Photon-specific energy changes−.100772717461MeV/b; the separately reconstructed photon+nonphoton energy budget agrees at7.82e−11 escape-scaled. Photons do not escape.

## Pair and entropy bookkeeping

Material charge is reconstructed from actual e− minus e+ pair densities at every EOS inversion, not from the electron particle density alone. Comoving weak particle sources are `(Rbeta−REC)/n` for e− and `−RPC/n` for e+. Removing their evolved integrals from actual changes in n_e−/n and n_e+/n yields equal internal EM pair creation. At eight steps these are.00428869532793pairs/b from either side. This is the pair adjustment required by the declared closure, not a separately derived QED reaction rate.

With ζ=μp+μe−μn, mechanical work cancels from the material entropy equation,

`d(s/n)/dt=[−Q−ζR]/(nT)`.

A separate midpoint entropy-source accumulator is compared to the independently evaluated EOS entropy difference. Actual material entropy DECREASES by.129457495645kB/b; the source-integral discrepancy is2.7141e−6 relative to this change. Material entropy need not increase in this transparent open system. Escaping energy/number moments do not determine the full neutrino distribution entropy; no total-entropy or H-theorem claim is made.

## Reproducibility and scope

Full endpoint/stage source moments, EOS inversions, conserved variables and progress checkpoints are retained in each trajectory file; failed coarse gates remain in their original raw records and summary. `weak_density.png/.pdf` show actual T,Ye and physical Q(t)/Q(0) at2/4/8steps. See COMMANDS,LOCK,UPSTREAM and final manifest for exact source/controls/runtime. All experiments<120s, singlethreaded. No fitted source, surrogate EOS or interpolation replaces the physical chain.

This supports one short explicitly controlled weak-on compression experiment with fixed bare masses and a stated pair/photon closure. It does not establish full hydro, long-time behavior, continuum-domain accuracy, a closed neutrino box, full weak inventory, interactions/mean fields, pairing, antibaryon/muon completion, AD or empirical uncertainty. CPU candidate minimum/runtime/normal-CI qualifications remain separate. Accepted weak-off binding qualification alone is not acceptance of this new evolution; independent review is required.
