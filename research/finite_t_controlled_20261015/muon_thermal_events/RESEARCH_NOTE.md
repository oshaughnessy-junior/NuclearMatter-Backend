# Moving-muon decay/inverse event moments in a prescribed FD bath

**Actual moving mu− decay and inverse-three-body total event and flavor-energy integrals are implemented.** The selected matched free-matter bath is T=2 MeV, nB=1e−4 fm^-3, Ye=.29, Ymu=.01. Electron and both neutrino occupation factors are evaluated at the boosted lab energies. The selected fine rules have **221 passing qualification checks plus47 passing tail-contract/replay checks**, with20 original coarse diagnostics excluded, of which18 failed. Independent review remains required.

This is a fixed-bath source reference for mu−↔e−+antinue+numu, not an energy-resolved opacity, coupled ODE, complete muon/leptonic inventory, or medium correction. The radiative, polarized, finite-W,2↔2 and mu+ branches are not included.

## Common event measure and moving parents

The accepted exact-electron-mass vacuum reference supplies

\[
\overline{|M|^2}=64G_F^2(P\cdot k_a)(p_e\cdot k_n),
\qquad d\Gamma_*={\overline{|M|^2}\over2m_\mu}\,d\Phi_3.
\]

There is no Vud or final-neutrino spin average. The source integrates a spin-two mu− parent density, a rest-frame decay event, a uniform global orientation, and a boost along the chosen parent momentum:

\[
d\mathcal R={P^2dP\over\pi^2(\hbar c)^3}
{m_\mu\over E_\mu}\,d\Gamma_*\,{d\cos\Theta\over2}{d\phi\over2\pi}.
\]

The parent spin factor2 multiplying the spin-averaged vacuum width is precisely the full-spin common thermal convention; it is not applied twice. The radial integration uses x=sqrt(Eparent−m_mu), with Fermi-surface panels. All other material/radiation distributions are isotropic in the lab frame, so fixing the parent momentum direction loses no physical angle.

Two independent rest phase coordinates are used: neutrino-pair invariant mass and pair angle; alternatively electron and antinue Dalitz energies. Both retain the exact massive electron and correlated flavor energies. The global plane is rotated before boosting. Numerical spinors are not recomputed in this leaf: the source uses the accepted exact invariant amplitude and honestly shares the vacuum event primitives. The independent vacuum trace/phase qualification remains the upstream reference.

For every event the actual separate factors are

\[
F=f_{\mu^-}(1-f_{e^-})(1-f_a)(1-f_n),\qquad
B=(1-f_{\mu^-})f_{e^-}f_af_n.
\]

Forward and inverse positive moments are summed independently in logs. Net events are additionally accumulated locally using a stable signed expm1 difference; KMS is a check, not how B is generated. `radiation=None` sets **both** neutrino occupations exactly to zero. `electron_blocking=False` is an explicitly unblocked benchmark, not the physical FD-bath source.

The EOS is the accepted free relativistic physical-mass state, with Yp=Ye+Ymu=.30 and separately gated n,p, net-e, net-mu roots. Total chemical potentials are mumu=97.74668143468162 and mue=18.061043957076254 MeV. The mu− particle density is9.999999999999866e−7 fm^-3; mu+ density is3.564706672685924e−49 fm^-3. The source tracks the mu− branch rather than silently calling a net pair density its particle population.

## Independent unblocked lifetime and boost anchors

A separate direct-momentum FD integral gives <m_mu/E_mu>=.9721753607540572. It independently predicts the unblocked lab event rate nmu_minus <m/E> Gamma_rest, which is recovered under a3e−10 gate. The rest tree width is457089.03163053177s^-1 with the declared input GF, not a measured-lifetime fit.

For an isotropic unpolarized rest ensemble, <E_i,lab>=gamma<E_i,rest>. The proper/lab time factor cancels gamma, so each unblocked energy moment must be **nmu_minus times its vacuum rest energy-rate moment**, rather than the number rate times an unchanged rest mean. Electron, antinue and numu anchors independently pass, as does the parent-energy moment nmu_minus*m_mu*Gamma_rest.

| Unblocked lab quantity | Result |
|---|---:|
| mu− events |.4443706942221289 fm^-3 s^-1|
| electron energy |16.905099028553455 MeV fm^-3 s^-1|
| antinue energy |14.487908079899231 MeV fm^-3 s^-1|
| numu energy |16.902277432496767 MeV fm^-3 s^-1|

The vacuum normalization and GF extraction convention follow [PDG Electroweak Model, Sec.10.3.2](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-standard-model.pdf); the unpolarized V−A distribution follows [PDG Muon Decay Parameters](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-muon-decay-params.pdf). QED corrections used in extracting GF have not been inserted into this tree-level source, and coordinate agreement is not a physical uncertainty estimate.

## Physical selected sources

With FD electron blocking and transparent neutrinos, the refined event rate is **.4116306896459381 fm^-3 s^-1**. Radiation receives antinue energy13.04263380056054 and numu energy15.269935877537744 MeV fm^-3 s^-1. Inverse events vanish exactly because both incoming neutrino occupations vanish.

The non-equilibrium common-T FD partner case uses mua=15 MeV and mun=mumu−mue−15−1.3 MeV, so the common chemical affinity A/T=.65. Finest results are:

| Quantity | Result |
|---|---:|
| Forward events |.004057980072648161 fm^-3 s^-1|
| Inverse events |.0021184513591063405 fm^-3 s^-1|
| Net forward events |.00193952871354182 fm^-3 s^-1|
| Net antinue energy |.04359147159569629 MeV fm^-3 s^-1|
| Net numu energy |.1237575494311726 MeV fm^-3 s^-1|
| Net electron energy |.049542980247723016 MeV fm^-3 s^-1|

The net material source is negative the sum of the two neutrino energy sources. Per event ΔNe=+1, ΔNmu=−1; radiation carries Le=−1 and Lmu=+1. Thus at the selected nB, **dYe/dt=+19.3952871354182 s^-1**, dYmu/dt is its negative and dYp/dt=0. These are instantaneous sources at the prescribed state, not an evolved trajectory. The material energy ledger also independently agrees with net parent energy minus net electron energy; full electron/muon rest energies are already counted and no separate muon-rest Q is added.

For common-temperature prescribed FD reservoirs,

\[
\dot s_m={-Q_\nu+(\mu_\mu-\mu_e)R\over T},\quad
\dot s_\nu={Q_\nu-(\mu_a+\mu_n)R\over T},\quad
\dot s_{tot}={A\over T}R\ge0.
\]

The finest material/radiation/total values are −.006398219541084549, +.007658913204886736, +.001260693663802187 kB fm^-3 s^-1. This is a fixed-FD-bath entropy derivative, not a finite spectral-grid entropy evolution. Transparent occupations have a singular initial entropy derivative and no finite entropy rate is claimed for that case. Separate LTE positive events cancel to ~8e−16 relative; reversing A gives inverse-dominated events and still positive total entropy.

## Numerical qualification, failures and restrictions

`initial_refinement_results.json` preserves the entire191-check first study:173 pass, **18 fail**. All failures occur in coarse FD-partner grids, with errors up to.00258 or.000193 at the next rule. The full initial source is preserved. These failures are not relabeled as successes. `qualified_results.json` explicitly excludes all20 diagnostics from those two coarse FD rules, including their two passing checks.

The qualified FD rules compare parent/pair/pair-angle/global-angle/azimuth orders32/40/24/20/32 against36/48/28/24/40: maximum moment difference6.02e−7. Independent Dalitz40 versus pair48 differs by at most6.10e−7, under the unchanged3e−6 gate. Isolated parent-tail50→70 checks at fixed orders are separate from this refinement and agree to roundoff. Each original physical experiment is under21s; the initial multi-case study is41.83s and the separate fine qualification54.59s, each below its120s experiment ceiling.

Parent truncation includes both mu_mu and, for a common-T inverse bath, mu_e+mua+mun. A local tail-contract extension was made after the selected studies;47 replay/contract checks establish **bitwise-identical selected forward/backward coefficients and net moments**, and the high-chemical-potential cutoff formula is checked algebraically. The preserved pre-extension source documents provenance. No high-chemical-potential rate convergence is claimed.

Only common-T FD radiation is accepted; hotter/differently heated partner tails need a new qualified rule. Finite/positive controls and explicit bools are enforced. `initial_import_failure.json` preserves a local generic module-name collision, repaired by naming the module `muon_events.py`; no frozen dependency was edited.

The declared numerical precision covers these selected baths and rules. It is not a broad degenerate/temperature domain certificate. No standalone flavor opacity, nonthermal partner function, mu+ branch, crossed scattering, polarization, radiative correction or coupled fluid/radiation evolution is supplied. Independent review is pending.

Reproduction from repository root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/muon_thermal_events/validate.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/muon_thermal_events/qualify.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/muon_thermal_events/check_tail_contract.py
```

The first command intentionally reports the preserved coarse convergence failures. The selected qualification report is `qualified_results.json`; the original coarse failures remain independently inspectable.
