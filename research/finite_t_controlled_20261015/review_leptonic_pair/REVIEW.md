# Independent review: physical electron-pair neutrino production

Accepted at selected scope; final counts and hashes are in CHECKPOINT.json. No author or upstream source/raw file is edited. The reference checks absolute amplitudes, massive total and separate neutrino/antineutrino energy moments, both FD partner occupations, direct inverse events, equilibrium, both affinity signs and entropy. This is a free Fermi-model source, not full neutrino inventory, plasma transport or coupled evolution.

## Primary normalization and the preserved source ambiguity

The [primary2006paper](https://arxiv.org/html/astro-ph/0511555), Eqs6–11, fixes the flavor couplings and charged-spin average4. HTML and [PDF text](https://arxiv.org/pdf/astro-ph/0511555) both show the disputed printed Eq17. Fresh local gamma matrices/Dirac traces and angular integration independently reproduce Eq11, the derived CM cross section and the separated-FD normalization. They do not reproduce printed17. The discrepancy is retained as an EXCLUDED external anchor, not corrected by fitting or called a verified erratum. The paper counts particle production F=2R; this code's R counts pair events.

With illustrative sin²thetaW=.23129, independently checked electron couplings are CV=.96258, CA=.5 (CC+NC), while mu/tau are CV=-.03742, CA=-.5 (NC). No Vud occurs. GF=1.1663785e-11MeV^-2 and exact me=.51099895069MeV are used. Initial charged-spin average4 is accompanied by two charged spin2 measures. Final nu and anti each have one active helicity. The same common phase measure restores the unaveraged matrix element for inverse events; an inverse physical cross section with incident neutrino helicities fixed would not use the charged-spin average4.

At fresh CM electron energies .515,.743,3.17MeV, three angles each and both e/mu flavors, a separately defined gamma trace agrees with a derived CM angular polynomial and the author scalar amplitude. Angular integration independently fixes the absolute cross section. For beta=sqrt(1-4me²/s), the polynomial is proportional to A0+A1c+A2c² with

    A0=2(CV²+CA²)+8me²/s(CV²-CA²),
    A1=8CV CA beta, A2=2(CV²+CA²)beta².

In particular the threshold annihilation sigma*v is vector-only; a finite axial threshold term cannot be silently imported from the conflicting formula. Explicitly, the angular-trace result minus printed17 is GF²(CV²-CA²)me²(2me²+P1·P2)/(12pi E1E2). This difference is independently verified at .6 and .89MeV for both flavors. At .6MeV the literal printed result differs by-12.5577% for electron flavor and+87.0399% for muon flavor. Both values remain in limits_source_results.json with accepted_anchor=false.

## New massive empty-radiation reference, including separate daughter energies

reference.py derives and integrates the initial relative angle and the outgoing CM angle analytically, then evaluates only one-dimensional adaptive massive FD moments. This is independent of the author's event-angle/radial rules and does not import its thermal or sigma_v functions.

Writing B=CV²+CA²+(me²/s)(2CV²-4CA²), the amplitude-weighted mean bath-frame neutrino energy for FIXED initial momenta is

    <E_nu>=(Eminus+Eplus)/2 + CV CA(Eminus-Eplus)/(2B),
    <E_anti>=(Eminus+Eplus)/2 - CV CA(Eminus-Eplus)/(2B).

The odd CM angular term is essential: equal partner energies must not be forced in a charge-asymmetric bath. Initial-angle integration leaves a polynomial in Eminus,Eplus. Products of L_j^±=integral_me^infinity dE sqrt(E²-me²) E^j f(E;±mu_e), j=0..3, therefore give event, separate nu/anti and electron/positron energy moments. Exact massive thresholds use a smooth square-energy coordinate in the adaptive reference. Tail50T→90T is checked independently.

Both flavors were freshly compared at (T,mu_e)=(2,.17446333200826397),(1.73,.29),(.37,.12)MeV. All FIVE moments match the author to3.22e-15 relative error against the declared2e-6 gate. The last, more massive state has electron-flavor R=2.2778465161866465e-13fm^-3s^-1 and Qnu/Qanti=3.609104629153711e-13/3.5970059016661513e-13MeVfm^-3s^-1. The new T1.73 electron-flavor event rate is5.9268789999119144e-8fm^-3s^-1, with Qnu/Qanti=4.220891563829444e-7/4.20835843992555e-7MeVfm^-3s^-1.

For the original physical T2 bath the independent values are R=1.8942019361365186e-7, Qnu=1.557789272873229e-6, Qanti=1.5553673088921586e-6 for electron flavor, and R=4.001065830606647e-8, Qnu=3.2971386326875793e-7, Qanti=3.296197101675387e-7 for muon flavor. Units are fm^-3s^-1 for events and MeVfm^-3s^-1 for energies. Tau has identical couplings here; this is not a separate tau-medium calculation.

Independent massless, mu_e=0 limits at T1/T2 verify the absolute coefficients R=49pi³ GF²(CV²+CA²)T8/(64800 hbar hbarc³) and Qtotal=7zeta(5)GF²(CV²+CA²)T9/(12pi hbar hbarc³), the powers and equal partner energies. These are normalization limits only. A CP mu_e sign reversal independently exchanges both daughter and initial charged-particle energy moments and preserves R.

## Independent actual FD partner/inverse integral

fd_reference.py supplies a separate event quadrature: p/T=sinh(t) momentum integration, an outgoing CM axis along the bath BOOST direction, and a separately coded scalar current. The author uses square kinetic-energy panels and the incoming electron CM axis. The new construction uses no author thermal/amplitude/boost routine. Actual electron/positron FD factors and both neutrino partner factors are evaluated directly for F and B. Separate positive F/B and stable pointwise signed differences are retained; KMS never generates B.

At the actual T2 bath, controls28radial/8initial/8outgoing/phi8/tail40→40radial/12initial/12outgoing/phi12/tail60 change positive moment branches by at most1.22e-8. An additional ISOLATED angular/phi12→16 change at fixed radial40/tail60 is3.53e-11. Forward electron/positron densities independently reproduce2.034286367843283e-7/1.7342863678432833e-7fm^-3, with NET charge3e-8fm^-3. Using net charge as an electron particle density would fail these source normalizations.

Empty, common-T LTE(.5,-.5), positive-affinity(.5,-2.1) and negative-affinity(.5,1.1) neutrino/antineutrino chemical-potential pairs were computed. Separate F/B number and all four energy moments, signed net moments, energy conservation and entropy pass the declared2e-6 numerical gate. Before the isolated author azimuth check, maximum disagreement with fresh author20/10/phi12 runs is1.55e-7. This is below the gate but materially larger than the author's reported radial/angle ladder with phi fixed; it is retained rather than reported as machine-precision FD agreement.

The common chemical force is -(mu_nu+mu_anti)/T, not a one-flavor chemical potential. Direct entropy integral(F-B)log(F/B) is nonnegative in both affinity directions, equals the chemical force times net pair events, and cancels at LTE relative to nonzero gross F+B. Removing e− and e+ together changes neither material net charge nor material net Le; emitting nu and anti changes neither radiation net Le. Rest-inclusive matter energy loses Qnu+Qanti with no extra2me source. Empty radiation is not assigned a finite initial FD entropy derivative. These are fixed reservoirs, not an evolved spectral H theorem.

## Scope and reproducibility

The finite-mass paper formula conflict is preserved as an excluded anchor even though the amplitude/current route passes. Final raw reports retain each method and control separately; no review failure or incomplete experiment is hidden. Author input/control tests also reject malformed kinematics, mixing angles, flavors and mixed radiation temperatures. Source hashes are verified unchanged.

Accepted scope is selected free electron-pair Fermi-model rates/moments, flavor-dependent empty radiation and selected electron-flavor common-T FD inverse cases. No full flavor/source inventory, arbitrary spectral radiation, broad degeneracy/temperature accuracy rectangle, derivative, QED/EW/plasma correction, running weak angle, finite-W/Z physics, oscillation, material feedback or production claim follows. Numerical method agreement does not measure those omitted physical uncertainties. Do not insert this pair process into an antineutrino-only source join without its neutrino partner and zero-net-Le ledger.

### Resolved azimuth margin and final status

An isolated author phi12→24 run at identical radial20/incident10/outgoing10/tail60 in the inverse-dominated case changes separate positive moments by1.54353e-7, matching the previously observed discrepancy. Its new phi24 moments agree with the independently converged boost-axis quadrature within2.68e-12. Thus the remaining phi12 numerical error was resolved without changing any source, interaction, physical model or tolerance. The author's original radial/angular refinement kept phi12 fixed; its smaller reported ladder difference should not be used as the total error estimate. All selected old/fresh values still pass the declared2e-6 review gate. azimuth_results.json preserves the new control and comparison.

**Final result:351/351 new checks pass** (102 vacuum/massive-empty,100 direct partner-FD,33 limits/source-diagnostic,31 coupling/density/ledger,65 isolated independent-angle,20 isolated author-azimuth). No source fix is requested. The printed Eq17 conflict remains excluded, and all limitations above remain in force. Numerical runs take less than30s individually; their actual timings are recorded. Frozen source and report hashes are in CHECKPOINT.json.
