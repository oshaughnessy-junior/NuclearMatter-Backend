# Selected GM1 lepton-integrated capture pilot

This research prototype integrates the independently reviewed GM1 differential capture ansatz over outgoing charged-lepton energy and angle. It inherits the explicit [Guo2020, Eqs23–28](https://arxiv.org/html/2006.12051v2) current prescription. The EOS, mean-field adapter, free tensor and vacuum current sources remain unchanged. No external GM1 forward qualification is inferred from internal agreement.

## Physical and numerical contract

The selected material factory solves `T=8MeV,nB=.22fm^-3,Ye=.17,Ymu=.04`, hence `Yp=.21`. Electron and muon net fractions/chemical potentials are independent. It checks the nuclear density/scalar roots, local nuclear Hessian/scalar curvature, field-energy ledger, Euler relation, charge roots and negligible thermal antibaryon population. Its common bare nucleon mass is939MeV; its Dirac mass is568.5572415829281MeV. Total chemical potentials include the same EOS rest convention. No physical vacuum neutron-proton Q is added.

All differential nodes use physical `omega=Enu-El`, kinetic `w=omega−Ub+Ua`, kinetic masses mD and chemical potentials `nu_i=mu_i−Ui`. Scalar form factors use physical q²; WM/PS vertices use kinetic q*. The weak-current mass is the common bare939MeV, not mD or a Landau mass. The full16-component tensor and physical massive-lepton trace are retained, without a conserved-current component reduction, RPA or correlation correction.

Absorption chi and inverse emission j are independently integrated in log space. They are occupation coefficients in `s^-1`: chi is true absorption; `j+chi` is relaxation opacity in frequency units. A length opacity is chi/c, while the relaxation length inverse is `(j+chi)/c`; these must not be conflated. No neutrino occupation, neutrino blocking, incident-energy density of states, or volume moment is included in the per-incident-energy coefficients. Their later radiation collision source is `j(1-fnu)-chi*fnu` for one neutrino helicity. Finite logs are retained with explicit ordinary underflow/overflow flags, never numerical floors.

The outgoing lepton endpoint uses `El=ml+x²`; the integral is `int_ml^infinity dEl int_-1^1 dcos(theta)` of the actual differential coefficients. Each response recomputes exact shifted support; no artificial threshold width is introduced. Thermal/degeneracy/recoil estimates place quadrature panels only. A finite upper energy cutoff and every numerical control are recorded. The alternate q coordinate uses `dcos(theta)=q dq/(Enu pl)`, between `|Enu−pl|` and `Enu+pl`. An exact-state run contains its full material state, current switches and orders; there is no cache or stale-state reuse.

KMS requires `log(j/chi)=−[Enu−(mu_l+mu_p−mu_n)]/T` for neutrino capture and inverse. It is checked after separate integration but does not prove quadrature convergence: this ratio is constant pointwise. Numerical gate choices concern selected integrals and do not estimate physical current/EOS uncertainty.

## Selected scope and pending numerical results

The pilot uses electron-neutrino and muon-neutrino incident energies40MeV on the same material state.40MeV lies below the free stationary-target muon-neutrino threshold for either physical unequal vacuum masses or common939MeV vacuum masses. A positive finite-density rate would demonstrate support in this declared Hartree/thermal/Fermi ansatz; it would not isolate thermal opening from mean-field release and target motion, establish a universal in-medium threshold, or qualify neutron-star Urca broadening.

Only n+nu_l ↔ p+l− capture and inverse capture are integrated. There is no crossed beta-decay contribution, antineutrino total source, lepton decay, full flavor inventory, incident-energy-integrated moment, coupled thermal evolution, or hydrodynamics. The external matched finite-T GM1 forward-reference gate, global coexistence, derivatives and production integration all remain open.

Final values, failures and refinement evidence will be appended at freeze. The interface requires explicit flavor and the CLI requires an explicit outer quadrature order; there is no presumed universal order.

## Frozen selected values and numerical evidence

The selected material has `Un=338.0682785735156MeV`, `Up=282.54875402213236MeV`, total `mu_n=1000.1139607177219MeV`, `mu_p=890.2455022016085MeV`, `mu_e=203.11989906077872MeV`, and `mu_mu=162.71685512607556MeV`. No beta equilibrium or neutrino bath is imposed. Each run retains the complete accepted factory's root, local stability, energy and charge diagnostics. The omitted antibaryon/net density ratio is3.31e-66 at this state.

| Incident neutrino, Enu40MeV | True absorption chi [s^-1] | Emission j [s^-1] | Relaxation j+chi [s^-1] |
|---|---:|---:|---:|
|electron|958899.1195336229|745802384.5583102|746761283.677843|
|muon|16495843.138006153|82200221.30898467|98696064.44699074|

The large difference between true absorption and relaxation is physically relevant here. In particular, the degenerate lepton populations make j larger than chi at this selected neutrino energy. Calling j+chi a true absorption rate would be wrong. These are coefficients, not gross interaction counts or escaping energy luminosities.

The original exploratory full-angle muon grids did not meet the1e-6 relative convergence target. Their sources and raw values remain in `exploratory_unpartitioned_angle.py`, `exploratory_unpartitioned_results.json` and `profile.json`. The first exact-angular-boundary grid also needed refinement; its source and results are separately preserved. No physical kernel, EOS, current or tolerance was adjusted.

For equal positive kinetic masses, the energy function has |velocity|<1, so `|Eb*−Ea*|<q` at every finite momentum. The positive-energy particle-hole response therefore has exact support `q>|w|`. This restricts the cosine upper endpoint to `min[1,(Enu²+pl²−w²)/(2Enu pl)]`; regions outside it are physically empty for this ansatz. The condition does not include pair-creation cuts, which are omitted explicitly. Additional **numerical energy panels** resolve where this boundary crosses the collinear lepton endpoints. Their independent closed forms are

`El=(X²+ml²)/(2X)`, with `X=Un−Up` or `X=2Enu+Un−Up`, for positive X and El inside the integration interval.

The implementation locates these roots numerically in `sqrt(El−ml)`; all selected roots are checked against those independent closed forms. At the muon point the two energies are108.94826389660089 and128.2981981098021MeV. There is no imposed physical width or threshold rescaling. `support_checks.py` has108 passing exact-support and q/cosine phase-measure polynomial tests. Its initial syntax and NumPy-boolean reporting errors are preserved; neither changed physical code or test tolerances.

`validation_results.json` has92 passing selected checks, in addition to the108 support tests. Electron8-cosine→12-q refinement differs by1.39e-8 relative. The muon16-q→20-q comparison **failed** at1.82e-6 and remains explicitly failed in the report. Its subsequent20-q→24-q refinement passes at2.04e-7; independently sampled20-cosine and24-q results agree within9.47e-7. The final muon point therefore just meets this pilot's unchanged1e-6 numerical target; it is not a stronger accuracy certificate. Inner response order24→32 and independent baryon/lepton tail enlargements are also checked at fixed outer controls. KMS and represented-log identities pass after separate integrations, without being substituted for a convergence test.

The zero-field check uses an explicit **common939MeV microscopic bath**, T8MeV,total chemical potentials947/932MeV, electron potential25MeV and Enu40MeV. It does not recompute a GM1 EOS with artificially removed fields and does not compare to physical unequal vacuum masses. The GM1 adapter with zero vectors and a separately integrated frozen common-mass free kernel use cosine and q phase coordinates respectively. The initial order8 and12 comparisons fail at4.42e-5 and1.03e-5; their raw results remain preserved. Refining the free q integral to24 yields agreement6.24e-7 with the order12 adapter cosine result, passing the unchanged1e-6 gate. This is a matched free-limit check, not an external GM1 calculation or a new independent absolute weak-current calibration.

All final selected rates and logs are ordinarily represented and positive. Before freeze, the all-empty finite-domain metadata label was corrected to avoid declaring a global physical zero from empty samples in a truncated integral. The prior source is retained as `pre_empty_domain_metadata.py`; this changes an unused reporting branch only and none of the positive pilot coefficients. No coefficient is floored. A finite-domain empty result is explicitly unqualified as a global support certificate.

Individual integrals took1.13–82.49s. The final muon24-q integral took81.10s and the refined free-limit q anchor74.53s. The200 passing checks concern two selected finite-T integrals and their numerical conventions; all earlier failed/coarse evidence remains visible. Independent review is still required before acceptance. Current/EOS model uncertainty, interaction corrections and external GM1 forward agreement remain unquantified/open.

Reproduce the selected material points from this directory with the existing base environment:

```bash
/Users/rossma/miniconda3/bin/python run_point.py --flavor e --order 12 --inner-order 24 --inner-tail 65 --outer-tail 40 --coordinate q
/Users/rossma/miniconda3/bin/python run_point.py --flavor mu --order 24 --inner-order 16 --inner-tail 80 --outer-tail 48 --coordinate q
```

The reporting checks are cheap: `python support_checks.py` and `python validate.py` use the retained evidence rather than rerunning the rate grid. `CHECKPOINT.json` pins sources, raw evidence and environment; no cache or foreign state is reused.
