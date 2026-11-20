# First compact detailed-physics moment fits: fixed pair bath

**Author freeze; different independent review pending.** This supplies an optional one-dimensional convenience approximation for actual free νe–e± scattering moments, with detailed quadrature retained as an explicit fallback. It is not a replacement for the AD physical chain, an EOS, a spectral redistribution kernel or a new production implementation. Positive moments pass the declared measured gates; **two signed-energy error gates fail and remain failed**.

## Exact state, closure and physical moments

The fixed bath is T=2MeV, µe=.17446333200826397MeV, physical me=.51099895069MeV, equilibrium e−/e+ FD pairs with opposite total chemical potentials. Actual target populations are ne−≈2.034286367843283e-7,ne+≈1.734286367843283e-7fm^-3; their net3e-8fm^-3 matches the earlier neutral free-npe example. Nucleons do not enter this scattering slice. Both target species and final charged-lepton FD holes are retained. Radiation is a test incident **νe** with outgoing-neutrino blocking absent. This is not common-T trapped FD radiation or a claim of kinetic equilibrium.

Use the accepted massive free Fermi current: GF=1.1663785e-11MeV^-2, fixed illustrative sin²thetaW=.23129, CV=.96258,CA=.5 for electron flavor including charged-current interference, **no Vud**. The primary [Guo et al. leptonic formulation, Eqs1–4 and AppA](https://arxiv.org/html/2006.12051v2) and existing independently reviewed scattering leaf remain the physical normalization anchors. The source includes exact moving-target two-body CM phase space, target spin2, one active incident neutrino helicity and actual final-target blocking. No stationary-target/classical approximation or empirical fit supplies the reference values.

At incident energy E in MeV, fit three positive observables:

- R: total scattering-event frequency per incident νe, s^-1;
- Rtr: integral(1−cos_lab)dΓ, s^-1;
- Qout: integral Eout dΓ, MeV s^-1.

The derived signed neutrino energy-change moment is `DeltaQ=Qout−E R`; matter receives its negative. Scattering changes no species or net lepton number. These moments contain the final-neutrino phase measure already; no extra neutrino DOS is multiplied into them. R is scattering frequency, not absorption or an arbitrary j+chi relaxation rate; Rtr is not by itself a diffusion coefficient for general inelastic transport.

## Preregistered data and selection, before heldout computation

PREREGISTRATION.json was written before source computations. The incident-energy domain is **1–30MeV**. Training energies are

```
1,1.2,1.8,2.5,4.5,5.5,9,10.5,16.5,19,24,30 MeV.
```

Six heldout energies1.45,3.5,7,13.5,21.5,27.5MeV lie in blocked bands[1.35,1.55],[3,4],[6,8],[12,15],[20,23],[26,29], with no training energies inside those bands. These are measured validation points, not a proof of error everywhere inside the bands or domain.

At every training/heldout state, actual detailed coarse/fine rules are radial/incident/outgoing/phi/tail `(24,24,24,32,60T)` and `(32,40,40,48,80T)`. Separate e− and e+ results are stored before summing. The predeclared quadrature gates are1e-4 relative for positive moments and1e-4 in signed difference/(Rfine T). The108 quadrature/event-energy checks pass; maximum observed error2.35e-8 is far below the1% fit gate. Independent adaptive actual target-density integrals supply24 additional forward population checks before fitting. Method agreement estimates numerical error within this free model, not missing EW/QED/plasma physics uncertainty.

Fit natural logarithms of the positive numerical observables in their stated units, using

```
x = 2 ln(E/1MeV)/ln(30) − 1,
O_fit = exp[sum_n c_n T_n(x)].
```

Candidate degrees2..7 were declared in advance. For each observable choose the smallest degree with maximum training relative error≤.003 AND leave-one-training-point-out maximum relative error≤.007; otherwise retain the failed policy and choose the minimum-LOO-error candidate. The rule selected degrees3,4,2 for R,Rtr,Qout. No heldout value enters coefficient or degree selection. FITTING_LOCK.json pins model SHA `9067c895b38e43451cce803a39681db961abb7066eea2cb5d9fa2d7fc5dd2e9f`, the preregistration and complete training report, and records that heldout source data did not yet exist. `compute.py heldout` requires that lock.

The twelve coefficients, in increasing Chebyshev degree, are:

| Observable | c0 | c1 | c2 | c3 | c4 |
|---|---:|---:|---:|---:|---:|
|R [s^-1]|1.888085875693068|1.783585218908185|−.0109577154379772|−.00567990164504653|—|
|Rtr [s^-1]|1.591720804824830|1.251929957124420|−.0960914343802928|−.0107347213237842|.00377853943265682|
|Qout [MeV s^-1]|3.940716768461922|2.583445879496178|.143475993521173|—|—|

Full float64 coefficients, all candidate/LOO errors, units, weak parameters, physical-source hashes, target densities and data hashes are in model.json. The table is a readable summary; the JSON numbers are authoritative for exact replay.

## Heldout accuracy and retained signed-energy failures

The declared positive fit gate is1%. All18 positive heldout comparisons pass:

| Observable | Maximum training error | Maximum LOO error | Maximum actual heldout error |
|---|---:|---:|---:|
|R|.2044%|.6027%|.1855%|
|Rtr|.1740%|.6154%|.2296%|
|Qout|.2839%|.5214%|.2078%|

No uniform error certificate follows from six heldouts or the dense sampling of the fitted polynomial. Dense mathematical positivity/bounds tests do not substitute for detailed physical validation data. Unsupported states and energies are rejected, not silently extrapolated.

The signed moment is derived by subtraction, never by `log(abs(DeltaQ))`. The **separately declared** absolute gate is

```
abs(DeltaQ_fit−DeltaQ_reference)/(R_reference T) <= .03.
```

A5% relative signed-moment gate is also checked away from zero where `abs(DeltaQ_reference)/(R_reference T)>=.2`. Two absolute gates fail in the training set: E19MeV has−.0314184, E30MeV+.0593932 in the normalized signed error. All predeclared heldout absolute gates and all applicable away-from-zero relative gates pass. These two failures are retained verbatim in validation_results.json and QUALIFICATION.json; **coefficients were not retuned after seeing them or the heldouts**. The positive-moment policy did not constrain subtraction error tightly enough for that absolute signed-energy budget.

The API explicitly returns `signed_energy_RT_gate_pass=False` and an instruction to use detailed quadrature when signed-energy accuracy is required. Its derived signed value is diagnostic, not admitted as satisfying the failed global requested gate. This separation matters even though each positive fit is quite accurate. A future jointly constrained signed-moment design requires a new declared training/validation protocol; it must not be tuned to these now-known heldouts and presented as an unchanged independent test.

The preregistered *post-lock* zero diagnostic finds the fitted signed root E8.15500215216MeV and computes actual detailed moments there and at±1% energy. Normalized errors.009247,.009088,.008920 pass the same.03 absolute gate. These are adaptive diagnostic points, not the six preregistered heldouts. A separate final batch also actually recomputes coarse moments at all three diagnostic energies; all12 positive/signed reference-refinement gates pass unchanged1e-4 budgets. A single incident-energy exchange zero is not full-spectrum thermal equilibrium, beta equilibrium, or a radiation-temperature measurement.

## Independent detailed phase anchor and fallback

At heldout7MeV, a separately assembled analytic target-azimuth/transverse-target-shell kernel is integrated over outgoing energy and direction for **both** target charges. It calls neither the CM event integrator nor the author's differential function. Outer12/16→18/24 controls, outgoing100→150MeV and target70T→90T supply an independent refinement. Fine results are

```
R=8.67214938237 s^-1,
Rtr=6.51062965694 s^-1,
Qout=65.02748727796 MeV s^-1,
DeltaQ=4.32244160139 MeV s^-1.
```

They agree with the actual CM references to about8.2e-10 relative or better for positive moments, with signed RT gate also passing. This is new forward normalization/coordinate evidence, not a fit-to-fit comparison or physical uncertainty assessment. The full validation batch has114PASS/2signedFAIL in37.20s, including the independent anchor, near-zero diagnostics and an actual detailed fallback call.

`api.evaluate(E,method='fit')` reads the immutable coefficient model, checks its hash and current physical-source pins, rejects any other T,µe,flavor,target set, outgoing-radiation closure or energy domain, and returns moments with state/units/domain/model identity plus the signed qualification flag. `method='detailed'` invokes the actual physical quadrature through the pinned adapter; it does not substitute a saved moment, interpolation or hidden rate table. It reproduces the fresh heldout7MeV source. Coefficient/qualification/adapter integrity and unsupported-input checks are explicit. This is trusted research code, not a hostile-process or concurrent-filesystem security service.

The539 final API checks consist mainly of501 **mathematical** fitted-positivity/transport≤2R samples and qualification/numerical-replay checks; they are not539 independently computed physical points. Metadata additions preserve every previously validated numerical coefficient/prediction exactly. The API never advertises the two failed signed gates as passed.

## Reproduction, evidence and limits

Actual source work is checkpointed after every energy. The full12-state training collection totals122.91s across separately bounded coarse/fine state integrations; the largest paired-target state call is9.60s, and heldout collection totals62.95s. No individual physical source or phase experiment reaches120s. The first reporting attempt passed an unsupported `flush` keyword to json.dumps after saving one actual state; original driver/partial/error are preserved and the training run was restarted, not recovered from an incomplete source integral. A metadata-probe relative-path error is similarly preserved; it executed no physical source. Neither operational failure changed the fit model or gates.

Frozen reports, coefficients and lock must not be overwritten. The compute/validation drivers now refuse existing evidence paths. To reproduce the full development sequence, use a fresh owned sibling campaign leaf with the same frozen function sources and preregistration, then run training compute, build_fit, heldout compute and validate; preserve the generated coefficient lock before heldout work. Pure API and independent-review probes can read the existing frozen leaf. The fitted numerical source and validation evidence are pinned separately from current runtime timings. PNG/PDF show the measured errors and two signed failures and were visually inspected.

This is a first honest1D fixed-bath compact model, not a T/density/composition fit, muon/antineutrino model, spectral/angular redistribution operator, final-neutrino FD closure, meanfield/plasma correction or production transport backend. It supplies no derivative-fidelity, AD-chain or uniform-domain certificate. Retain direct detailed physics for scientific checks and signed-energy use; expand fits only with independently reviewed physics, declared states/closures, new heldouts and controlled physical-error metrics.

Final final_checks.json adds17/17 passes in4.83s: the12 actual zero-reference refinement gates, three exercised overwrite-rejection commands, exact evidence preservation and own-source syntax. These are separate from the two retained fit failures; they do not retune the model or qualify the failed signed output.
