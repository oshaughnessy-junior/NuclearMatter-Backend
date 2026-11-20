# Independent review: three-source photon/pair cooling element

**ACCEPTED for the selected .002 s fixed-density free-npe beta+PC+EC cooling experiment with trapped LTE photons and instantaneous electromagnetic pair equilibrium.** There are **195 passing review gates, no new physical/numerical gate failure**: 85 forward/provenance, 29 admission, 8+12 independent RK4, 21 alternate EC and 40 source/time/accounting comparisons. Repeated moments and provenance hashes are correlated checks, not 195 physical benchmarks. All 55 initially frozen author/interface/dependency files remain unchanged. This acceptance does not extend the reaction inventory or qualify long-time cooling.

Reviewed author manifest SHA256 `9c6abbbe5734c1905748809e7d681b75499c8db81afb5c301798ab1ff0dc0c77`, element `a2711240850f639d0259e9eb2e74b3931d4a4f3ec1726c7bb4934fdcf402a27b`, worker `0fe9f30a9dcf985c92a58f9223cae74f0174816aa592ad57828d1dc1db1c240b`, and client `15c93ab17f10a895324fac06210847cf9972c929ba2b7de34c52d68d158f07e7`. The actual experiment plan is `EXPERIMENT_PLAN.json`. I read the note, plan, source/interface functions and previously accepted source/balance reviews before probes. Current source/review pins and every author-manifested artifact verify. The accepted initial source record and alternate EC reference additionally match the earlier independent balance review's frozen dependency hashes.

## Independent forward state, photon normalization and source gates

My EOS uses **infinite original-momentum scalar adaptive quadrature**, independently inverted particle-only neutron/proton densities and the electron-minus-positron charge density. It does not call the author's finite kinetic-energy `quad_vec` EOS for its state/energy inversion. Baryon rest energy is explicitly mn nn+mp np, with separately integrated kinetic energy; the relativistic electron and positron energy integrals already include their rest masses. A separate Planck integral, 2/(2pi²) integral x³/(exp x−1) dx, establishes the two-polarization coefficient pi²/15. Photons have mu=0 and energy pi²T⁴/[15(hbarc)³], with Cv=4epsilon_gamma/T.

At both initial (T2,Ye.3) and actual changed Heun4 endpoint, independent total u agrees with the author within **3.38e−11 MeV/baryon**. Chemical potentials and actual e−/e+ populations pass separately; net charge is not substituted for either incoming population. Independent EOS finite differences give u_T within 2.48e−9 relative and u_Y within 5.39e−9 MeV/baryon. The rest-inclusive Euler identity closes within 3.38e−14 relative. The independent conserved-chain temperature derivative agrees, including the composition term.

Fresh initial **fine** source moments match all 18 earlier accepted number/species-energy moments under a verified source-record hash. A fresh changed-endpoint base/fine evaluation gives maximum separate-moment change **5.23e−9**, against the unchanged 3e−6 gate. Positive channel energy, incoming FD densities, actual EOS and conservative source ledgers pass before time integration. No saved source totals, interpolated rates or fixed-bath scaling supplies feedback.

At that new endpoint I also invoke the previously accepted **Dirac-trace/invariant-s/linear-proton-momentum/boost-axis CM** EC reduction using the independently reconstructed chemical potentials. Its (radial/invariant,angular,phi,tail) rules (24,10,8,70T)→(32,12,10,70T) take 7.44 and 24.21 s. The separate EC moments refine and match the author's scalar-current phase integral within **2.22e−10** relative. This is a structurally different EC amplitude/phase reduction. The beta and PC phase functions remain the already reviewed functions; I do not claim a new independent implementation of them here. All share the same physical free full-current model, so agreement does not quantify omitted interactions or physical uncertainty.

## Independent time and EOS integration

I integrate the conserved vector (u,Ye,three escaped event counts,three escaped channel energies) with **classical RK4**, and recover each stage temperature by a separately bracketed scalar root of my independently integrated EOS. This differs from the author's Euler/Heun and analytic-Cv Newton inversion. Every RK4 stage launches a fresh pinned source child at that actual (T,Ye), checks the independent EOS against the returned state, and uses newly integrated beta/PC/EC moments. No trajectory/source cache is used.

| Method | Final T [MeV] | Final Ye | Escaped energy [MeV fm−3] |
|---|---:|---:|---:|
| Author guarded Heun4 | 1.996225222710608 | .3151827905895560 | 3.000347851599e−8 |
| Independent RK4, one step | 1.996225203707660 | .3151834037221295 | 3.000366867946774e−8 |
| Independent RK4, two steps | 1.996225203706281 | .3151834054639028 | 3.000366881861683e−8 |

RK4 one→two changes escaped energy by 4.64e−9 relative and **composition change Ye−Ye0** by 1.15e−7 relative. Author Heun4 versus RK4 two differs by **6.34e−6 of escaped energy**, **4.05e−5 of composition change**, and 5.04e−6 of temperature change. All meet the selected .001 comparison gates. The energy/time comparison is not normalized by the large rest energy, and the composition comparison is not normalized by total Ye. Small RK4 differences are a selected numerical comparison, not an absolute continuum-time error bound: source rules are shared and their measured endpoint quadrature changes are of comparable size to the RK4 energy refinement. This review does not infer an observed RK4 order from two rules.

## Energy, lepton and EM-pair accounting

At fixed density/volume, total rest-inclusive energy obeys u_dot=−Qescape/nB with no pressure-work term; Ye_dot=(Rbeta+RPC−REC)/nB. Independently assembling all literal stored Heun stage weights reproduces every channel's integrated event and energy total exactly. Reconstructed author energy closure is **3.30e−10 of exchanged energy**. Independent RK4 exchanged-energy ledgers close within 1.36e−12; their integrated electron-lepton ledgers close within 4.32e−16 of emitted gross numbers. These discrete conservation checks are separate from the source and time comparisons above.

Beta and PC raise material proton/net-electron Le by one and emit anti-nu_e; EC reverses these material changes and emits nu_e. Baryon and charge conserve, and NB deltaYe cancels escaped nu_e minus anti-nu_e number. Both outgoing neutrino species carry positive energy. The already integrated event phase receives no extra neutrino DOS.

The author's independently reconstructed photon-energy decrease is **1.0314796661063923e−8 MeV fm−3**; the material n+p+e± decrease supplies the rest of the 3.000347851599e−8 escape. Photons and pairs are internal parts of the evolving reservoir. There is no external thermostat and no photon escape. Using the SAME initial source, my finite-difference EOS gives photon-ON Tdot=−1.89788065474 MeV/s and photon-OFF Tdot=−2.95345816946, matching the stated reservoir/Cv diagnostic. This is not a second photon-OFF trajectory.

The weak particle changes are delta e−=Ibeta−IEC and delta e+=−IPC. Comparing these with the final FD populations independently infers the same EM pair creation count from either sign: **4.263712128456544e−10 fm−3** for Heun4; relative equality residual 2.09e−15 of gross emitted event number. It diagnoses the assumed rapid-EM-equilibrium closure, not a calculated EM rate. Because total EOS and species energies already include rest energies, adding a reaction Q, 2me times the pair count, or a separate photon-replenishment heat term would double count internal energy. No such addition occurs.

## Admission/isolation and preserved records

Twenty-nine independent controls reject changed/nonfinite/Boolean state inputs, wrong density/action/profile/identity, nonliteral photon flag, false/integer balance acceptance, missing mandatory review or loaded-source paths, altered photon energy/Cv, wrong conservative derivative signs, incorrect neutrino/positron counts, modified controls and zero separate moments. A real worker-entry spy verifies malformed admission fails **before physical preparation**. A genuine fresh packet is the positive control. Returned child packets carry the exact 13 admitted campaign Python paths. This is selected pinned research isolation/admission, not a security sandbox, arbitrary-input API or universal packet-integrity certificate.

One reviewer operational failure is retained: my first manifest reader assumed a flat mapping and attempted to open the metadata key `status` as a file. No physical/source call ran. `initial_forward_source.py.txt` and `initial_forward_error.txt` preserve that exact script/error; only my reader was corrected to use the manifest's `files` mapping. No scientific equation or gate changed. All later experiments completed, and no physical failure was relaxed or suppressed. Prior author/source coarse failures and review exclusions remain untouched, including the original guarded-replay execution provenance.

This acceptance covers only fixed nB1e−7, this short T≈2/Ye≈.3 path, the three named free charged-current sources, empty incoming neutrinos and **assumed** trapped LTE photons/EM-pair equilibration. It excludes a full weak-energy inventory, weak pair annihilation, bremsstrahlung, modified Urca, muons, interacting/paired EOS, Coulomb/radiative corrections, trapping/rescattering/transport, photon escape, long-time or global thermal/chemical equilibrium, physical-model uncertainty and all derivatives/AD. No initial empty-neutrino entropy derivative or H theorem is qualified.

## Exact reproduction

From `/Users/rossma/Genesis_LANL`, using the existing interpreter:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_three_source_cooling_element/forward.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_three_source_cooling_element/rk4.py 1
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_three_source_cooling_element/rk4.py 2
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_three_source_cooling_element/admission.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_three_source_cooling_element/alternate_ec.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_three_source_cooling_element/compare.py
```

Complete forward, RK4-one, RK4-two and alternate-EC experiments take 11.31, 5.48, 9.77 and 31.70 s; each is below 120 s. Raw JSON retains actual stage packets, independent EOS roots, numerical gates and exclusions. `before.json`/`CHECKPOINT.json` pin evidence and frozen dependencies. Initial quota/time verification at 00:28:18 UTC matched 22% used on the same account/window; no reset, install, heavy build, production, paper or Git operation occurred.
