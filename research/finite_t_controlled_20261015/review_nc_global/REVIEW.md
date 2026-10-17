# Independent review: global NC finite-energy exchange

**Accepted for selected prescribed spectra in a fixed free-NR bath and finite energy domains.** Twenty independent checks pass; the new numerical experiment costs21.85s. No author/frozen source was edited. The author's245qualified checks,8coarse failures, interrupted first orchestration, partials and reporting-failure evidence retain their original classifications. This review does not qualify evolving matter, infinite-energy limits or arbitrary cold states.

## Measure and sign audit

For one active helicity, integrating the incident solid angle gives4pi/(2pi hbarc)³=1/[2pi²(hbarc)³]. Multiply that incident measureE²dE exactly once by the bare kernel, which already includes outgoingE'² and outgoing azimuth. The reverse event uses its **own** incidentE'² and actual reversed kernel. There is no additional outgoing DOS factor and no incoming spin-two factor.

On the half triangleE<E', J=A−B moves radiation energy(E'−E)J and bath energy(E−E')J. Radiation entropy is[h'(f')−h'(f)]J; bath entropy isQ/T for the **prescribed-temperature reservoir**. Their sum isJln(A/B)>=0. This does not evolve a finite-capacity material entropy/temperature. On the full square the equivalent entropy production has the usual1/2 pair-counting factor; the direct occupation-source integral includes that factor through exchange symmetry rather than a manual multiplier. Positive full-square forward/reverse event counts each match the half-triangle sumA+B to1.12e-7 relative in the independent case.

## New bath, asymmetric full square and refinement

The held-out bath isT=1.7MeV,n=.0019fm^-3,free neutronM939MeV,offset0. Prescribed radiation isFD(Tnu=2.35MeV,munu=2.8MeV). The first physical box is0<E,E'<24MeV, then extended to36MeV. These are intentionally finite boxes with measurable boundary contributions.

`probe.py` integrates the actual forward and reverse kernels directly, without calling the author's `events`, `signed_difference` or physical integrator for its independent values. Incident integration uses a newlog(1+E) variable and different energy panels; outgoing integration uses a directE' rule with separate redistribution panels. Incoming/outgoing orders are8/16 then12/24, with cosine32/64. Distinct rules on the two axes mean number cancellation is **not imposed by a symmetric quadrature or antisymmetric accumulator**. Both occupation/blocking products are assembled from log kernels before exponentiating.

The independent fine box24 result is:

| Quantity | Independent value |
|---|---:|
| global radiation number [fm^-3s^-1] |−5.80487e-13|
| radiation energy [MeVfm^-3s^-1] |−.00199059345865|
| bath heating [MeVfm^-3s^-1] |+.00199059345865|
| radiation entropy [kBfm^-3s^-1] |−.000847061045541|
| bath+radiation entropy [kBfm^-3s^-1] |+.000323876283075|

Energy/entropy refinement changes are≤7.70e-6 relative, within the unchanged3e-5 gate. Comparison to a newly evaluated author paired16/32/q96 integral is≤7.03e-9 relative. The nonzero direct number residual is2.18e-11 of the positive event scale, not forced to zero. Positive event counts provide a non-canceling normalization comparison. FD entropy with finite numerical number residual obeysSrad+Q/Tnu+munu*Nrad/Tnu=0 to3.55e-17 relative, independently confirming the chemical-potential sign. Hotter radiation heats this bath, and the total entropy source is positive.

## Finite-box leakage and independent block decomposition

An independently integrated low-high crossingE∈[0,24],E'∈[24,36] has retained-low-box number source **+5.62130027082e-5fm^-3s^-1**. This is not global particle creation: the exterior has the opposite contribution. Its energy/entropy outputs count the whole crossing pair, whereas its first component counts only retained particles, as the author note explicitly declares.

An independently integrated upper-upper square[24,36]² completes the decomposition. Combining independent low-low, complete low-high and upper-upper energy/entropy agrees with a fresh paired full36 integral within1.90e-8 relative. Bath heating rises from.00199059346 to.00231557591MeVfm^-3s^-1, a change.000324982447MeVfm^-3s^-1. Thus an exactly closed low-box ledger cannot establish negligible physical boundary leakage. The review compares energy/entropy components of that decomposition only; the crossing first-component convention is not incorrectly summed as a full-domain number source.

## Accuracy and open gates

The independent energy-opposition check is algebraic because the same material/radiation increment is used; it is not treated as separate accuracy evidence. The independent rules, actual reverse kernels, gross event counts, non-forced number residual, FD entropy identity and block comparison provide the relevant forward checks. These calculations assess numerical accuracy within the same specified microscopic model, not physical uncertainty or an external response code.

Author results for72/108/144MeV and other spectra remain their own separately qualified evidence; this review does not repeat those expensive grids or claim a new infinite-tail bound. The new held-out example demonstrates finite-box sensitivity rather than closing it. Cold/narrow redistribution, arbitrarily tiny departures from LTE, in-medium/relativistic current corrections, radiation evolution and a coupled material ledger require further work.

`results.json` checkpoints every completed incoming panel/probe and retains both independent resolutions, fresh paired references and crossing/upper blocks. `bookkeeping.json` adds4measure/event/entropy checks. All20checks pass with no new runtime or physics failure; selected author/failure/upstream hashes match before and after. `MANIFEST.json` pins the review evidence and source hashes.
