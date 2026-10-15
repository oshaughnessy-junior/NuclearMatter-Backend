# Independent numerical review: FD-baryon selected-energy quantum slab

15 October 2026. `probe_quantum_slab.py` executes this scoped review; evidence, source hashes and the preserved control counterexample are in `quantum_slab_review.json`. No author files changed.

**The selected valid-parameter transfer cases pass 17 independent checks. The general numerical-control failure contract needs one repair before acceptance:** a negative tail can bypass validation through the broad fallback. This is separate from the accepted physical normalization and collapsed-panel treatment at valid controls.

## Guard and numerical measure

The original failing moderate-FD state at neutrino energy 60 MeV, order72 and tail50 independently produces a positive original-E panel width 7.105427357601002e-15 MeV whose transformed sqrt endpoints are identical in float64. Skipping such an unrepresentable transformed interval is a local numerical measure decision, not physical reaction support removal. The guard's electron momentum `x*sqrt(2me+x²)` avoids cancellation relative to `sqrt(Ee²-me²)` near the true mass endpoint.

This review estimates the discarded original-E panel contribution using its actual angular weight, physical integrand and width. Its relative j/chi measures are below 1e-12. Guarded order72/tail50 coefficients also agree with order96/tail60 inside 3e-5. Thus the discarded interval is insignificant at the reported valid test state. That result does not authorize dropping finite panels at arbitrary energies or changing the frozen response physics.

## Held-out rays, fluxes and collision sources

At the same FD matter state, held-out optical depth tau=37.5 and ray cosine 0.37 are tested for neutrinos at E=20 MeV and antineutrinos at E=0.85 MeV. These include an occupation near one and a tiny antineutrino equilibrium occupation. The antineutrino channel inventory explicitly combines the guarded/mapped capture branch with the crossed neutron-decay branch; it does not call the stale old combined helper.

An independent DOP853 integration evolves the actual collision law `df/dt=[j(1-f)-chi f]/[(j+chi)mu]`, with scaled occupation/error tolerances that resolve the tiny antineutrino equilibrium. The resulting endpoint agrees with the ray evaluator.

Independent outgoing angular integrals do not use E2/E3 functions or the author's flux algebra. They reproduce each outward face, using the one-helicity full-angle DOS and one-face factor 1/2. A separate nested angular/depth quadrature integrates the **actual collision term** from both sets of ray occupations, with boundary-layer depth panels. It reproduces the net boundary number gain and the equal/opposite bath energy exchange. No copied analytical E2 source identity is used in this volume calculation. Thin-slab subtraction is avoided at the held-out opaque case, and tiny outgoing flux tolerances are scaled to their own occupations.

This is selected-energy, fixed-bath conservation evidence. Multiplying an individual group's number source by E checks that group's energy convention; it is not a broadband redistribution or evolving-fluid conservation demonstration.

## Reproducible control-validation blocker

The following invalid request unexpectedly returns finite coefficients:

```
matter = matter_state(2.,0.002,0.001)
coefficients(matter,'nue',60.,order=72,tail=-1.)
```

The mapped evaluator correctly raises `ValueError('Invalid quadrature controls')`. `fixed_bath.coefficients` catches every ValueError and retries `guarded_capture`, which does not validate those controls. Because this state also has an unrelated collapsed panel, the fallback sees a nonempty discard record and accepts the invalid request. The saved record explicitly reports original failure `Invalid quadrature controls` and the discarded panel.

This needs numerical-control validation before fallback and/or equivalent validation inside the guard, with a regression that the invalid request remains rejected even when collapsed panels are present. Selected valid-control results remain numerically accepted; an assertion that the callback preserves the original invalid-input contract is not yet accepted. No repair was made by this reviewer.

All prior model boundaries remain: free NR common-mass baryons, leading allowed currents, declared channels, no scattering/redistribution, no mean-field response or phase-complete EOS, no broadband state/evolution coverage.
