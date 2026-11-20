# Independent verification of scoped repairs and physical spectral slab

15 October 2026, 11:04 UTC. This addendum supersedes the two repairable findings in `REVIEW.md` for the files pinned by `repaired_source_hashes.json`. Original `review_results.json` remains unchanged, preserving the pre-repair failures.

## Repair disposition

**Both scoped repairs are independently verified.** Rerunning the preserved numerical audit produces 30 passing checks and no failed checks. All original density-inversion counterexamples now reconstruct the requested density within the audit's 1e-8 gate. The original NaN/infinite energy and integration-control counterexamples now raise exceptions. The supplemental independent momentum quadrature, massless weak moment and selected dilute heat-capacity checks still pass. `repair_verification.json` stores the run; `probe_repaired_reference.py` reproduces it without overwriting original evidence.

The three saved extreme weak-rate underflows remain. This is an explicitly excluded relative-accuracy/log-rate domain, rather than a newly repaired capability. The repairs do not establish a filled thermodynamic or weak-rate parameter domain.

## Spectral slab disposition

**Accept `spectral_slab/run_slab.py` as a fixed-bath physical-rate coupling benchmark at its specified state**, T=3 MeV, nB=1e-5 fm^-3, Ye=0.15 and four reference optical depths. A copied script run in the review directory independently reproduces all 40 checks; output is `spectral_slab_reproduction.json`. No parent or author output was overwritten.

The angular, DOS, two-face and length conventions are consistent. The one-helicity isotropic number DOS `D E²` includes integration over the full solid angle; a single outgoing face has flux `c D E²/2 int_0^1 mu f dmu`. Summing two equal vacuum faces yields `c D E² int_0^1 mu f dmu`, as implemented. Spatial optical depth is `(j+chi)L/c` with rates in s^-1, L in cm and c in cm/s. Multiplication of the fm^-3 DOS by 1e39 converts to cm^-3 before multiplication by c. The resulting flux units are cm^-2 s^-1 and MeV cm^-2 s^-1.

An additional independent normalization test increases the reference depth to 1e8 and compares the solved escaping flux with scalar QUADPACK Fermi-Dirac integrals over the allowed neutrino bands. Both faces of a fully equilibrated isotropic active band give `c D/2 int E^k f_eq dE`, for k=2 (number) and k=3 (energy). The antineutrino gap is excluded exactly. All four flavor/moment comparisons agree to 4.2–4.7e-15 relative error; saved as `spectral_slab_blackbody_checks.json`. This checks the factor of two, helicity normalization and broadband energy weighting without relying on the slab's angular quadrature or energy nodes.

The thin spectral number/energy integrals reproduce independently integrated weak target moments in the parent script; its depth source/boundary identity and analytical E3 comparison validate the fixed-coefficient attenuation calculation. This qualifies a broadband fixed-bath escaping flux, not a thermal feedback solver or a full conservation test for evolving matter and radiation.

## Remaining scope boundaries

- The slab state has positive electron chemical potential. Its generic energy-grid upper cutoff uses `max(mu,0,threshold)+70T`; large negative mu would require the positron Fermi scale `-mu` in that cutoff. Arbitrary chemical-potential signs and very degenerate surface resolution are not accepted by this single-state demonstration. This is a range restriction, not a failure at the reported state.
- `solve()` assumes valid positive slab length and valid integration orders. Guard these if exposing it beyond the controlled driver.
- The spectral antineutrino gap follows the static no-recoil reaction ansatz. It must not be interpreted as a physical universal transport feature.
- Coupled fluid-element/box evolution still needs trajectory-wide charge residuals, finite input/output gates, composition-energy accounting, time refinement, and matter+radiation conservation. The new slab result does not remove those gates.

No author code changed in this review.
