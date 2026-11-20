# Independent review: restricted transparent antineutrino fluid element

**ACCEPTED for the stated short, fixed-density, clean-process research experiment.** Frozen author element SHA256 `20295acc73d60235448485da397c201ff934275113b0f9d6568b0a207f3c7635`. This is a coupling/integration review; it relies on the separately accepted physical beta and positron-capture integrals, not an independent new weak-matrix-element qualification.

The new review probe passes **42/42** gates in **42.98 s**. It imports the library by a unique absolute module name, never imports or runs a `study` driver, and verifies all author/hot-beta top-level file hashes unchanged before versus after. The restored upstream beta results have the exact required original SHA256 `57e2e89e820b0e80b9d0cd4956d67c54e208cd78d5e2609edcd8e1ef54f94d88`. The earlier overwritten payload and import failure remain preserved in the author leaf.

## Physical closure and signs

Both included reactions, neutron beta decay and neutron positron capture, turn one neutron into one proton and increase material **net** electron lepton fraction by one event per baryon. The emitted antineutrino has electron lepton number minus one. The particle baryon EOS and electron/positron pair EOS use total rest-inclusive energy. Thus fixed-density `du/dt=-Q/nB` and `dYe/dt=R/nB` are consistent with the direct event material-energy loss; adding a separate vacuum Q would double-count rest energies. Net pair charge, not the actual electron population alone, equals proton density. Pair annihilation/creation implied by rapid local rethermalization is part of the declared EOS closure, not an additional weak reaction.

From the Helmholtz differential, `(∂epsilon/∂n_i)_T=mu_i-T(∂mu_i/∂T)_n`. Applying `dn_p=-dn_n=dn_e,net=nB dYe` gives the author's composition derivative including its neutron minus sign and pair-electron term. A new heldout endpoint direct EOS perturbation agrees with `u_Y` to 4.60e-7 relative and `u_T` to 4.29e-10. Its composition-energy contribution is several percent of the radiative cooling term and is retained in the conservative inversion.

## New forward trajectory and time error

The review integrates the actual changing-state physical sources with an independent explicit **Heun** method in `(u,Ye)`, using its own bracketing and direct species-energy sum for temperature inversion. It evaluates new predictor and corrected states, rather than replaying saved sources or the author's Euler update. The underlying reviewed EOS and physical source integrals are shared intentionally; this comparison tests time integration and coupling, not external physics/model uncertainty.

| Heun steps over 0.002 s | Ye final | T final (MeV) | u final (MeV/baryon) |
|---|---:|---:|---:|
| 2 | 0.321807694298 | 1.995569364215 | 965.893049372026 |
| 4 | 0.321808926719 | 1.995569110678 | 965.893036068491 |
| 8 | 0.321809232606 | 1.995569047711 | 965.893032764705 |

Successive difference ratios are 0.24834 in energy, 0.24820 in composition and 0.24835 in temperature, close to the second-order value 0.25. The author's four-step Euler endpoint differs from Heun8 by approximately **1.309e-4 in Ye, 2.798e-5 MeV in T and 0.0014644 MeV/baryon in u**. Those discrepancies confirm its quoted first-order error scale. Euler first-order Richardson extrapolation agrees with Heun8 within 1.8% of the last Euler-grid difference. Conservation precision must not be presented as trajectory accuracy.

Every Heun endpoint separately satisfies baryon number, actual pair-charge neutrality and material-plus-escaped electron lepton number. The largest independently reconstructed energy residual, normalized by escaped energy, is 1.07e-10. This ledger partly follows the conservative variable definition; direct endpoint EOS evaluation provides a separate inversion check, while refinement supplies the independent time-error evidence.

At the new Heun8 endpoint, refined physical source number is **1.06421633304e-6 fm^-3 s^-1** and neutrino energy is **1.14001142188e-5 MeV fm^-3 s^-1**. Each individual channel's R/Q passes fresh base-versus-fine quadrature checks at 1e-7 relative. The beta contribution remains nonzero (`Rbeta=4.11558e-11`); PC dominates. The source declines by several percent along the path and cannot be described as frozen initial forcing.

## Provenance and execution boundary

The tested fresh process loads the exact intended hot-beta, crossed-kernel and PC-reference file paths. No generic `study` module is loaded. The runtime fingerprint includes the transitive vacuum-current source, loaded campaign Python and the declared source directories/environment. Synthetic stale identity is rejected **before** a cached source return. Every physical source in this experiment is recomputed; no persisted joined source is substituted.

Acceptance is explicitly for clean-process execution with states constructed by the reviewed factory. Imports still use some canonical upstream names (`hot_moments`, `kernel`, etc.) and mutate `sys.path`; this is not a general isolation mechanism for arbitrary pre-populated Python interpreters. The source cache keys exact T/Ye/nB and controls, but takes mutable factory-state dictionaries rather than independently authenticating arbitrary caller-supplied bath/derivative dictionaries. Do not promote that interface as an adversarial cache or arbitrary-state API. Supplemental `extra_controls.py` is an executable driver with top-level work, not an importable library; import only `element.py` in fresh controlled processes. These restrictions do not invalidate the demonstrated fresh process or the complete immutable source record.

No blockers remain for this selected evidence. Open: full electron-flavor weak inventory (including electron capture and neutrino reactions), pair-neutrino source coupling, radiation evolution, transparent kinetic equilibrium, long times/stiffness, physical thermalization rates, broader density/temperature domain, and interacting matter. Empty incoming radiation is an imposed sink; a finite total radiation entropy derivative at exactly zero occupation is not claimed. No extra entropy or hydro qualification follows from these tests.
