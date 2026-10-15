# Independent numerical review: selected Urca-threshold spectra

2026-10-15. **Accept the restricted nine-point numerical demonstration.** The colder T=0.01 MeV capture extension remains excluded. This review adds one held-out finite-temperature state, checks forward densities first, and independently checks cold support. It does not establish an interacting EOS, universal threshold law, absolute-current physical accuracy, or a colder integration domain.

Pinned author `urca_threshold/threshold_study.py` SHA256: `d658eceaf83574971940540fdc7f413cec47df6ba7ce0f955d7c1f5a9b787d1a`; identical after the review. No author or upstream file was edited. All review work is confined to this new directory.

## What was checked

The archived qualified set contains exactly Yp=(0.10,1/9,0.12) × T=(0.03,0.1,0.3) MeV, nB=0.01 fm⁻³, Eν=3T. Its 48 author gates pass. Rather than replay all nine optical calculations, this review first reconstructs their forward states and performs new energy-coordinate FD density quadratures for neutrons, protons and net electrons. These differ from the author root and author independent momentum-coordinate quadratures. The largest density discrepancy is 4.66×10⁻¹⁴ relative; reconstructed archived chemical potentials agree exactly at the recorded precision.

The cold offset is reconstructed directly from independently evaluated Fermi momenta and the common kinetic mass: Δ=(pFp²−pFn²)/(2M)+sqrt(pFe²+me²), pFe=pFp. Agreement with the author calibration is within 7.11×10⁻¹⁵ MeV at every point. The source and state construction hold this offset fixed during each composition’s T scan. It is neither the vacuum neutron–proton Q value nor a GM1 potential. Different Yp values intentionally specify different local Hamiltonians; no derivative or EOS sequence across them is qualified.

A new composition, temperature and Eν/T ratio are tested: **Yp=0.105, T=0.047 MeV, Eν=0.1269 MeV=2.7T**, with independently calibrated Δ=55.5453585862 MeV. Before evaluating a rate, its three independent density residuals pass, worst 6.44×10⁻¹⁵ relative. The author order72 capture coefficients are compared with a separate direct-cosine angular rule and adaptive electron-energy integration in y=(Ee−μe)/T, with a factored logarithmic scale. This comparator does not use the optical panel generator or its energy quadrature; it also differs from the author’s independent q-coordinate angular rule. Angular orders64→96 and tails60T→120T agree at recorded precision. The independently integrated true χ agrees with author72 to 7.11×10⁻¹⁵ relative, with a reported adaptive error 1.68×10⁻¹¹. The held-out rates are j=110.234872862 s⁻¹ and χ=1668.12837102 s⁻¹; KMS passes using the actual finite-T affinity, without imposing zero affinity.

This comparator shares the reviewed analytic nuclear response and weak prefactor. Agreement is a numerical integration check, not independence from their physical approximations or absolute weak-current normalization.

## Cold-support crossing

A separate dense electron-energy/direct-angle grid tests the zero-T response interval, without using the author’s scalar optimized-q support routine. Energy conservation restricts the final electron to [μe0,μe0+Eν]. At each energy and angle, the independently calculated longitudinal initial-momentum bound and initial/final Fermi spheres determine whether any phase-space interval remains.

At Yp=0.10, the onset is 0.308918980 MeV: support is zero at 95% of that value and positive at 105%. At the new Yp=0.105, the onset is 0.167419783 MeV, with the same below/above crossing. At Yp=0.12, positive support already exists at the selected 0.03 and 0.3 MeV neutrino energies. The held-out finite-T point Eν=0.1269 MeV lies below its cold onset and nonetheless has a finite, independently converged χ. Thus its opening cannot be attributed solely to an incoming neutrino already exceeding the cold onset.

These are support-sign checks using a resolving dense grid. They do not certify an arbitrary low-T rate or a global onset formula for a different dispersion/current model. The tiny-neutrino cold triangle and finite-neutrino absorption onset remain distinct.

## Exclusion and limitations

The qualified JSON contains no T=0.01 point. The author plotting code reads only the passing qualified JSON and explicitly selects T=(0.03,0.1,0.3); it does not read the failed cold-extension output. The preserved cold-extension report records order48→72 error 0.00662339 and order72 versus adaptive error 0.000360226, both exceeding the unchanged 3×10⁻⁶ gate. Correct density roots and KMS do not qualify that failed optical integration. This review neither repairs nor reclassifies it.

All **80/80 independent checks pass** in 1.77 s (about 3.9 s across the initial and final review runs, far below the 90 s numerical bound). No new numerical failure was observed. Raw tests, results and source hashes accompany this note. This acceptance applies to the named free common-mass NR constant-offset ansatz and selected warm spectra only. Antineutrino/crossed inventory, full neutrino-energy moments, interacting-EOS mapping, collision broadening and fluid evolution are outside this threshold review.

Reproduce with `python outputs/nmb-finite-t-20261015/review_threshold_numerics/probe_threshold.py` from the workspace root.
