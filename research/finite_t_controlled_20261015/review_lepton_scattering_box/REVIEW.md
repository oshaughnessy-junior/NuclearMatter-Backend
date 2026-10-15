# Independent review: finite closed νe–e± spectral box

**ACCEPTED for the explicitly finite closed-box research scope, with a roundoff qualification on derived DOS weights. No continuum result is accepted.** Reviewed box.py SHA `7c724c889678581ecec4c5a645a5e99e940f4ae8142fe39278ff4870342ea77f`; commissioned MANIFEST SHA `5770812ee3fbefb9b7fbd08aafbce6a6eaaf1d316a6d0ab79d554ca4eebb5317` independently matches. All frozen source/raw bytes remain unchanged.

New fresh physics/ODE evidence: **47/47 passing checks in0.93seconds**. A separate saved-boundary arithmetic audit has **20 passing checks and3 failed strict bitwise-W assertions**, preserved in boundary_results.json. The three strict failures are only1–2ULP recomputation differences in derived DOS weights, diagnosed below; they are not suppressed or counted as passes. The original six authored numerical failures also remain preserved/excluded.

## Physical rates, measures and occupations

The accepted elementary scattering kernel D already includes final-neutrino phase measure and target-spin2, and reports per incident one-active-helicity neutrino per outgoing energy/cosine. Consequently the node measures must be `Wi=wi Ei²/[2π²(hbar c)³]` and directed physical events `Aij fi(1−fj)`, where `Aij=Wi wj Kij`. The outgoing multiplier is quadrature width wj, **not another Ej² DOS**. The reviewed implementation has these factors once. Both e− and e+ target FD occupations and final target holes enter the independently computed K; positive-lepton chemical potential is −mu_e. The pair EOS/adaptive actual populations pass their forward gate before edges.

A new five-node21MeV grid is actually computed using angle40/target32/phi8/tail80T. Three fresh forward **and reverse** physical edges are compared against the review's own adaptive q-outer/transverse-r shell reduction, with analytic target azimuth. It does not call the author's differential function or edge reference. Maximum discrepancy is9.55e-11 relative. The independent elementary-current normalization/crossing qualification remains in review_neutrino_lepton; no new EW accuracy model is inferred here.

Both directional K entries are evaluated from physical integrals; they are not filled by detailed balance or symmetrized afterward. Fresh reverse references satisfy the required `(Ei/Ej)² exp[(Ej−Ei)/T]` ratio. With the measured reciprocal relation, physical event coefficients obey `Aij/Aji=exp[(Ei−Ej)/T]`, so same-T FD distributions with arbitrary chemical potential are stationary. Two new chemical potentials−1.8 and+.4MeV pass independent dense gain/loss tests. No collision residual is forced to zero in that reference.

For a fresh non-FD distribution, a dense forward-event matrix `Fij=Aij fi(1−fj)` gives `dni=sum_j Fji−sum_j Fij`. It agrees with the author's unordered-pair/expm1 update, including both radiation hole factors. Omitting final holes would change this nonlinear equation; the review does not make that approximation. Diagonal same-energy transitions cancel for isotropic occupation and energy, but physical scattering at equal incoming/outgoing energy need not vanish. Their omission from the generator does not justify reporting its offdiagonal activity as a total continuum scattering frequency.

## Independent time integration, thermostat and entropy

The review integrates **particle populations ni=Wi fi**, plus thermostat heat and total entropy-production quadratures, with adaptive DOP853. This differs from the author's fixed-step SSPRK2/Heun update in occupations and uses an independent ordinary dense event assembly. The bath remains T2MeV with fixed pair composition; no bath heat-capacity evolution or material work is implied.

Fresh initial radiation temperatures1.6 and2.8MeV, mu−.55MeV and duration0.06seconds give:

| Initial radiation T | Independent bath heat (MeV fm^-3) | Heun32 relative heat error |
|---|---:|---:|
|1.6MeV|−7.06684343385e-9|−1.51e-5|
|2.8MeV|+1.11662710379e-7|−2.92e-5|

Colder radiation extracts energy from the thermostat; hotter radiation heats it. The conserved total is radiation energy plus signed thermostat heat. Dense particle-number conservation and the independently accumulated heat agree with endpoint energies. Heun8/16/32 errors against DOP853 show orders2.017/2.009 and2.046/2.023 in weighted occupations. Thus time-error claims have a different-integrator witness, rather than only an algebraic energy cancellation.

For the actual nonlinear state, independent microscopic `sum(Fij−Fji)log(Fij/Fji)` and macroscopic radiation entropy derivative plus heat/T agree and are nonnegative. DOP853 entropy-production integrals agree with actual endpoint radiation entropy plus bath entropy to2.7e-13 relative or better. Stationary and nonequilibrium tests use actual physical rates. In the author's discrete Heun evolution, the integrated entropy-production/endpoint discrepancy remains a finite-step error, not an exact invariant. Positivity is checked rather than clipped; the selected runs remain inside0<f<1. Scattering changes no species, neutrino number or flavor lepton number and supplies no material composition source.

## Finite boundary: essential scope and a strict roundoff finding

Independent reconstruction of saved extended-grid forward and reverse flows across24MeV gives `2.10057846312e-8` and `9.55517128741e-8 fm^-3s^-1` for the specified hot state. The dense core number derivative equals incoming minus outgoing cross-boundary flux; full extended-box number still conserves. These are **two omitted directional exchanges** in the closed24MeV model, not an implemented one-way boundary escape sink. Nonequilibrium omitted exchanges do not cancel merely because LTE would balance them.

The arithmetic reproduces the actual hot heat sensitivity:

- closed24MeV heat:2.37621292410e-7MeVfm^-3;
- closed36MeV heat:2.80668710133e-7;
- closed48MeV heat:2.83929988241e-7;
- 24→48 change: **19.4884%**; 36→48 change: **1.16197%**.

These differences are far larger than time error and prohibit an infinite-energy convergence claim. Passing selected12→16 heat and6→8 tail-order gates qualifies those heat comparisons only, not every spectral shape or omitted high-energy transition. The six earlier authored failures remain failed: two6→8 heat, one8→12 heat, two tail4→6 and one coarse actual-edge gate.

For the boundary comparison, **energy nodes E, Gaussian quadrature widths w and core K entries are bitwise fixed**, as verified. The derived DOS weights W are recomputed using `w*E**2` in the extension versus `w*E*E` in the original grid. Our strict bitwise-W assertion therefore fails in each of three saved cases: maximum relative difference2.22e-16, absolute1.69e-21fm^-3, with1–2ULP differences. Exact E/w/K preservation and roundoff-equivalent W are the supported statement; a literal assertion that *all derived weights* are bitwise unchanged is not endorsed. This is a disclosed floating-operation difference, not a physical resampling or explanation for the percent-level cutoff sensitivity. The raw failed assertions and weight_roundoff_diagnostic.json are preserved without changing author files or relaxing those assertions.

## Remaining gates and reproducibility

The figure script plots saved E²f at the actual eight nodes and explicitly titles a closed0–24MeV box; connected lines do not supply an independently converged continuum spectrum. No figure or author source is changed by this review.

Accepted scope: selected physical free νe/e± edges; actual pair-lepton thermostat; finite isotropic spectral generator with both neutrino holes; number/energy/entropy consistency; independent new time convergence; explicitly finite boundary sensitivity. Open: continuum energy range and every-bin shape convergence, angular anisotropy, spatial streaming, open external irradiation boundaries, evolved matter, other flavor/antiflavor box inventories, plasma/medium/QED/EW corrections, extreme underflow/cold domains and production packaging. This remains a useful controlled transport ansatz rather than full hydro. Fresh-process legacy imports are required; no general module-isolation or untrusted-code sandbox claim follows.

Reproduce with `python probe.py` and `python boundary_audit.py` using the existing campaign interpreter. All experiments are subsecond apart from startup. Unique spec imports load function modules only; no frozen driver is imported. `results.json` records new physical grids, references, DOP853/Heun trajectories and immutable source/raw hashes; boundary_results.json preserves its three strict failures separately. CHECKPOINT.json pins this review and its qualified acceptance.
