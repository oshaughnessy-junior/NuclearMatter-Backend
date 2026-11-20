# Independent numerical review: continuum NC redistribution

**Scoped acceptance, with an excluded colder extension.** The frozen `nc_continuum_grid/redistribution.py` source SHA256 is `0e72e2846e784f8961e5dab95129f4863e3ddb43754114a4c75a7d47d51ab544`. All measured author source/data hashes are unchanged. No author or upstream edits were made. Existing 248 accepted checks and 69 failed coarse-grid gates retain their original meaning; this review does not relabel them.

The new experiments have 110 passing checks, six preserved failed independent-coarse-rule refinement checks, and one explicitly excluded broader-domain OverflowError. Each experiment finishes under 7 seconds. Exact raw records and source hashes are in this directory.

## Independent coordinates and held-out states

The author's outgoing-energy-outer/angular-inner integral was compared with a new **q-outer, outgoing-energy-inner** quadrature. The physical triangle is q>=0 and |E−q|<=E'<=E+q, with E' limited by the chosen physical upper cutoff. The Jacobian is dc=q dq/(E E'). Both kernels are evaluated directly: forward K(E→E',c), reverse K(E'→E,c), including reverse gain DOS ratio E'²/E². Log FD occupations supply incoming f and outgoing(1−f). The reverse kernel is never reconstructed from KMS in the independent integral. Its cancellation-stable net uses the difference of independently calculated forward and reverse log rates.

New states are:

| Bath T [MeV] | neutron n [fm^-3] | incident E [MeV] | radiation (Tnu,munu) [MeV] |
|---:|---:|---:|---:|
| 1.1 | .0017 | 13.7 | (1.8,2.3) |
| .2 | .0042 | 19.6 | (.7,5.1) |
| 1.1 | .0017 | 13.7 | (1.10000011,3.2) |

Independent momentum-coordinate density integrals first confirm the underlying FD density inversion at those baths and T=.08. These checks use a momentum integral rather than the EOS kinetic-energy-coordinate integral, and all six pass a3e−11 relative gate. These are free NR gases; finite pF/M and T/M remain ansatz limitations, not empirical model errors.

Warm q56/E32 agrees with all seven author moments to1.78e−11 relative. At the near-LTE fractional temperature displacement1e−7, net C(E), computed with the actual reverse response, agrees to2.62e−8 relative; the positive moments agree to~1e−11 or better. Exact LTE with arbitrary neutrino chemical potential3.2MeV has a direct-reverse net/gross residual5.96e−16. Nine selected direct forward/reverse kernel log ratios verify reverse DOS plus exp(−omega/T) to3.56e−15 absolute log error. Six separate final-neutrino blocking and incoming-loss checks pass.

The colder T=.2 case demands resolution: independent q32/E20→q56/E32 has six failed3e−5 refinement gates, with errors1.23e−4 to2.60e−4. Those failures remain in `results.json` and `initial_results_preserved.json`. Refined q96/E48 and q144/E64 both pass unchanged3e−5 gates against the author; their maximum author discrepancy is1.10e−6, and independent96→144 changes are smaller than the gate. This qualifies this selected colder point numerically, not a uniform cold domain or a new global release range.

An actual1.5-fold absolute upper-energy extension, not merely a nominal tail parameter change, leaves the independent moments unchanged at floating-point resolution in the three held-out cases. This is finite-cutoff evidence only. It neither bounds the infinite tail nor proves relative accuracy of exponentially small contributions. Direct log-tail evidence is retained separately.

## Excluded colder-domain overflow

At T=.08MeV,n=.0042fm^-3,E=19.6MeV with FD radiation(Tnu=.4,munu=5), the public author adaptive call accepts the inputs but throws `OverflowError('math range error')`. Its width-dominated upper cutoff is118.27309753794682MeV. The author integrand separately evaluates exp[(E'−E)/T]; at E'=77.2MeV this exponent is720 and overflows before multiplication by an underflowed forward rate. At c=−.3 the forward/reverse log kernels remain finite (−2983.0580 and−2265.7998), as does the actual gain log integrand−2443.5580. It is a floating-point representation failure, not physical support ending.

The independent combined-log calculation remains finite and its q96/E48→q144/E64 moments agree to3.22e−6 at this bath. That does **not** repair or qualify the frozen author API: the T=.08 extension is excluded. Parent was notified; future colder-domain use requires a separately scoped combined-log author repair and independent recheck. The previously qualified selected T>=.5 states retain acceptance. No broader API-domain certificate is implied by finite-input validation.

## Reporting failures and limitations

Two review-only reporting bugs were preserved and corrected: helper-prefix import registered an atexit writer that overwrote the first initial report; the unchanged initial experiment was replayed to restore it, and the atexit registration was disabled in the repaired refinement driver. An independent EOS check initially produced a numpy.bool JSON serialization error; the original script/error is retained and the explicit-bool replay passes. These are reporting failures, not author physics changes or hidden numerical repairs.

This review reuses the already reviewed physical free-NR response and tests a new integration ordering. Agreement assesses the numerical continuum operator at the stated points; it does not measure physical uncertainty or validate interactions/current corrections. It does not qualify a global double-energy heating integral, radiation time evolution, an evolving-matter ledger, global number conservation on open grids, arbitrary nonthermal spectra, or a cold/narrow-response parameter domain. The six independent coarse failures, original author coarse failures and colder overflow remain explicit.
