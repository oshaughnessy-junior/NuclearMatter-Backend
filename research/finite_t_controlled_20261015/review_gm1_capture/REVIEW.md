# Independent numerical review: selected GM1 capture integrals

**Accept the selected pilot with its stated approximation and convergence limits.** This review has30 passing completed checks:7 unperturbed factory/convention checks,20 current/rate/refinement bookkeeping checks and3 comparisons of a complete new outer integral. One attempted stronger independent refinement stopped at the108s experiment guard and remains incomplete. `results.json.passed` remains false because its experiment did not complete; `selected_comparison.json` explicitly harvests only its completed first integral. No author source or failure evidence was edited. Source hashes before and after agree.

## Independent outer integration

The new calculation uses the reviewed differential response as a **shared physical kernel**. It independently integrates **direct total lepton energy El with adaptive QUADPACK vector quadrature**, rather than the author's fixed Gauss sqrt(El−ml) panels. It uses separately chosen fixed physical energy partitions and closed-form collinear-boundary energies, without calling `capture.integrate_state`, reading its energy panels or copying its root scan. The cosine integral at each El explicitly restricts the equal-positive-mass support. Both j and chi are integrated separately after scaling their positive differential logs; neither coefficient is constructed from KMS.

The completed new electron pilot is T8MeV,nB.22fm^-3,Ye.17,Ymu.04,Enu40MeV, direct El up to600MeV and12 cosine nodes. Results:

| coefficient [s^-1] | new adaptive direct-El | frozen pilot | relative difference |
|---|---:|---:|---:|
| true chi |958899.1217295825|958899.1195336229|2.29008e−9|
| j |745802386.2662604|745802384.5583102|2.29008e−9|

Both pass the unchanged1e−6 comparison target. The scaled adaptive outer-error estimate is1.44e−12; this estimator concerns integration at the chosen12 angular nodes and shared inner response, not physical model error or all quadrature dimensions. The separately accumulated integrated KMS residual is zero at reported precision; its ratio is pointwise constant and is not an accuracy proof.

The next20-angle/direct-El900MeV refinement started and completed only low-energy intervals before the guard. Its partial sums remain in results.json; **no20-angle total, new independent tail comparison, or independent two-resolution convergence claim is made**. The total bounded experiment took108.05s. This review therefore strengthens the electron pilot with an independent outer-integration ordering, while its angular/inner/tail refinement qualification still uses the author's retained evidence. It does not turn the pilot into a general accuracy certificate.

## Forward state and conventions

The unperturbed material factory is recomputed and all existing nuclear root, field-energy, Euler, charge and antiparticle diagnostics pass before the rate comparison. Independent Ye/Ymu are retained; Yp=.21. The weak bare mass is exactly939MeV, distinct from mD568.55724158MeV. This is the common-bare-mass GM1 convention, with no physical vacuum neutron-proton Q inserted.

At a separate differential node Enu40,El180,cos−.3, the returned mapping is explicitly checked: physical omega=Enu−El; kinetic w=omega−Up+Un; scalar form-factor invariant is physical omega²−q² while WM/PS vertices use kinetic w²−q². The code implements the declared Guo/Hartree current ansatz. These bookkeeping checks are not an independent derivation of a unique conserving interacting current or a repeat absolute weak normalization calibration; the prior differential/current reviews remain prerequisites.

Equal positive kinetic masses require q>|w| for a finite-momentum particle-hole transition. The resulting cosine boundary and collinear energy roots are exact support restrictions, not fitted widths. The independently computed electron collinear energies27.76211388019217 and67.76072567904059MeV are included as adaptive interval boundaries. The EOS and free response omit baryon-antibaryon pair cuts. No current-conservation projection removes tensor components.

## Muon convergence evidence, true opacity and retained failures

I independently recomputed comparisons from the frozen muon raw outputs, preserving the original1e−6 gate:

- q16→q20 is still **failed**, about1.82e−6. The earlier failure is not erased.
- q20→q24 passes, about2.04e−7.
- separately sampled cosine20 versus q24 passes, about9.47e−7, close to the target.

The new independent adaptive outer integral was performed only for the electron pilot. No new full muon integral is claimed. Acceptance of the muon point remains narrowly tied to its recorded coordinate/refinement tests, including that small margin; a coordinate difference below1e−6 does not mathematically bound all errors by1e−6. The initial exploratory support/angle grids, zero-field order8/12 failures and metadata-branch repair source remain pinned in the review manifest.

For each of the four muon raw runs, chi is checked separately from j+chi, with the latter equal to their positive sum. The pilot chi≈1.6495843e7s^-1, j≈8.2200221e7s^-1 and relaxation≈9.8696064e7s^-1 must remain distinct. The electron difference is larger still. These are occupation coefficients; chi/c is true inverse length, whereas(j+chi)/c is relaxation inverse length. No incident-neutrino DOS or neutrino occupation belongs inside them.

The zero-field anchors explicitly use mD=bare939MeV and exactly zero vector potentials. They are matched common-mass free baths, not physical unequal-mass vacuum-Q benchmarks or a fresh GM1 EOS recalibration. The preserved pre-repair empty-domain reporting branch was replaced with an explicit warning that empty samples on a finite truncated domain do not certify global physical zero. No positive pilot number changes follow from that metadata correction.

## Accepted scope and remaining gates

The two selected **capture/inverse-capture** coefficients under their declared Hartree/current prescription are reviewable. The independent outer electron comparison passes; the more stringent independent refinement is pending, not inferred from its partial sum. This review makes no new uniform parameter-domain, full muon outer, total antineutrino inventory, crossed beta, spectral moment, feedback/evolution, interacting-current uncertainty, external finite-T GM1 forward agreement, derivative or production claim. A40MeV muon-capture rate in this material does not isolate thermal threshold broadening: mean-field release, degeneracy and target motion all contribute. The physical source decomposition and a matched external GM1 forward gate remain separate work.
