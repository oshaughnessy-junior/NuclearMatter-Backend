# Independent review: full source-defined vacuum current

**Accepted at the stated tree-level vacuum scope.** All224 independent selected checks pass:208explicit-spinor/component checks and16crossing/phase-space/zero-limit checks. No blocker was found. Author/current/point-source hashes remained unchanged. This supplies no finite-T matter rate, empirical form-factor calibration, radiative correction or external-code execution.

## Independent construction

`spinor_probe.py` builds its own Pauli/gamma matrices and normalized externalu/v spinors. It directly sums the squared amplitudes over external spin states, averaging the initial baryon by1/2. Both massless incident spin states may be summed because the inactive state is annihilated by the left-chiral projector; there is **no** incoming-neutrino1/2 average. It does not call the author's gamma/slash/adjoint/trace implementation. PointV−A amplitudes are gated first against the frozen scalar invariant before full-current comparisons.

For antineutrinos the explicit leptonic current isvbar(k)gamma_mu(1−gamma5)v(l). The baryonic conjugate transition is independently constructed asbarGamma(−q), rather than manually reversingF2 orGP. The neutrino usesubar(l)gamma_mu(1−gamma5)u(k). This independently checks the crossing/conjugation convention responsible for the interference signs.

Sixteen fresh configurations cover electron and muon channels, neutrinos/antineutrinos, electronE=17.3/63.7MeV, muonE=threshold+.37/187.4MeV, new angles, and two declared coupling choices. Alternative pole masses823/1071/138.7MeV and(gv,ga,xi)=(.82,1.09,4.2) are algebraic diagnostics, not fitted physical uncertainty models.

Full explicit-spinor amplitudes agree with the author's full trace within6.67e-16 relative. IndividualVV,AA,FF,PP,VA,VF,AF,AP,VP,FP components agree within the recorded gates; the largest ordinary relative discrepancy is4.85e-12 in a suppressed term. ForbiddenVP/FP terms are tested on the total-amplitude scale. The independent construction agrees with the full trace without empirical normalization factors.

## Source map and signs

The metric+---,q=pb−pa=k−l, sigma=i[gamma,gamma]/2 and the WM coefficientiF2/(2M) agree with [Guo et al., Eqs26–27](https://arxiv.org/html/2006.12051v2). WithM=(mn+mp)/2, the declaredGP is dimensionless andGP/M=2MGA/(mpi²−q²). The review independently evaluates the dipole/pion formulas at each fresh point. Bare averageM, not effective/lepton mass, is used throughout. In vacuumq*=q; this identity is not qualified for a medium.

The equal-baryon-mass, identical-kinematics test exposes the crossing directly: onlyVA andAF change sign, whileAP andPP retain their sign andVP/FP vanish. Massless charged-lepton spinor tests independently remove the pseudoscalar contribution. These results support the author's signs and2M/M factors. No whole-current vector-conservation identity is imposed on unequal-mass charged currents.

[Strumia/Vissani, Eqs3–7](https://arxiv.org/html/astro-ph/0302055) usesf1=GV,g1=−GA,f2=F2,g2=−GP. Its reduced squared amplitude has the normalization mapphysical|M|²=8GF²Vud² times the published reduced polynomial. Its positiveDelta=mn−mp convention andB reversal for neutrinos match the inspected author map. The source has a different vector-pole numerical choice; inserting the same selected form factors tests algebra rather than equality of the papers' parameter sets. This review's independent execution is the explicit-spinor construction, not a second transcription of the A/B/C polynomial.

## Independent phase-space normalization

`normalization_probe.py` constructs the CM four-momenta directly froms andt, integrates the explicit-spinor result in **t**, and uses

    dσ/dt = physical|M|²/[64 pi s p_initial²].

Conversion tocm² is(hbarc)²×1e−26. For fresh physical default-current points, this agrees with the author's cosine64 integration within7e-15 relative. Internal adaptive error estimates reach2e-9 relative; no tighter integration-error bound is inferred merely from the forward agreement.

| IncidentE [MeV] | Channel | Independentσ [cm²] |
|---:|---|---:|
|37.6|nue+n|1.44870902586e-40|
|37.6|antinue+p|9.31548938919e-41|
|218.4|numu+n|2.84977434329e-39|
|218.4|antinumu+p|9.48071518512e-40|

Thus target spin averaging, incident helicity convention, amplitude normalization, two-body flux and azimuth factor are consistent across two numerical constructions. This does not compare with a radiatively corrected tabulation or measured cross section.

## Limits and provenance

The declared `q2_dependence=False` switch correctly retains the pion denominator while freezing the other selected momentum dependences; it must not be interpreted as constantGP. The pole guard conservatively limits the source-defined sub-pole interface. The existing source's zero-correction and sub-threshold conventions remain intact; no arbitrary threshold width or medium shift is introduced.

Omitted QED/electroweak radiative effects, second-class currents, nuclear correlations, thermal blocking, medium dispersion/current choices and form-factor uncertainties remain omitted. Full finite-T integration needs a new EOS-consistent derivation and forward benchmark, includingq versusq* and inverse/crossed reactions; the vacuum interference factors are not reusable dense-matter correction scalars.

`results.json` and `normalization_results.json` preserve amplitudes/components, cross sections and all224gates. The two bounded numerical probes cost approximately.054s and.130s. `MANIFEST.json` pins review and author files. No physics failure occurred in these probes, and no author file was edited or rerun in place.
