# Stationary allowed charged-current rates: four-state accuracy assessment

**Reuse assessment, not a new weak-rate kernel.** The existing independently checked `weak_rates/static_capture.py` already supplies the requested simple one-dimensional physical integrals. This leaf executes that source unchanged, supplies actual matched FD electron chemical potentials and target densities, and compares its number/energy moments with immutable detailed full-current/FD event results at four selected transparent kinetic states. No rate is fitted, rescaled to the detailed answer or calibrated to a measured lifetime. There are158 passing numerical/normalization checks and20 retained model-accuracy failures against illustrative criteria declared before execution.

## The approximation and its absolute normalization

The weak interaction is the same GF/sqrt(2) Vud one-body V−A interaction as the detailed reference, with zero-transfer gV=1 and gA=1.2723. After averaging the initial unpolarized baryon spin, the leading static squared matrix element is16 GF²|Vud|² MnMp EeEν [(1+3gA²)+(1−gA²)ve cosθ]. The angular term integrates to zero. The standard three-body phase measure, with one neutrino helicity and no identical-particle factor, gives

    A=GF²|Vud|²(1+3gA²)/(2π³ hbar).

Equivalently the absorption cross-section is GF²|Vud|²(1+3gA²)Eepe/π in natural units, with one incident neutrino helicity. Crossing to electron/positron capture averages two initial charged-lepton spins; the charged-lepton phase density contains those same two spins. Thus per-target capture has the same A and no extra factor of two. Multiplying by total target density does not add another nucleon spin factor. In spectral language one active neutrino DOS is Eν²/[2π²(hbar c)³]; its product with j reproduces the same target moments.

The three integrands, with k=0 number and k=1 neutrino energy, are

    EC:   A ∫max(me,Q)^∞ dEe Ee pe (Ee−Q)^(2+k) f−(Ee)
    PC:   A ∫me^∞      dEe Ee pe (Ee+Q)^(2+k) f+(Ee)
    beta: A ∫me^Q      dEe Ee pe (Q−Ee)^(2+k) [1−f−(Ee)].

Here Q=Mn−Mp from the same physical mass constants, µplus=−µe, and µe includes rest mass. EC target density is np, PC/beta target density nn. The scalar normalization source is [Chiesa et al. 2412.04570v2, Appendix A43–44](https://arxiv.org/html/2412.04570v2#A1), taking its target blocking factors to initial densities and corrections W/Wbar to unity. Its displayed gA=1.23 is not adopted: our frozen reference and detailed current both use1.2723, checked by the full current matrix at zero transfer. Its optional mean-field/recoil/weak-magnetism recipe is not part of this approximation. The existing static technical note documents constants and units; no source-code parity with that paper is claimed.

The detailed model keeps moving relativistic nucleons, recoil, separate actual baryon occupations/vacancies, full reciprocal charged current with weak magnetism, induced pseudoscalar and q² form factors. The allowed model drops all of those corrections together, while retaining the actual relativistic FD lepton occupations, finite electron mass, exact reaction Q, beta electron blocking and physical target densities. The measured discrepancies cannot be uniquely assigned to recoil, weak magnetism or blocking without additional controlled switches. In particular the allowed model's stationary spectral endpoints are not finite-temperature support cuts for the detailed model.

## Forward numerical anchors before errors

The allowed vacuum beta integral has the closed form A me^5 f(w), w=Q/me, with

    f(w)=[sqrt(w²−1)(2w⁴−9w²−8)+15w acosh(w)]/60.

The absolute width is0.0010555526228422007 s^-1, lifetime947.3710531904902s. This is an uncorrected allowed-model diagnostic. The immutable detailed tree-current rest width is0.0010533846209151876 s^-1 (949.32086547s), so even the vacuum models differ by0.2058%; neither is a measured-lifetime fit. The parent proper-density/coordinate-density factor at the original hot root is0.99681549: an unblocked moving gas decays at Γrest∫dn Mn/En. Warm electron-blocked beta rows are not substituted into that unblocked identity. This supplementary anchor reuses the accepted actual rest-phase integral and proper density, rather than rerunning a detailed vacuum calculation.

The analytic massless Q=0 FD moments T^5 F4(0) and T^6 F5(0) also pass before any model comparison. The unchanged existing thermal routine uses Ee=me+x² and Gaussian panels. An independent p/T QUADPACK reduction uses Ee pe dEe=p²dp and infinite capture bounds. At each state, its absolute number and energy moments agree within3e-10. Isolated order96→192 at fixed tail70 and isolated tail70→90 at fixed order96 separately pass, rather than inferring tail convergence from a simultaneous refinement.

## Measured differences at the same exact detailed states

Percentages below are100*(allowed/detailed−1), with number/energy kept separate.

| T MeV, nB fm^-3 | EC number / energy | PC number / energy | beta number / energy |
|---|---|---|---|
|2,1e-7 original|−2.259% / −2.441%|+6.872% / +8.423%|+0.473% / +0.212%|
|1,1e-7|−1.235% / −1.364%|+3.480% / +4.200%|+0.307% / +0.206%|
|3,1e-7|−3.198% / −3.397%|+10.388% / +12.823%|+0.631% / +0.211%|
|2,1e-6|−2.244% / −2.419%|+6.828% / +8.390%|+0.458% / +0.199%|

The comparison uses saved immutable detailed moments at exactly those EOS states, not a new independent detailed integration. All reference denominators are explicitly positive; no relative error is formed around a net source zero. The full separate rates and both energy moments remain in results.json. The20 illustrative assessment failures are16 EC/PC moment failures at1%, plus4 root failures at|deltaYe|≤.001. All8 beta moment tests pass1%. Numerical integration gates remain separate and all pass. These criteria are practical assessment choices, not a universal physical accuracy requirement.

## Transparent composition balance

Using only fresh EOS/lepton roots and the simple physical integrals at every trial, the allowed model independently solves Rbeta+RPC−REC=0. Detailed saved rates never enter this approximate root function. In both cases the bath temperature/density are imposed, incoming radiation is exactly empty, EM pair equilibration is assumed and a thermostat supplies escaped energy. Photons/muons are not evolved. Zero composition drift is not energy equilibrium or trapped beta equilibrium.

| T MeV, nB fm^-3 | detailed kinetic Ye | allowed kinetic Ye | deltaYe | detailed / allowed-root affinity (MeV) |
|---|---|---|---|---|
|2,1e-7|.556044822|.575319840|+.019275018|.516198 / .348141|
|1,1e-7|.385056710|.390874991|+.005818281|.902423 / .866622|
|3,1e-7|.549006580|.580009790|+.031003211|.555982 / .169542|
|2,1e-6|.278627758|.287768400|+.009140642|1.662006 /1.527039|

Affinities are evaluated in the same actual FD free EOS at each model's own root, not through the static model's classical trapped-equilibrium formula. The static approximate roots have a numerical residual≤1e-9 relative to gross emission and independent moment refinement. At detailed roots the allowed dYe/dt is+.6022,+.00910,+6.8067,+.5389s^-1 in table order; quoting a relative error in the zero detailed drift would be meaningless. Individual modest rate errors can therefore matter strongly to the balanced composition.

## Provenance, scope and next use

Commands: `python allowed_rate_assessment/assess.py` and `python allowed_rate_assessment/supplement.py` relative to the campaign root. Runtime3.28s plus.60s. Source/raw SHA pins are in results.json and MANIFEST.json; all remain unchanged after the assessment. The new grid was initially labelled conditional in immutable results.json. Its later different independent review explicitly accepted the three selected baths; supplement_results.json pins that acceptance at review_transparent_balance_grid/REVIEW.md SHA0b2fe53e9a343a8a75f3cfbd38ea5afbb6bdc34895553202975c616deb9a988e. This resolves conditional data admission without modifying the frozen grid or the original assessment record.

Reuse the stationary model for inexpensive diagnostics with these explicit channel-dependent errors; retain the detailed chain for the selected balance/energy assessment. This leaf gives four selected-state differences, not a broad fit, physical uncertainty estimate, arbitrary-density admission, dense-matter Urca model, spectral opacity, complete interaction inventory, evolution or AD qualification. A different reviewer must independently audit this assessment before promotion. No upstream files, dependencies, production or paper files were changed.
