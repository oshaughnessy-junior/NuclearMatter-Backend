# Independent affinity-targeted GM1 potential review

**Accept the frozen single-potential approximation at the newly sampled states. 384 independent main checks and126 figure-data checks pass.** The unchanged balance-factor gate is1%; Tmin remains.12MeV. No uniform domain certificate, T→0 rate-error bound, external GM1 validation or actual weak-rate accuracy is established.

Frozen sources are `potential.py` SHA256 `5996a98f69d76f56ab03e5c59361fc7e8b4a75c71d0a877936850c1785806be2` and `coefficients.json` SHA256 `1e03c7e192ab869018c2da575d613026f8be81d75ca7709f8b5de1d8ce59ea8b`. Every author source/report, the frozen exact EOS/stability source, and the old potential/coefficients/original review report have identical before/after hashes. No author validator, fitting script or coefficient update was run. Final physical experiment11.03s; no new failure or interrupted experiment occurred.

## One thermodynamic potential, correct variables

The representation is one nuclear Helmholtz free energy **per baryon**,

\[
a(n,T,Y)=939+C_0(n,\delta^2)+T^2C_2(n,\delta^2)
+T^4C_4(n,\delta^2)+T^6C_6(n,\delta^2),\quad\delta=1-2Y.
\]

Each of four fields has7×7 Chebyshev coefficients:196 scalar parameters total. The declared domain is n=.18–.26fm^-3,Y=.18–.33,T=.12–3MeV. Common939MeV bare nucleons match the exact internal reference. No physical neutron/proton mass difference, leptons or extra weak Q is added.

With f=na, the consistent derivatives are

\[
P=n^2a_n,\quad s=-na_T,\quad e=n(a-Ta_T),\quad C_{V,nY}=-nTa_{TT},
\]
\[
\mu_n=a+na_n-Ya_Y,\quad
\mu_p=a+na_n+(1-Y)a_Y,\quad\mu_n-\mu_p=-a_Y.
\]

Pressure and densities are MeVfm^-3, a and chemical potentials MeV, entropy/Cv fm^-3, and the chemical Hessian dmu_i/dn_j MeVfm³. C2,C4,C6 have MeV^-1,MeV^-3,MeV^-5 units. The isospin-coordinate derivatives zY=−8delta/span and zYY=16/span are correct; the full density Hessian must also differentiate n and Y=np/(nn+np), not only fixed-Y compressibility.

The review independently constructs a **value-only Chebyshev recurrence**, then obtains chemical potentials by complex steps directly in independent(nn,np). Pressure and entropy use separate complex steps in(n,T) at fixedY. These agree with the author's analytic output within4.45e−16 relative. Density differences of those independently obtained chemical potentials also reproduce the full analytic fitted Hessian within9.03e−9 relative. A delta² reflection test is purely a mathematical basis symmetry check; its outside-domain partner is not counted as a physical holdout.

## Forward physical reference gates before derivatives

Four new coordinates were recorded in `REVIEW_PROTOCOL.json` before any exact solve, and checked against the65 training/development coordinates and all30 author holdouts:

| nB [fm^-3] | T [MeV] | Yp | Fractional balance error |
|---:|---:|---:|---:|
|.2549|.127|.3251|5.6438651e−6|
|.1847|.143|.1879|4.9352349e−6|
|.2327|2.47|.3113|1.3649132e−7|
|.2589|.121|.3287|1.1257173e−5|

Density/composition points above the training maxima.25/.32 are genuinely new joint extrapolation samples within the declared evaluation domain. The old failure(.1811,.12,.1817) remains **development**, not a new holdout.

Each exact finite-T GM1 solve first passes its density/field residual gate. Separate momentum-space quadrature at32/48nodes per physical panel reconstructs density, scalar field balance, pressure, energy and entropy from the actual solved kinetic chemical potentials and Dirac mass. These forward values agree within1.37e−14 relative, including the scalar/vector interaction decomposition rather than a quasiparticle energy sum that double counts fields.

The exact full two-charge Hessian is then independently reconstructed by **re-solving GM1** at neutron- and proton-density perturbations, at fixed T and other species density. No author's implicit-Hessian function is called. Decreasing steps yield relative Hessian changes at most2.71e−8, and reciprocity passes. Cv is independently differentiated from actual **energy**, not the author's entropy-difference reference; its decreasing-step uncertainty is at most4.79e−7 relative. All perturbed forward roots are separately gated.

## Fresh physical errors under unchanged gates

Across these four states:

| Quantity | Maximum measured error | Existing gate |
|---|---:|---:|
| abs(expm1(delta(mu_n−mu_p)/T)) |1.1257173e−5|.01|
| abs(delta(mu_n−mu_p)) |1.36213e−6MeV|balance gate above|
| Free energy per baryon |2.67527e−7MeV|.005MeV|
| Energy per baryon |2.65980e−7MeV|.005MeV|
| Pressure relative |5.56344e−7|.01|
| Entropy relative |1.71363e−6|.01|
| Cv relative |1.89213e−6|.01|
| Full Hessian Frobenius relative |7.67415e−6|.02|

The smallest exact/fitted chemical Hessian eigenvalues and heat capacities are positive at every fresh state. This is sampled local two-charge stability, not a global coexistence construction or proof of all intermediate-state stability. The exact and approximate potentials agree at these states; neither is thereby externally validated.

## Training, Sommerfeld and freeze audit

The review independently reconstructs the65×49 value design and the stacked cold free/chemical/pressure derivative design using its complex-safe value basis. Both have rank49. The measured condition numbers288.198453579864/193.2195464794784 reproduce the declared values. The normalized C0 least-squares normal-equation residual is4.38e−17. Maximum cold training residuals are3.60e−8MeV in a,3.29e−7MeV in aY, and3.49e−6MeVfm³ in an. The derivative weights are modeling choices, not statistical uncertainties or a likelihood.

The independently derived leading Sommerfeld coefficient is

\[
C_2=-\frac{1}{6n(\hbar c)^3}
\sum_{i=n,p}p_{Fi}\sqrt{p_{Fi}^2+m_D^2},
\]

using cold stationary fields at fixed composition. The review verifies its units and all65 training targets. Independent fresh cold GM1 states reproduce the fitted C2 within1.72e−6 relative at the four review coordinates. This is a low-temperature structural anchor, not a uniform weak-rate certificate as T→0.

The two higher thermal targets exactly reconstruct the stored T=2 and3 physical warm free energies, and independent thermal normal-equation checks verify the least-squares C2/C4/C6 solutions without refitting. Higher powers provide restricted warm anchoring; they are not claimed asymptotic over arbitrary temperature.

The current coefficient/source/protocol hashes match `PRE_HOLDOUT_FREEZE.json`; the author heldout source checks those hashes before evaluating and again at completion. The development table includes the old failed point explicitly. No new coefficients or basis changes occur in the review. Author holdouts, fresh review states and development data remain separate in raw provenance.

## Figure comparison audit

`figure_audit_results.json` independently checks all root figure source hashes and the same30 exact states for both frozen fits, then reevaluates the old potential. The old maximum is **3.1325100718023737%** and the new maximum **.0013236905759471884%**, using100*abs(expm1(delta_mu/T)). All126 checks pass. The old failed development point is absent.

The correct caption is **new-fit blocked samples evaluated with both frozen potentials**; these need not all be heldout states for the old fit. The plot supplies sampled chemical balance-factor errors at the same temperature/lepton contribution, not a uniform rectangle bound, an actual weak emissivity, or an external physical validation. Root figure files were not edited.

## Qualification

All510 new checks pass with frozen coefficients and preserved old failures. The unchanged1% balance gate is comfortably met at the new samples. This result supports use as a compact **internal thermodynamic-potential approximation** alongside the detailed reference, not a substitute for the physical N2N or weak chain. A potential fit does not supply quasiparticle masses/current vertices or momentum-dependent weak thresholds.

Maintain Tmin=.12MeV and the declared domain. Finite nonzero chemical fit error generally makes exp(delta_mu/T) unbounded as T→0. More samples do not turn the observed errors into a uniform certificate. External matched GM1, broader phase qualification and actual weak-rate comparisons remain open.

Reproduce from the repository root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_gm1_affinity_fit/probe.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_gm1_affinity_fit/figure_audit.py
```
