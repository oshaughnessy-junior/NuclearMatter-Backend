# Independent screened-Coulomb Lorentz transport review

**Accept the selected static external-ion screened Born model and its numerical transport implementation. 68 independent checks pass in0.095s. Realistic crust/core conductivity, ion dynamics/correlations and non-Born accuracy are not qualified.**

Frozen author source `screened_coulomb_transport/lorentz.py` has SHA256 `15f4c1b5604aeef2dd89aa2991c2fe48a08326ecd1878444969d95cae5f513ae`. All author sources/reports and frozen EOS dependencies have identical before/after hashes in `results.json`. No author validator was run, and no author file changed.

## Forward screening and units before transport

A new heldout bath uses T=.041MeV, ne=5.3e−8fm^-3, Z=2. Independent momentum-space density inversion gives the same total chemical potential mu=2.346911873506782MeV. Independent susceptibility integration gives chi=.5452473223780425MeV² and ks²=.049999853872627065MeV². Screening agrees within1.56e−15 relative.

The conversion is **n_natural=n_fm3(hbarc)^3**, and chi_natural=chi_fm3(hbarc)^3, not its inverse. The explicit ion density ni=n_natural/Z is independently checked. These are electron-only particle densities: positrons are excluded from transport/screening by the declared ansatz, with independently verified nplus/nminus=5.83404845e−34. Thus the actual pair-neutrality correction at this selected state is negligible, but exact pair-neutrality is not silently claimed for the general electron-only factory. A pair-rich state is correctly rejected.

The susceptibility screening convention is consistent with [Potekhin et al.1999, Eq.3](https://arxiv.org/html/astro-ph/9903127). Its realistic dielectric/structure-factor model is more elaborate; the present all-q Yukawa continuation of a local susceptibility is a distinct declared approximation.

## Independent absolute cross section and l=1 operator

The review constructs separate Weyl-basis gamma matrices and evaluates the spin-averaged external scalar-potential overlap

\[
\tfrac12\mathrm{tr}[(\slashed p'+m)\gamma^0(\slashed p+m)\gamma^0]
=4E^2[1-v^2\sin^2(\theta/2)].
\]

For U(q)=4πZα/(q²+ks²), the invariant normalization of external scattering gives

\[
\frac{d\sigma}{d\Omega}=\frac{E^2|U|^2}{4\pi^2}
\frac{\tfrac12\sum|u'^\dagger u|^2}{4E^2}
=\frac{4Z^2\alpha^2[E^2-q^2/4]}{(q^2+k_s^2)^2}.
\]

Three fresh momentum/angle combinations reproduce the author cross section within6.67e−16 relative. This tests the electron spin average and absolute potential factors; there is no target-spin2 factor for a static external charge. Spin2 is instead retained in the electron phase measure. A separate nonrelativistic unscreened test approaches the Rutherford normalization.

The collision frequency relevant to a charge/heat dipole is **ni v∫dΩ(1−cosθ)dσ/dΩ**, not the total scattering rate. Its normalization agrees with [Potekhin et al., Eqs.6–9](https://arxiv.org/html/astro-ph/9903127) after distinguishing the source Coulomb logarithm Lambda=I/2. For exact elastic scattering, f'(1−f)−f(1−f')=f'−f: no extra final-blocking factor belongs in that frequency. The review checks this algebra separately.

An independent full lab-axis angular gain-minus-loss integral, with an oblique incoming dipole, uses280 polar and640 azimuth points. It reproduces the exact −nu1 times dipole eigenvalue within2.86e−14. This is a check of the angular operator, not merely a substitution into the analytic Coulomb logarithm.

## Independent nested angular and momentum integration

For the reference collision frequency the integration variable is u=log(1+q²/ks²), with

\[
\nu_1=\frac{2\pi n_i vZ^2\alpha^2}{p^4}
\int_0^{\log(1+4p^2/k_s^2)}du\,(1-e^{-u})
\left[E^2-\frac{k_s^2}{4}(e^u-1)\right].
\]

The positive angular integral uses neither the author's dcs nor its analytic I/frequency functions. It matches absolute frequencies within3.0e−15 across p=.013,.19,.83,2.7,8.1MeV, spanning strong and weak screening. At each momentum, independent screening multipliers.2,1,8 reproduce the kernel and confirm that increased screening lowers the collision frequency. These multipliers are controlled kernel diagnostics, not new self-consistent EOS states.

This independent angular integral is nested inside a separate direct momentum quadrature for

\[
K_n=\frac1{3\pi^2}\int dp\,p^2v^2\nu_1^{-1}(E-\mu)^n
\left(-\frac{\partial f}{\partial E}\right).
\]

The reference uses its independently solved mu and susceptibility, finite Fermi-surface breakpoints, and an85T outer tail. All three moments agree with the author energy-window calculation within4.0e−15:

| Quantity | Independent result |
|---|---:|
| K0 [MeV] |793.5806233805656|
| K1 [MeV²] |7.1364671538889555|
| K2 [MeV³] |4.461609620727111|
| sigma [MeV,HL] |72.77232473535292|
| open-circuit kappa [MeV²] |107.25446830892355|
| Seebeck |−.724303798024443|

This checks the full screening/cross-section/phase/moment chain. No collision-time fit is inserted.

## Thermoelectric sign and degenerate limits

With q_e=−sqrt(4πα) and Estar=E−grad(mu)/q_e, independent substitution in

\[
j=q_e^2K_0E_* - q_eK_1\nabla T/T,\qquad
J_Q=q_eK_1E_* - K_2\nabla T/T
\]

verifies that Estar=S gradT gives j=0 for S=K1/(q_e T K0), and then JQ=−kappa_open gradT. The open-circuit subtraction K1²/K0 and the Peltier sign T*S both pass. Natural **Heaviside–Lorentz e²=4πα** is explicit: using Gaussian e²=α in sigma while retaining this screening convention would introduce a spurious4π. No CGS conversion is inferred.

The independent Onsager matrix T[[K0,K1],[K1,K2]] is symmetric and positive definite, and both chosen force vectors have positive entropy production. These are local linear-response statements for the specified Lorentz operator.

A new low-T sequence at ne=2.3e−8fm^-3,Z=1 re-solves the full density/screening state at T/TF=.017,.0085,.00425. Relative Wiedemann–Franz errors are9.23764e−4,2.31074e−4,5.77769e−5, with the expected T² decrease. Independent cold-screening/q-integrated Drude errors decrease5.94055e−4→1.48589e−4→3.71519e−5. These limits and energy-average conventions agree with [Potekhin1999](https://arxiv.org/html/astro-ph/9909100); finite-T WF equality is not imposed or assumed.

## Born scope and preserved reference failure

At the main heldout state, Zα/vF=.0149527. The formal low-velocity region where Zα/v>1 contributes about2.48e−31 of K0 in the independent model integral. This supports identification of the dominant kinematic region but **is not a quantitative bound on non-Born corrections**. Static uncorrelated external charges, local screening, point nuclei, no recoil, and no electron-electron collisions remain physical assumptions. The Lorentz collision operator relaxes angular modes; it does not itself thermalize arbitrary isotropic distributions to FD.

One initial **review-reference** check failed: an unscaled positron-density quadrature used an absolute tolerance vastly larger than its tiny result, producing a log-ratio discrepancy−2.9236e−8. Its exact source and report are preserved as `initial_unscaled_antiparticle*`. A positive scaled quadrature repairs only that independent reference; no author source, acceptance tolerance or physical model changed. All68 final checks pass, including invalid controls and the pair-rich rejection.

The accepted scope is a reproducible static external-ion Born transport reference. It does not replace NMB crust/core conductivity, validate a realistic stellar medium, or supply correlated-ion/phonon/impurity, dynamic/finite-q dielectric, electron-electron, magnetic, pair-rich or baryon transport. No physical uncertainty is inferred from agreement of numerical methods.

Reproduce from repository root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_screened_coulomb/probe.py
```
