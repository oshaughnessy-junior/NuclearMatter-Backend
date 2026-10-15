# Independent relativistic delta-shell oracle

15 October 2026. This leaf owns a separately derived numerical oracle for three prescribed free-baryon finite-temperature charged-current points. It does not import the new tensor author's integration, contraction, vertices or support code. It independently reconstructs Dirac matrices, external spinors, source-defined form factors, forward/inverse occupations and absolute invariant phase measure. Shared inputs alone were coordinated. The source conventions follow [Guo et al., Eqs. 26–30](https://arxiv.org/html/2006.12051v2) and the campaign's independently reviewed relativistic response contract; this is a numerical free-gas reference, not a new interacting-current derivation.

## Coordinate and normalization derivation

Use metric +---, total chemical potentials including rest mass, free dispersions Ea=sqrt(ma²+p²), Eb=sqrt(mb²+|p+q|²), omega=Enu−El. Orient q along z but integrate **transverse momentum**, not the author's initial-energy/angular delta reduction. Write r=|p_perp| and A=q²−omega²>0. The original delta condition Eb−Ea=omega gives

`omega Ea − q pz = D/2`, `D=mb²−ma²+A`.

A longitudinal Lorentz transformation therefore fixes `z' = −D/(2 sqrt(A))`, `E'=sqrt(ma²+r²+z'²)`, and uniquely

`pz=(q z'+omega E')/sqrt(A)`, `Ea=(q E'+omega z')/sqrt(A)`.

The delta derivative is `J=|(pz+q)/Eb−pz/Ea|`. Its numerical value is tested against a finite difference of the original energy delta; a separate generic bracketed original-delta root confirms pz at nine transverse points. No use is made of the q-axis energy-angle support polynomial. Strictly spacelike transfer is this oracle's explicit domain: timelike/linear boundaries are rejected and are outside its evidence.

After integrating the baryon three-momentum delta, the invariant measure is

`d³pa (2pi)delta(Ea+omega−Eb)/[(2pi)³ 4EaEb]`.

Thus `W=∫r dr dphi H fa(1−fb)/(16pi² EaEb J)`. Numerical integration uses `u=r²/(2ma)`, so `r dr=ma du`, with Fermi-surface panels and positive-energy upper tails. H is the full baryon spin sum, constructed from four explicit ubar_b Gamma u_a amplitudes; no target spin average or extra factor2 is inserted. W has units MeV² and no weak constant. Lepton currents are separately spin summed with their inactive incident helicity annihilated by V−A. The inverse W uses the same microscopic matrix element with independently evaluated fb(1−fa), without substituting the KMS ratio.

Integrating only the charged-lepton azimuth yields

`dchi_s/(dEl dcos)=GF² Vud²/(32pi² hbar) * pl/Enu * (1−fl) Σμν metricμ metricν Lμν Wμν`.

The corresponding j uses fl and the inverse W. Both outputs have units s^-1 MeV^-1 per unit cosine. GF=1.1663785e−11 MeV^-2, Vud=.974, hbar=6.582119569e−22 MeV s. No c/hbarc density conversion enters W. These absolute factors are derived from the invariant measure, not fitted to another implementation.

The full Guo V/A/F2/GP current is reconstructed directly, with Mweak=938.91875434 MeV for the three physical-mass points, pion pole139.57 MeV and vector/axial poles840/1000 MeV. Antineutrinos use v-spinor lepton currents and the conjugate baryon vertex barGamma(−q). Charged-lepton El is total energy. Lepton mu inputs are already signed for the charged particle: −30 MeV for the antimuon point.

## Selected evidence

`python run.py`, `python check_controls.py`, `python check_nr.py` produce **39+28+4=71 passing current checks**, each in less than one second. The three physical points are fully recorded in results.json. The finest transverse quadrature is order32 per panel,16 azimuths, tail55T; comparisons12/4/30 →24/8/40 →32/16/55 pass unchanged3e−7 gates. These are selected refinement checks, not a rigorous tail bound or a rectangular error certificate. Full complex forward/inverse W and L matrices are retained rather than just contractions.

| prescribed bath / lepton | chi | j |
|---|---:|---:|
| n→p, T4, mu_n945/mu_p938, Enu40/El38/c.2, electron mu20 | 1362119.543016284 | 1594.878013999143 |
| n→p, T12, mu_n950/mu_p940, Enu150/El120/c−.3, muon mu80 | 4514098.74591983 | 5744.7946466432695 |
| p→n, T12, mu_p940/mu_n950, Enu160/El125/c.4, antimuon mu−30 | 936515.7523492385 | .28648234183014876 |

Units are s^-1 MeV^-1 per unit cosine. KMS follows from the two independently accumulated occupation products, `j/chi=exp[−(Enu−mul−mub+mua)/T]`, at2e−12 gate; it does not substitute for convergence. Shell, W Hermiticity and positive semidefiniteness checks pass. These baths prescribe chemical potentials, not EOS-neutral densities or weak equilibrium. Baryon antiparticle cuts are omitted; their Boltzmann suppression exp[−(m+mu)/T] is below~2e−68 for the listed baths.

An independent **point-V/A** diagnostic at common masses1e5,1e6,1e7 MeV compares the entire differential coefficient to the accepted spin-two NR FD density response and absolute allowed prefactor. Relative errors fall1.18573e−4 →1.18700e−5 →1.20006e−6. This is a mathematical heavy-mass limit, not physical neutron-star matter. It tests the 2pi/spin/flux convention without calibrating a scale. The source-defined full current is intentionally not identical to a contact V/A allowed current even at large M: fixed-q form factors and the pseudoscalar term retain corrections. No full-current allowed-limit equality is claimed.

## Preserved failed diagnostics and limitations

Initial heavy-mass comparison driver omitted two axial spin components in the *reference* angular factor (used1+ga² instead of1+3ga²), producing an order-unity validation error. Its driver/results are preserved. Correcting that reference still gave a~7e−4 mismatch because a full form-factor/GP current had been compared to contact allowed physics. That raw failure is also retained; the declared point_VA diagnostic supplies the genuinely matched model instead. The three physical full-current point values did not change. The first timelike rejection test accidentally supplied a spacelike point; its failed control result is preserved and the point was corrected to an actual timelike transfer. Two JSON numpy-bool reporting failures have preserved initial drivers and a reporting record. These are review/probe corrections, not evidence for extending physical domains or relaxing gates.

The selected ordinary-positive interface raises if integrated coefficients are zero/nonfinite, rather than certifying an underflow zero. It does not return reliable extreme log tails. Fermi factors are individually positive stable expit calls; no1−expit cancellation is used. Tail controls12..120T are exposed and bounded, with no artificial physical width/support cut. Generic transverse-tail accuracy, extreme degeneracy, timelike particle-hole cuts, angular/energy integrated rates, density-matched EOS, crossed decay inventory, interacting GM1, medium prescriptions and radiative corrections remain outside this leaf. Independent tensor comparison is pending until the separate author freezes.
