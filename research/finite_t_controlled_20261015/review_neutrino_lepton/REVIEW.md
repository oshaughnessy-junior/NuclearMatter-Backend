# Independent review: free neutrino–pair-lepton scattering

**ACCEPTED for the stated free Fermi-model selected points and moment controls, with the six original coarse transport failures retained.** Frozen author `scatter.py` SHA256 `8628fb1fe917a6d84caf6aa789a50feab359790db1c8216eb936c8e0dcfa4064`. This review adds **124/124 passing checks**, including a new antineutrino/positron thermal case; it does not certify the full guarded input rectangle or a coupled transport collision operator.

## Current and absolute normalization audit

The physical current is `GF/sqrt2 [nu current gammaμ(1−gamma5)][electron current gammaμ(CV−CA gamma5)]`, with **no Vud**. Electron flavor includes charged-current interference through `CV=1/2+2sin²thetaW,CA=1/2`; mu/tau use `CV=−1/2+2sin²thetaW,CA=−1/2`. Fixed illustrative sin²thetaW=.23129 is a model input, not a running-angle precision prescription.

The independent review defines its own gamma matrices and performs full Dirac traces rather than calling the author's spinors/gamma implementation. Target traces use `(slash p+m)` for electrons and `(slash p−m)` for positrons; reversing the positron bilinear and the antineutrino bilinear gives the literal crossed order. Forty-eight fresh amplitude comparisons cover e/mu couplings, all four charge-conjugation combinations and massive CM momenta0.2,3.7,14MeV at two nontrivial angles. All pass at2e-12 relative, and imaginary trace residuals pass separately. The target spin average is2; the neutrino is not averaged over two helicities. Consequently `GF²/4` multiplying the unaveraged two-current trace and a spin2 target occupation measure are consistent.

The primary [Guo et al. 2006.12051v2](https://arxiv.org/html/2006.12051v2), Eqs1–4 and AppendixA40–41, explicitly supplies the target-spin2 kernel and the three massive-lepton invariants. Its nu_mu/e− coefficients reduce to `16GF²(CV+CA)²`, `16GF²(CV−CA)²`, and `−16GF²me²(CV²−CA²)` in the author's convention; the antineutrino exchanges the first two. Thus source coefficients, literal fresh traces and the local scalar formula have consistent normalization and crossing signs. This is free weak-model agreement, not an EW/QED uncertainty bound or an empirical scattering measurement comparison.

## Phase measure, FD occupations and reciprocal DOS

The initial target occupation is per spin. Final electron or positron blocking uses that species' same chemical potential, with `mu_positron=−mu_e`; its sign is not the initial neutrino chemical potential. The CM integral has the target-spin2 measure, the relative incident-angle measure and the Moller velocity once. It is a rate **per incident neutrino**, so no additional incoming-neutrino DOS belongs there.

Independently reducing the invariant scattering rate, integrating the final charged-lepton momentum delta and target polar delta yields

```
D(E,F,c) = F/(64π³ E Q hbar) ∫dEt <Mavg²>phi f(Et)[1−f(Et+omega)],
omega=E−F, Q²=omega²+2EF(1−c).
```

The factor includes target spin2 and final-neutrino azimuth2π. It has units s^-1 MeV^-1 per cosine. Adding another final-neutrino DOS to this definition would double-count its outgoing phase measure.

The review implements a **different inner reduction**: target transverse momentum r with

```
a=Q²−omega²,
Et(r)=[−omega+Q sqrt(1+4(me²+r²)/a)]/2,
dEt/dr=2Qr/[a sqrt(1+4(me²+r²)/a)].
```

The azimuth is integrated analytically rather than sampled: with `h=EF(1−c)`, `u=E Et−kz pz`, `v²=kx²r²`, the two relevant averages are `<(k·p)²>=u²+v²/2` and `<(k·p−h)²>=(u−h)²+v²/2`. Adaptive r integration supplies an independent differential reference without using the author's differential function or finite phi quadrature.

Three new redistribution points at E6.7MeV, F3.9/10.2/6.85MeV and c−.38/.46/.97 agree with the author's exact-shell kernel within3e-8. Direct reversed calculations pass

```
D(F,E,c)/D(E,F,c) = (E/F)² exp[(F−E)/T].
```

Combining the one-active-helicity DOS `rho(E)=E²/(2π²(hbar c)³)` with BOTH FD neutrino occupation/hole factors gives local same-T stationarity for two new chemical potentials−.7 and2.1MeV. This is a kernel detailed-balance check, not an implementation or validation of a finite-grid gain/loss evolution. The optional final-neutrino blocking in the author loss integral is physically correct, but alone is not a complete radiation collision operator.

## New thermal forward and moment comparison

The heldout bath has **T1.4MeV, mu_e−.11MeV, incoming anti_nu_mu E6.7MeV and e+ targets**, with final antineutrinos unblocked. Its target is the actual FD positron population with mu_plus+.11MeV. This is a standalone pair bath; no new matched baryon composition or neutrality constraint is asserted for it. An independent adaptive momentum density integral agrees with the CM population within2e-9.

The review integrates its own transverse-shell differential over outgoing energy and `z=sin(theta/2)`, including the Jacobian `dc=4z dz`. It refines outer energy/angular order12/16→18/24, outgoing energy cutoff100→150MeV, and independently changes target tail70T→90T. The resulting moments agree with a newly computed author CM event integral using28/36/36/44 controls and75T tail:

| Moment | Independent shell | Author CM |
|---|---:|---:|
| Total rate (s^-1) |0.239783843983|0.239783844181|
| Transport rate (s^-1) |0.163890114007|0.163890114260|
| Signed neutrino energy change (MeV/s) |−0.111404398595|−0.111404398528|
| Outgoing energy moment (MeV/s) |1.49514735609|1.49514735748|

The maximum cross-method discrepancy is approximately1.55e-9 relative; selected outer refinement is about2.9e-9 or smaller. All four parent-tail changes pass3e-8. Individual experiments take under3seconds. Forward values are checked before any derivative claim, and no derivatives are qualified.

The transport weight is the actual bath-frame `1−cos_lab`, nonnegative and distinct from a diffusion coefficient in an inelastic medium. Negative neutrino energy change means positive matter heating. Scattering preserves the target species, neutrino species and their lepton numbers; no beta affinity or material composition source is produced. The direct four-vector residual and integrated energy bookkeeping are consistent. The latter is partly an algebraic moment identity, not a substitute for the independently refined phase integral.

## Scope and retained failures

The review verifies all frozen source/raw hashes unchanged, and explicitly confirms **six coarse author transport failures** remain in the original results. It does not reclassify them based on later passing refinements. The original outgoing-energy guard failure and unqualified discontinuous histogram also remain; histogram bins are not accepted as a converged transport grid.

Accepted evidence: the free massive Fermi current, literal crossing and normalization at selected vacuum points, actual FD target blocking, selected redistribution/detailed balance, author-refined pilot moments and this additional heldout anti_mu/e+ case. Finite-control guards are admission restrictions rather than domain-wide accuracy certificates. Exact collinear limits, arbitrary very cold/degenerate baths or ordinary underflow, all-flavor/all-target radiation closures, finite spectral grids and continuum evolution require separate gates. Plasma/medium correlations, running EW/QED corrections, muon targets, flavor oscillations and material/radiation evolution are absent. Numerical agreement between these reductions is not a measure of missing physical-model uncertainty.

Reproduce with `python probe.py` using the existing campaign NumPy/SciPy interpreter. It imports only the frozen function module by a unique explicit spec, never an author driver. `results.json` records literal/differential/moment checks, controls, forward values, target density and before/after frozen-artifact hashes. CHECKPOINT.json pins review source/raw/report.
