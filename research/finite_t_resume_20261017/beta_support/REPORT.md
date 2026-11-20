# Conditional parent/cluster beta spectral quadrature

Author slice, 17 October 2026. Exact prior PR55 full free Guo current, physical neutron/proton/electron masses, spin conventions, fixed T2 chemical bath and separate actual FD products are retained. This directory is research only. Previous outputs, PC candidate, PRs and production modules remain unchanged. DIFFERENT integrated-phase review remains pending.

## Exact prior witness reproduced

`OLD_TAIL70.json` and `OLD_TAIL90.json` make two fresh calls to the exact copied original source at Eν=1.2 MeV, using unchanged fine physical orders and an 85-second resource budget. They reproduce

```
old tail70 log j = −53.594720186261625
old tail90 log j = −53.594496435116596
relative change = +2.2377617918348388e−4.
```

This fails the original 3e−6 local criterion. The original last parent-momentum panel changes from its fixed interior cut at 429.13263966 MeV to a moving upper endpoint. At tail90 it is [429.13263966,757.72202327] MeV. The same 12-point rule is stretched over this changing interval, while the parent Boltzmann factor concentrates near pmin=414.74233561 MeV. This is coupled tail/interior sampling sensitivity, not an independently measured missing-tail contribution. Neither rate smallness nor agreement of integrated moments waives this failure.

`PRIOR_BAND_REPORT.md` and `PRIOR_CONTROL_ASSESSMENT.json` preserve the prior local/band context. The earlier upper incident-panel 12→20 discrepancy is **not rerun or claimed resolved** by this eight-call slice. Resolving selected local coefficients is a prerequisite, not a replacement for that separate band convergence gate.

## Exact support and stable coordinates

Let E be the fixed laboratory antineutrino energy, M=mn and

```
E0 = [mn²−(mp+me)²]/(2 mn).
```

For the selected upper branch E>E0, a parent neutron must have rapidity eta≥eta0=log(E/E0). Its exact minimum energy and momentum are A0=M cosh(eta0), p0=M sinh(eta0). This is thermal motion support, not a new laboratory endpoint or physical width. Write

```
z=(A−A0)/T >=0,   deltaA=T z,   p=sqrt(A²−M²),
R=deltaA [1+(A+A0)/(p+p0)] = A+p−(A0+p0),
width=R/p.
```

The rest-frame antineutrino direction relative to the parent boost has the conditional cone

```
c*=1−width u,  0<u<1,
D=gamma(1+velocity*c*)=(A+p−R u)/M,
eν*=E/D.
```

The electron–proton pair's invariant excess above threshold is evaluated without subtracting two nearly equal endpoint energies:

```
delta_s=2 M E0 R(1−u)/(A+p−R u),
s_pair=(mp+me)²+delta_s,
p*=sqrt[delta_s(delta_s+4 mp me)]/(2 sqrt(s_pair)).
```

The exact u=1−v² map removes p*'s square-root corner; cone measure dc*/2 becomes width*v*dv. The map changes no masses, occupations, current or physical support. The endpoint has zero phase measure mathematically; positive interior nodes are never skipped or turned into zero coefficients.

The electron and proton are constructed on their exact pair CM shells and boosted into the neutron rest frame. The same original full literal Guo matrix element is evaluated there. Daughter rest vectors sum with the antineutrino to the future neutron vector. A parent boost supplies laboratory energies for every actual FD factor. The unchanged 1e−10 proton-shell guard is retained, with explicit future energies, an independently evaluated unsquared rest-shell energy equation, and a laboratory energy-conservation guard. At the selected rules the largest shell residual is 4.44e−16 and unsquared residual 1.21e−16.

## Absolute phase, spins and DOS

No normalization is fitted. The copied `decay_amplitude` is the literal neutron-spin-averaged-two full tree amplitude, with the named Guo form factors. Its inverse is evaluated with reciprocal external spinors/vertices on an actual event batch; agreement is checked independently of any FD or KMS identity.

An unreduced invariant normalization follows from

```
dΓ = |M|²/(2 mn hbar) × [d³kν/(2 Eν (2π)³)] × dΦ2(pair),
dΦ2 = p*/[16π² sqrt(s_pair)] dΩ_e*.
```

For an unpolarized rest neutron, integrate the overall neutrino orientation and express the relative electron azimuth as its average. This yields the original rest differential measure

```
[eν* p*/sqrt(s_pair)] × amplitude
  / [64 π³ mn hbar] × deν* dcos_e* × average_phi.
```

The moving-parent spin-two density is p²dp/[π²(hbar c)³], laboratory time dilation is 1/gamma, and fixing E inserts deν*/dE=1/D. Divide the volume emission spectrum by the one-active-helicity neutrino DOS E²/[2π²(hbar c)³]. Changing p to parent energy A then gives the new per-occupation measure

```
p dA/[32 π³ E² hbar]
 × [dc*/2]/D × [eν* p*/sqrt(s_pair)]
 × dcos_e* × average_phi × amplitude.
```

Both the baryon spin factor and the active-neutrino helicity are therefore fixed by the original invariant measure. There is no extra spin two for neutrinos. The coefficients j and true chi are s−1; χ/c is true absorption, and (j+χ)/c is relaxation. No gray weighting or source moment is evaluated here.

As an additional absolute coordinate check, the transformation to the old parent/cone variables satisfies

```
|∂(p,c*)/∂(z,v)| = 2 T A width v/p.
```

`phase_jacobian.py` checks this determinant with an independent finite-difference sequence at both energies. The smallest-step errors are about 7.2e−11 and 5.4e−11. These are unintegrated coordinate probes, not extra spectral profile calls.

## Actual occupation directions

With the unchanged total chemical potentials in the copied fixed bath,

```
j: f_n(A) [1−f_p(Ep)] [1−f_e(El)]
chi: [1−f_n(A)] f_p(Ep) f_e(El).
```

Both directions are independently accumulated in log space from their actual FD products. Neither is generated from KMS. The diagnostic j/chi=exp[−(E−mu_antinu)/T], mu_antinu=mu_n−mu_p−mu_e, follows from event energy conservation. The exact bath values and all controls are in every raw result.

## Eight-call evidence and its limits

Only eight fresh integrations were made: two original tail profiles at E=1.2 and three candidate profiles at each of E=1.2 and 1.3 MeV. Every call used one numerical thread, checked current quota immediately before dispatch, and finished below one second; each has an 85-second enforced wall budget.

The new parent-energy finite rule uses fixed excess-z panels to tail70. The independent generalized Laguerre rule covers the exact semi-infinite domain z∈[0,infinity) with weight z^(3/2) exp(−z). Its threshold power comes from cone width×pair momentum. Factoring that quadrature weight in logarithms leaves the full actual FD products and current intact; it is not a Maxwell–Boltzmann replacement. Thus the semi-infinite comparison checks the parent tail independently of a moving finite upper panel.

| Eν (MeV) | log j (s−1) | log true chi (s−1) |
| --- | ---: | ---: |
| 1.2 | −53.59471799782704 | −54.39940008263694 |
| 1.3 | −72.28178199277569 | −73.03646407758560 |

At each energy, fixed-panel 12/16/16/10 and semi-infinite 16/20/20/14 and 24/24/24/16 rules agree around 1e−14 in both forward logarithms. The profile tuple means parent/cone/electron/phi orders. These are coupled inner refinements, not separate isolated-axis certificates. Their agreement and exact endpoint maps support the selected author checkpoint; isolated angular-axis checks and a DIFFERENT integrated-phase oracle remain required before broader acceptance. No extra profiles were taken to spend past the eight-call authorization.

The new E=1.2 result differs from old tail70 by about 2.19e−6, but from old tail90 by about 2.22e−4. That one old cutoff happens to lie close to the converged answer does not close its failed tail stability gate. The old failed witness remains preserved as a qualification blocker for that original quadrature.

No whole-energy spectra/moments, upper incident-band convergence, transport evolution, EOS/GM1 binding, derivatives or empirical current uncertainty qualification follows. The selected E>E0 domain does not implement the below-endpoint cone-switch branch. Normal production CI and all drafts remain outside this research slice.
