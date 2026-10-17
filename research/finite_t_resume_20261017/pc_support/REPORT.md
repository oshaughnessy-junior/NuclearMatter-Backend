# Exact conditional support for soft positron capture

Author checkpoint, 17 October 2026. Scope is the exact PR55 bath and current copied under the explicit `pr55_reference` namespace. This is a physical relativistic free-particle calculation, not the earlier nonrelativistic allowed-current prototype. The masses, Guo VA/F2/GP form factors, spin conventions and occupation definitions are unchanged. Source identities are pinned in `ORIGINAL_SOURCE_PIN.json` and the final manifest.

## Reproduced frontier

At Eν=0.1 MeV the unchanged PR55 fine spectral call fails `Final positive-energy baryon shell unresolved`. The first saved witness has El=0.788301782324728 MeV, lepton cosine −0.8656312023878316, omega=−0.688301782324728 MeV and q=0.6886354307050477 MeV. Its support starts at initial proton energy 3639812.416935636 MeV. Direct subtraction in b² produces a relative neutron-shell residual 4.71e−9, exceeding the original 1e−10 requirement. The final energy is large and positive; this is not a negative-energy state or analytically empty support.

The conditional map below does not weaken this guard or declare that tail absent. It describes the full physical event manifold using incoming proton energy and exact two-body events. A separate semi-infinite radial rule supplies a forward integral without imposing a finite incoming-energy cutoff.

## Exact support, with unsquared energy conservation

Consider the inverse process antiν(k)+p(a) → e+(l)+n(b) at fixed laboratory neutrino energy E>0. Let the incoming proton energy and momentum be A and p=sqrt(A²−mp²), and c the angle between the proton and antineutrino. The invariant is

```
s = mp² + 2 E (A−p c),
s_threshold = (mn+me)²,
D = [s_threshold−mp²]/(2 E).
```

For the soft branch D>mp, the exact minimum incoming proton energy is

```
A0=(D²+mp²)/(2D),  p0=(D²−mp²)/(2D).
```

Every A≥A0 has the conditional incoming support

```
−1 ≤ c ≤ (A−D)/p.
```

It is a continuous bijective event domain, rather than a cold support test. Define deltaA=A−A0 and evaluate the small positive quantity without subtracting nearly equal large numbers:

```
R = A+p−D = deltaA [1+(A+A0)/(p+p0)],
width = R/p,
c = −1+width u,       0<u<1,
s−s_threshold = 2 E R (1−u).
```

The last identity constructs the physical invariant excess directly. It never clips a negative discriminant into positive support. For E=0.1 MeV, A0=8498.883134642389 MeV. This is the physical threshold of an extremely rare target tail, rather than the 3.64-million-MeV coordinate witness above. Both energies lie on the same unbounded exact support manifold.

The final CM momentum is computed as

```
p_star = sqrt[delta_s (delta_s+4 mn me)]/(2 sqrt(s)).
```

Final electron and neutron CM energies are their ordinary exact two-body energies. Boosting the back-to-back future CM four-vectors by the total initial four-velocity enforces positive-energy shells and conservation. The original unsquared delta is independently checked as

```
A+E−El = sqrt[mn²+|pa+k−l|²].
```

The unchanged relative 1e−10 shell gate and additional scaled 1e−10 four-conservation/unsquared-delta gates are retained. No tolerances are relaxed to admit the outputs. Direct boosted shells are well conditioned at the selected threshold boosts, unlike the original coordinate's far ultrarelativistic nodes.

## Spin/current normalization and absolute spectral measure

The copied full-current scalar contraction is the spin-averaged-four amplitude for forward capture n+e+→p+antiν, exactly as in the reviewed source. In inverse absorption the active antineutrino has one helicity, so the initial-baryon-spin-averaged amplitude is twice this forward avg4 quantity. The proton phase density has two spin states. Starting from the invariant two-body phase space gives

```
χ(E) = integral dA dc dΩ_star
       [p p_star /(64 π⁴ E sqrt(s) hbar)]
       × amplitude_avg4 × f_p(A) [1−f_n(Eb)] [1−f_e+(El)].
```

The spontaneous PC emission coefficient uses the same event measure and amplitude with the **actual independent** product

```
j(E): [1−f_p(A)] f_n(Eb) f_e+(El).
```

Here f_e+ uses the total positron chemical potential −mu_e. The bath is exactly T=2 MeV, mu_n=919.8287578100662 MeV, mu_p=916.8449303084382 MeV and mu_e=0.17446333200826397 MeV. Both coefficients are s⁻¹ per active-antineutrino occupation. No external radiation occupation or neutrino density of states appears inside j or χ. Reconstructing volume source moments later uses E²/[2π²(hbar c)³], with one active helicity. True absorption length is χ/c; relaxation length uses (j+χ)/c. No moments or transport coefficients are evaluated in this slice.

As an absolute phase anchor, transform each event to the original source coordinates (El, cos(k,l), phi_a about k−l). The new-to-old three-coordinate Jacobian at fixed A is

```
|∂(El,cos(k,l),phi_a)/∂(u,cos_star,phi_star)|
    = width p p_star q / [p_e sqrt(s)].
```

Combining it with the original pe/[32π³ q E hbar] measure and its normalized azimuth dphi/(2π) gives exactly the new measure. No prefactor is fitted. `phase_jacobian.py` checks the determinant numerically at both incident energies. Finite-difference steps 0.01, 0.003, 0.001 and 0.0003 display the expected convergence, ending at relative residuals 1.67e−8 and 4.31e−8. The probe shares exact boosts but uses the original coordinates and an independently computed finite-difference determinant.

## Numerical maps and preserved failed attempts

The first incoming-u quadrature retained a sqrt(1−u) endpoint through p_star. Its order change exposed relative errors about 2e−5. Those rows and the original candidate are preserved, rather than discarded. The exact u=1−v² substitution, du=−2v dv, removes this corner. It changes coordinates, not physical support or any reaction parameter.

The finite radial rule uses z=(A−A0)/T and exact sqrt-z panels. Comparing tail70 and tail90 preserves every sub70 panel/node and adds only the 70–90 panel, so it isolates an explicit tail extension rather than moving all interior nodes. Neither a small weighted contribution nor an underflowed ordinary value is accepted as an absent channel.

An additional generalized Gauss–Laguerre rule integrates the exact z∈[0,infinity) domain. Its weight z^(3/2) exp(−z) matches the radial threshold power from width×p_star; the integrand is evaluated from the same actual FD products with the known quadrature weight factored algebraically in logarithms. It does not impose Maxwell–Boltzmann occupations, manufacture j from χ, or replace the full current. Orders 8,16,24,32 independently check the semi-infinite radial integral and agree with the finite-panel forward values near 1e−11 in log space. This supports selected numerical convergence; it is not a universal rigorous quadrature error bound.

## Results and limits

At the selected higher inner rule the forward logs are

```
E=0.10: log j = −3786.584859767180
        log χ = −3787.939541852008
E=0.15: log j = −2381.239481736936
        log χ = −2382.569163821738
```

All ordinary values underflow, but the physical logs are finite and positive-support. Literal external u/v spinor sums agree with the independent invariant current reduction. Chemical KMS follows from actual products as a diagnostic, with mu_antinu=mu_n−mu_p−mu_e; no rate direction is constructed from that identity. Inner, tail-extension and semi-infinite radial comparisons are recorded separately.

The exact original source failure remains a valid failure of that source's coordinate realization. The new result is a research candidate at two soft energies, pending DIFFERENT independent integrated-phase review. Literal-current agreement and a local absolute-Jacobian anchor do not substitute for that independent integration. This slice establishes no all-energy qualification, source moments, closed transport evolution, EOS/GM1 coupling, gradients, interaction/correlation corrections or empirical uncertainty estimate. The copied free current's named model limits remain intact.
