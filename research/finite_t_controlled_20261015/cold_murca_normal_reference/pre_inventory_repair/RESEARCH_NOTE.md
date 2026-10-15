# Optional normal degenerate modified-Urca reference

**A source-labelled research recipe is implemented with four separate lepton/spectator energy channels and the complete cold proton angular regions on its declared neutron-rich domain.** It is not a correction to existing NMB and is not a new complete MU calculation. The accepted legacy port remains untouched. Absolute strong-interaction and current accuracy, combined superfluidity, warm/nonequilibrium kernels, widths and DU/MU matching remain open. Different independent review is required.

## Source choice and coefficient

We deliberately choose the numerical prescription in [Yakovlev, Kaminker, Gnedin & Haensel2001 Eq140 and Sec3.4](https://arxiv.org/pdf/astro-ph/0012122): C=8.1e21erg cm^-3 s^-1, alpha=1.13, beta=.68, n0=.16fm^-3, the proton-density factor and lepton Fermi velocity. Its common neutron/proton matrix-element approximation from Eq142 is explicit. This coefficient is taken from that source; it is neither rederived from a new OPE amplitude nor adjusted to match the legacy8.55e21 normalization. Rounded source coefficients limit physical precision regardless of printed arithmetic digits.

For each lepton l=e,mu, the chosen baseline is

```
Qn,l = C (mLn/mn)^3 (mLp/mp) (np/n0)^(1/3)
       alpha beta (T/1e9K)^8 (vFl/c).
Qp,l = Qn,l (mLp/mLn)^2 G(kn,kp,kl).
vFl/c = HC kl / sqrt[(HC kl)^2+ml^2].
```

mLn,mLp are **Landau** masses, not Dirac masses. Bare masses are recorded as mn939.56542052/mp938.27208816MeV; ratios are supplied independently. The physical bare-mass ratio in `(mLp/mLn)^2` is retained, rather than silently replacing it by a ratio of dimensionless input ratios. These effective masses are prescribed recipe data, not extracted from GM1. Electron/muon masses are.51099895069/105.6583755MeV. HC=197.3269804MeV fm and kB=8.617333262145e−11MeV/K.

For comparison only, [Page etal2004 Eqs23–26](https://arxiv.org/pdf/astro-ph/0403657) choose8.55e21, momentum factor `(ke+kmu)/1.68`, and alpha(kn)=1.76−.63(1.68/kn)^2, beta=.68. We save that distinct neutron normalization as a diagnostic and analytically check its mapping to ours. On the two test states our/that neutron total is.9741236677 and.8427794666; these are recipe differences, not calibrated equality or an uncertainty bound. Its old proton angular approximation is not used. In particular we do not silently re-label a source hybrid as an exact published full rate.

## Complete proton geometry on the admitted domain

The geometric source is [Gusakov2002 Eqs10–15](https://arxiv.org/pdf/astro-ph/0204334), independent of the chosen coefficient. Let kn,kp,kl denote positive Fermi wave numbers, with0<kl<=kp<=kn. Our neutron baseline is restricted to kn>kp+kl, where

```
An = 2pi (4pi)^4 / kn^3.
Ap = 0,                                      kn >= 3kp+kl;
Ap = 2(2pi)^5 (kl+3kp−kn)^2/(kn kp^3 kl),     3kp−kl <= kn < 3kp+kl;
Ap = 8(2pi)^5/kp^2 (3/kn−1/kp),              kn < 3kp−kl.
```

Both A include the neutrino4pi and all five material solid-angle integrals; their units are fm^3 when k are fm^-1. The full neutron formula outside its stated region is not implemented. The proton helper itself covers all three regions in the stated neutron-rich domain; a forbidden proton branch returns exactly zero from its support condition, not by clipping the old squared polynomial.

From the independent phase-space density-of-states products, the proton/neutron momentum-product ratio is kp²/kn². Therefore `G=(Ap/An)(kp/kn)^2` reduces to0, `(kl+3kp−kn)^2/(8kl kp)`, and `(3−kn/kp)/2` respectively. The middle/low boundary is C1; the outer boundary vanishes quadratically. This derives the remapping instead of fitting a prefactor to existing code. Equal branch matrix elements remain an explicit interaction approximation, not a consequence of geometry.

`angular_oracle.py` supplies a different positive construction. For five independent fixed-length isotropic vectors, split them into a two-vector and three-vector sum. Their normalized vector densities are

```
rho2(R)=1/(8pi a b R), |a−b|<R<a+b;
rho3(R)=length([|c−d|,c+d] intersect [|R−e|,R+e])/(16pi c d e R).
```

Integrating `4pi R² rho2 rho3 dR`, then multiplying by the five material solid angles and neutrino4pi, yields `A=128pi^5/(abcde) * integral overlap_length dR`. The executable evaluates this overlap-area integral with independent adaptive quadrature and explicit breakpoints, including permutations of the length grouping. This verifies absolute2pi factors and the **8**, rather than8/3, in the low-kn proton expression without relying on OCR or a relative curve fit. It does not validate an angularly dependent strong matrix element.

## Admission and reaction contract

`State` takes T in kelvin, kF in fm^-1, positive Landau/bare mass ratios, and a supplied rest-inclusive `delta_mu_np_MeV`. `evaluate()` requires actual charge closure ke³+kmu³=kp³ and vanishing nuclear/lepton affinities: delta_mu_np=mu_e and, when present, mu_mu=mu_e. Muons must be included above their cold onset, while exactly absent muons return exactly zero in both muon channels. The example nuclear affinity is prescribed to satisfy this boundary contract; no nuclear EOS was solved, so these fixtures are not newly self-consistent stellar states.

This deliberately narrow research admission also requires kn>=1.68fm^-1, nucleon vF/c<=.8, and kBT<=.03 times the smallest supplied-model kinetic Fermi scale. The numeric cuts are conservative **chosen admission controls**, not literature-certified error bounds or a complete empirical validity band. The NR nucleon kinetic scale uses kF²/(2mL); it does not substitute for a full EOS. Muon kinetic degeneracy is checked separately. Low-kn inputs, including the legacy negative-alpha example, are rejected; alpha is never floored or extrapolated as a density-dependent fit. The model remains approximate even when all admissions pass.

Four energy channels are returned: neutron/electron, proton/electron, neutron/muon and proton/muon. For each lepton, neutron spectator means `n+n -> n+p+l+antinu_l` and `n+p+l -> n+n+nu_l`; proton spectator means `n+p -> p+p+l+antinu_l` and `p+p+l -> n+p+nu_l`. These are the two radiating beta-equilibrium directions, not a calculation of neutrino absorption/time-reversed opacities. Each reported Q combines both escaping flavor energies. No number rate or spectrum is computed.

In a decay event material Le or Lmu increases by1 and the emitted antineutrino carries−1; capture reverses those signs. Baryon number and electric charge are conserved event by event. Leading degenerate equilibrium assumes equal opposing number rates, so net Ye,Ymu,Yp and radiation-flavor lepton drifts are zero; the returned ledger zeros are labelled assumptions rather than independently integrated moments. The material energy loss is the sum of the four positive Q channels, without an extra Q-value term. This interface does not support arbitrary chemical imbalance.

## Executed tests and limits

All **95 selected author checks pass**, with no numerical failed gate in this batch. Independent absolute angular checks span both proton regions, the internal boundary, exact outer threshold, and forbidden side, with three different vector groupings. Largest scaled discrepancy is6.65e−14 at an unchanged3e−12 gate. Separate tests cover the neutron absolute factor, phase-space remapping, C1 boundary convergence, T^8, Landau-mass powers, actual charge/muon closure, zero T, absent muons, and source-normalization mapping. Eleven controls reject invalid/nonneutral/nonbeta/warm/DU-open/out-of-slice/Boolean/nonfinite/complex inputs. This millisecond-scale algebra/quadrature batch is not an absolute-rate physical benchmark.

At T6.37e8K and prescribed mass ratios.731/.863:

| nB,Yp [fm^-3,1] | Qn,e | Qp,e | Qn,mu | Qp,mu | total Q [erg cm^-3 s^-1] |
|---|---:|---:|---:|---:|---:|
| .1937,.0673 |2.46578858e19|1.07617156e19|1.51653783e19|6.30237495e18|5.68873546e19|
| .413,.0519 |2.91036504e19|9.00642886e18|2.09865557e19|5.79041868e18|6.48870536e19|

The first state's electron proton channel uses Eq14 while its muon proton channel uses Eq15. At the original kn2.01,kp=ke.5,kmu0 forbidden-side diagnostic, the new geometric proton contribution is exactly zero. The original frozen parity study's two physical failures remain failed and untouched; this optional model neither erases them nor proposes an automatic production replacement.

Run `python outputs/nmb-finite-t-20261015/cold_murca_normal_reference/validate_reference.py`. `results.json` records states, channels, angular checks, rejections and source hashes. No frozen author/reviewer/production file, dependency or repository state was changed. A future review must assess the source remapping and conditional physical normalization separately from numerical agreement. Realistic cooling requires an explicitly selected interaction prescription and compatible EOS/pairing treatment; detailed finite-T MU requires the still-missing full matrix/current/phase-space and matching work.
