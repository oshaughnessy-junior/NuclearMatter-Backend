# First finite-temperature thermodynamics reference checkpoint

15 October 2026. Scope: controlled homogeneous matter and simplified transport, consistent with the revised campaign plan. This deliverable is a small independent CPU reference calculation. It does not modify NMB, qualify an interacting EOS, or replace the existing Genesis Helmholtz kernel. Production NMB baseline remains the inspected commit `25e68182f726e5bd9c0b372c9a667536940fbd07`; this checkpoint does not claim fresh execution of that remote source.

## Delivered and reproduced

`lepton_reference.py` evaluates an ideal relativistic spin-1/2 lepton–antilepton gas at specified temperature and either total chemical potential or net conserved number density. It supplies separate particle/antiparticle densities, total energy, pressure, entropy, free energy, fixed-net-density heat capacity, number susceptibility, and the induced chemical-potential derivative. The same implementation accepts the electron or muon mass. `thermal_energy_density(state)` computes the fixed-density thermal increment without subtracting two large cold energies.

Run from the Genesis root:

```bash
python outputs/nmb-finite-t-20261015/thermodynamics/validate_reference.py
```

Measured result: **86/86 checks pass** on Python 3.13.13, NumPy 2.4.6, SciPy 1.17.1. The full actual values, tolerances and errors are saved in `validation_results.json`. No installation, accelerator, random sampling or new dependency was required. Files in this directory are the entire checkpoint; no external assets are needed.

The tests independently check exact massless formulas, a nonrelativistic Maxwell–Boltzmann limit, massive zero-temperature formulas, Sommerfeld heat capacity and thermal energy, density inversion, Euler identity, Maxwell relation, quadrature/tail refinement, and fixed-density derivatives against finite differences. These are selected benchmarks, not a dense domain qualification or a comparison with an independently installed Helmholtz EOS.

## Definitions and equations

Use `T=k_B T_physical`, lepton mass energy `m=mc²`, momentum energy `p=|p_phys|c`, and **chemical potential including rest mass** `mu`, all in MeV. Let `hbar*c=197.3269804 MeV fm`, `E=sqrt(p²+m²)`, and `A=1/[pi²(hbar*c)³]`, including spin degeneracy two. The gas is in pair equilibrium with antilepton chemical potential `-mu`:

```
f_minus = 1/[exp((E-mu)/T)+1]
f_plus  = 1/[exp((E+mu)/T)+1]
n_net   = A integral p² (f_minus-f_plus) dp
neach   = A integral p² f_each dp
e       = A integral p² E (f_minus+f_plus) dp
P       = A/3 integral p⁴/E (f_minus+f_plus) dp
s       = A integral p² [h(f_minus)+h(f_plus)] dp
h(f)    = -f log(f) -(1-f) log(1-f)
f_free  = e-Ts = mu*n_net-P
```

Densities and `s` are in fm^-3 with `k_B=1`; energy, pressure and free-energy density are in MeV fm^-3. `cv_fixed_net_density=de/dT|n_net` therefore has units fm^-3. Conversion to a physical temperature derivative multiplies by `k_B` in the relevant energy/K convention. To get entropy in erg cm^-3 K^-1, multiply the numerical `s` by `10^39 k_B[erg/K]`. To get volumetric heat capacity in those units, use the same conversion on `cv`. Energy and pressure use `10^39` times MeV-to-erg.

The occupation entropy uses the stable expression `log(1+exp(-|x|))+|x|/[1+exp(|x|)]`, where `x=(E-mu)/T`. Net-number subtraction near `mu=0` uses an exponential-difference identity, avoiding cancellation of nearly equal Fermi factors. Negative net density is supported by particle/antiparticle symmetry. Pair densities remain nonzero at vanishing net charge, provided they have not underflowed.

For response moments define `w_minus=f_minus(1-f_minus)` and `w_plus=f_plus(1-f_plus)`:

```
A0 = A integral p² (w_minus+w_plus) dp
A1 = A integral p² [(E-mu)w_minus-(E+mu)w_plus] dp
A2 = A integral p² [(E-mu)²w_minus+(E+mu)²w_plus] dp

 dn/dmu|T = A0/T
 dn/dT|mu = A1/T²
 dmu/dT|n = -A1/(T A0)
 Cv|n     = [A2-A1²/A0]/T²
```

These expressions remove the large rest/degeneracy energy from the heat-capacity moments. The variance form guarantees nonnegative `Cv` in exact arithmetic. They differentiate at frozen **net** charge while allowing pairs to equilibrate. Holding the electron particle density fixed instead gives a different derivative.

For the thermal energy, let `pF=(3pi²|n_net|)^(1/3) hbar*c`, `mu0=sqrt(pF²+m²)` and `theta0=theta(mu0-E)`. At fixed net density,

```
e(T,n)-e(0,n) = A integral p² [(E-mu0)(f_minus-theta0)
                              +(E+mu0)f_plus] dp .
```

Subtracting `mu0` times the exactly vanishing number difference provides this stable expression. Below the cold Fermi surface the occupation difference is evaluated as `-1/[1+exp(-(E-mu)/T)]`, avoiding subtraction from unity. At the cold electron benchmark, the total energy is about `0.9864 MeV fm^-3` while the thermal increment is only `3.7521e-10 MeV fm^-3`.

## Measured reference states

All densities below are net densities. Energies include rest mass and both members of the pair plasma.

| State | T [MeV] | n [fm^-3] | mu [MeV] | P [MeV fm^-3] | e [MeV fm^-3] | Cv [fm^-3] |
|---|---:|---:|---:|---:|---:|---:|
| Pair-rich electrons | 10 | 1e-6 | 0.23058423 | 4.9936751e-4 | 1.4986672e-3 | 5.9931605e-4 |
| Degenerate electrons | 0.001 | 0.01 | 131.52150 | 0.32879630 | 0.98641868 | 7.5042890e-7 |
| Warm muons | 20 | 0.005 | 134.04016 | 0.12922285 | 0.76581432 | 8.4000719e-3 |
| Cold muons | 0.01 | 0.002 | 130.68819 | 0.019125841 | 0.24225057 | 4.3607277e-6 |
| Low net electron density, pair supported | 0.1 | 1e-12 | 0.03070565 | 3.3677463e-13 | 2.3266523e-12 | 1.5340899e-10 |

The final row illustrates why net density alone cannot define whether pair contributions are negligible.

## Numerical boundaries and lessons

Integration uses kinetic energy `k=E-m`, with Jacobian `pE`, to avoid momentum-domain endpoint difficulties. Panels split at the Fermi surface and ±60 T. A refinement test extends to 80 T and tightens relative quadrature tolerance from 2e-12 to 5e-13. Density inversion uses bracketed Brent iteration and the exact odd symmetry of net density. Quadrature failures raise an error; a roundoff-only status is accepted only when its reported error already meets the requested absolute/relative tolerance. This accommodates SciPy’s stricter internal stopping target without ignoring an unresolved error estimate.

An earlier trial at quadrature tolerance 2e-10 passed bulk-value comparisons but missed the cold electron fixed-density heat-capacity finite-difference test by about 0.17%. The error was the chemical-potential inversion's precision relative to its tiny thermal shift. Tightening to 2e-12 restored that test without changing the response-moment formulas. The cold finite-difference step is 0.05 T, because subtracting nearly equal total energies at 0.002 T magnifies floating-point/root error. The independently known Sommerfeld result and the stable thermal increment provide the stronger cold reference.

Selected tests span T=1e-4–20 MeV and massless/electron/muon limits, with one massless degenerate benchmark at mu/T=1e4. They do not establish every state in that rectangle. Very cold dilute massive gases at mu≈0 have exponentially underflowing susceptibility; the reference raises an explicit error. There is no T=0 entry point. Cooling temperatures substantially below the demonstrated cases need a matched Sommerfeld/asymptotic implementation. Root and second derivatives have not been qualified across that wider domain. No magnetic, Coulomb, nuclear, pairing or trapped-neutrino contributions are included. This is not an EOS for total matter.

## Integration and next bounded milestone

For a nucleonic composition interface, the independent inputs are `(nB,T,Ye,Ymu)` with `ne_net=Ye*nB`, `nmu_net=Ymu*nB` and `Yp=Ye+Ymu`. Compose lepton free energies with the baryonic free energy and optional photons. At fixed `(nB,Ye,Ymu)`, add the fixed-density sector heat capacities. If `Ye` or `Ymu` evolves, include its contribution to the derivative of total energy; a tabulated equilibrium `Cv` is not interchangeable with that frozen-composition derivative.

A deliberately simple closure can already support a transparent capture element and closed thermal box: classical free neutron/proton gases with one shared kinetic mass in their thermal wavelength but distinct rest masses, plus this lepton gas and photons. With spin degeneracy two,

```
f_i = n_i [m_i + T(log(n_i lambda_M³/2)-1)]
lambda_M = sqrt(2pi/(M T)) hbar*c
mu_n-mu_p = (m_n-m_p)+T log(n_n/n_p).
```

This exactly matches the shared-thermal-wavelength static-target weak ansatz developed by the weak-rate worker. Use it only where nucleon degeneracy and interactions are intentionally neglected; it supplies controlled composition/energy accounting rather than quantitative dense-neutron-star predictions. Retain consistent rest masses in weak Q values and matter energy, so weak composition energy is not added twice.

Next accept this reference as an independent oracle and implement a separate JAX quadrature/asymptotic evaluator, with implicit density inversion `dmu=(dn_target-n_T dT)/n_mu`, validating primal values before tangents. Then add one RMF free-energy construction at arbitrary composition. Require cold-limit reproduction, scalar/vector field stationarity, Gibbs/Maxwell relations and fixed-composition derivative qualification before using its dispersions in Urca. Density-dependent couplings need their rearrangement terms. Treat compact thermal-index fits as benchmark products; they do not supply chemical potentials or weak responses.

The requested application scale permits deterministic converged quadrature and small spectral problems. Throughput optimization, GPU kernels and multidimensional GRMHD adapters are unnecessary at this stage.

## Sources and provenance

Equations above are explicit ideal-gas definitions and independently derived response identities implemented in this checkpoint. Sources orient the conventions and future reference comparisons:

- [FLASH developer guide: Helmholtz EOS](https://flash.rochester.edu/site/flashcode/user_support/flash_ug_devel/node113.html) distinguishes rest-mass-subtracted electron degeneracy and the corresponding positron convention. Our `mu` includes rest mass, so `eta_FLASH=(mu-m)/T` and `eta_pos=(-mu-m)/T`.
- [Timmes Fermi–Dirac integral reference](https://cococubed.com/code_pages/fermi_dirac.shtml) documents the relativistic two-parameter integral convention and links the reference EOS implementation.
- [Timmes & Swesty 2000, ApJS126,501](https://www.pas.rochester.edu/~yishengtu/research_files/EOS_reference/HELM_Timmes_2000_ApJS_126_501.pdf) motivates thermodynamic consistency through a common free-energy potential. No source code or tables were copied here.
- [Campaign research plan](https://github.com/oshaughnessy-junior/nmb-papers/blob/master/notes/PLAN-FINITE-T-CONTROLLED-TRANSPORT.md) establishes the broader controlled-transport scope and references.

## Independent review repair

The numerical reviewer found that a fixed absolute chemical-potential root tolerance could return an incorrect tiny net density near pair symmetry, including a zero result for a positive requested density. The revised inversion uses the pair-state susceptibility to scale its bracket, solves a dimensionless root and explicitly enforces a relative charge residual of 1e-9. Eighteen signed massless near-symmetry regressions were added; the full suite now passes86checks. The original counterexamples remain under review_numerics/. This repair does not qualify the exponentially underflowing massive-pair regime.
