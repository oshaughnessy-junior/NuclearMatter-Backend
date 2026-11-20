# First weak-rate reference: controlled static free nucleons

Prepared 15 October 2026. Research prototype in the Genesis workspace; no NMB production source was edited. The purpose is a transparent spectral and moment contract for fluid elements, supplied trajectories and simple transport problems. This is **not a full interacting or relativistic finite-temperature Urca kernel**.

## Supported reactions and conventions

The three spontaneous channels are electron capture `e− + p → n + νe`, positron capture `e+ + n → p + anti-νe`, and neutron beta decay `n → p + e− + anti-νe`. Their inverse processes are supplied through the collision coefficient `chi`, including the three-body inverse of neutron decay. Neutrinos are massless with one helicity per species. Electron mass is finite. All energy arguments and `T=kBT` are MeV; `mu_e` includes the electron rest energy and `mu_e+ = −mu_e`. Target number densities are fm⁻³. The module does not infer electron chemical potential or impose charge neutrality.

Nucleons are classical and static: their thermal velocities, recoil, Pauli blocking, interaction potentials, pairing, momentum-space Fermi surfaces and width corrections are omitted. For consistency of the equilibrium closure, their common thermal wavelength is assumed; the neutron–proton rest-energy difference Q is kept. This is the infinite-common-mass kinetic approximation with a finite reaction Q. It is useful in dilute warm matter. It does not describe neutron-star core degeneracy or the direct-Urca momentum-triangle threshold.

The implemented spectral structure is the bare, classical-target specialization of the charged-current formulae in Appendix A, Eqs.43–44 of [Chiesa et al., arXiv:2412.04570v2](https://arxiv.org/html/2412.04570v2#A1). We set their correction factors to unity and blocking target factors to the initial densities. Our explicit prefactor includes `|Vud|²`; this is a convention that must be matched in comparisons. We do not copy their optional weak-magnetism or mean-field treatments. Their paper is an equation source; **no executable cross-code comparison has been performed**.

## Equations and returned quantities

Define `p_e=sqrt(E_e²−m_e²)`, `f−(E)=[1+exp((E−mu_e)/T)]⁻¹`, and `f+(E)=[1+exp((E+mu_e)/T)]⁻¹`. With natural-unit `GF` in MeV⁻², define

```
A = GF² |Vud|² (1+3 gA²) / (2 pi³ hbar)      [s⁻¹ MeV⁻⁵]
D = 1 / (2 pi² (hbar c)³)                  [fm⁻³ MeV⁻³]
```

The target number/energy moments are the following integrals, with `k=0` for number and `k=1` for neutrino energy:

```
lambda_ec[k] = A int_(max(m_e,Q))^infinity dE E p_e (E−Q)^(2+k) f−(E)
lambda_pc[k] = A int_(m_e)^infinity       dE E p_e (E+Q)^(2+k) f+(E)
lambda_nd[k] = A int_(m_e)^Q              dE E p_e (Q−E)^(2+k) [1−f−(E)]
```

Neutrino energies are respectively `E−Q`, `E+Q`, and `Q−E`. Positron capture therefore emits only above `E_nu=Q+m_e`; neutron decay emits only below `Q−m_e`. Inverse absorption has exactly the same support. A zero rate below support is kinematic, not numerical smoothing.

At a fixed neutrino energy, `spectral()` returns spontaneous emission `j` and **true** absorption `chi`, both s⁻¹, in

```
C[f_nu] = j (1−f_nu) − chi f_nu.
```

Each coefficient equals `(A/D) E_e p_e` times the appropriate target density and electron/positron occupation or vacancy. The neutrino number source is `int dE_nu D E_nu² C`, and its energy source inserts one additional `E_nu`. There is no neutrino spin factor of two. Vacuum transparent emission sets `f_nu=0`; an irradiated application must supply its actual occupation.

The true absorption inverse mean free path is `chi/c`. The coefficient for relaxation of an occupation perturbation in a fixed bath is `(j+chi)/c`, often called stimulated absorption. They must not be interchanged. A gray average additionally needs an explicitly chosen radiation spectrum.

`source_terms()` reports directional transparent rates separately, flavor-separated number emission and summed energy loss. Composition obeys `dYe/dt = (R_pc + R_nd − R_ec)/n_B` for free nucleons. An equilibrium composition source can vanish while the emitted energy remains positive.

## Detailed balance and energy accounting

Within this static classical approximation,

```
mu_n − mu_p = Q + T ln(n_n/n_p)
mu_nu_e = mu_e − Q + T ln(n_p/n_n)
mu_antinu_e = −mu_nu_e.
```

Both electron capture/inverse absorption and the two antineutrino channel pairs obey `j/chi=exp[−(E_nu−mu_nu)/T]`. A Fermi occupation with that chemical potential is stationary point by point. These are trapped-equilibrium statements in the declared approximation. **Transparent kinetic equilibrium must instead solve the sum of directional emission rates**, and need not occur at `mu_nu_e=0`. Zero target densities are accepted for spectra and rates, but logarithmic equilibrium statements require both densities positive.

For a matter EOS that includes rest mass, the matter energy change from radiation exchange is minus the neutrino energy source. There is no extra Q-heating term. The reaction Q is already in the emitted neutrino energy and changing matter composition. For an application evolving thermal energy alone, derive the composition correction from its chosen EOS. The present prototype does not evolve thermal energy, momentum or hydro.

## Numerical checks actually executed

Run from any directory:

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/weak_rates/check_reference.py
```

All 44 current checks pass; the full report contains measured errors, sampling states, quadrature orders, constants, library versions, UTC time and source hashes. The checks cover five `(T,mu_e)` points spanning pair-rich and degenerate electrons; independent lepton-energy QUADPACK integration; Gauss-Legendre order refinement; massless `Q=0` Fermi integrals; the degenerate massless finite-Q polynomial limit; spectral detailed balance and stationary collisions; positron support; and the normalization from occupation coefficients back to target moments. They do not constitute domain-wide qualification or a physical comparison to a different reaction theory.

The independent quadratures share equations and constants. Their agreement checks integration and conversion conventions, while the analytical limits check selected structure. They are not independent evidence for an omitted recoil or in-medium effect.

Two initial numerical failures are retained in `initial_failures.json`: a narrow Fermi surface was unresolved by a single integration panel, and an exponentially tiny positron integral defeated QUADPACK's absolute stopping rule. The first was resolved by splitting at chemical-potential offsets; the second by scaling out a common exponential in the independent reference. Current `results.json` records the repaired run. This is why relative agreement on suppressed rates needs explicit scaling.

The uncorrected allowed neutron-decay lifetime is about **947.37 s** with these inputs. This is a normalization diagnostic of the restricted kernel, not a prediction of the measured neutron lifetime. Coulomb, radiative, recoil and other precision corrections are absent. Do not calibrate this by silently multiplying all channels to a measured lifetime.

## Constants, provenance and restrictions

The prefactor uses `GF=1.1663785e−11 MeV⁻²`, `hbar=6.582119569e−22 MeV s`, `hbar c=197.3269804 MeV fm`, and `m_e=0.51099895069 MeV`, with [PDG 2025 physical constants](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-phys-constants.pdf) as the source. `Q=1.29333236 MeV`, `gA=1.2723` and `Vud=0.974` are explicitly frozen benchmark choices, rather than a current precision-fit claim. Their values are serialized in the numerical report. Paper equations were inspected on 15 October 2026; no external source code or datasets were incorporated.

No dependencies were installed. The research kernel uses the already available NumPy and SciPy; it is not JAX differentiable and does not establish a production dependency choice. The standard order of 96 is a convenience, not a universal convergence guarantee. Refine it, the panel scheme and the tail cutoff for each new domain; underflowed extreme tails require a log-rate interface.

## Next integration step

Compose the independently developed finite-T electron/positron density inversion with this module using `n_e,net=Ye*n_B`, `n_p=Ye*n_B`, `n_n=(1−Ye)*n_B`. First hold density and temperature fixed and evolve composition alone to transparent kinetic equilibrium. Compare the rate-balance root with its time evolution, record number and energy budgets separately, and test step-size convergence. This is an intentionally controlled application of capture physics, not a cooling calculation with a mismatched heat capacity.

The next physics extension should replace the static nucleon approximation by a finite-recoil response with nucleon Pauli blocking. It must be supplied with an EOS-consistent dispersion relation. Mean-field shifts, weak magnetism, muon channels and collision-broadened/modified-Urca comparisons then become separately qualified options. None should be mimicked by a universal multiplicative correction to this kernel.
