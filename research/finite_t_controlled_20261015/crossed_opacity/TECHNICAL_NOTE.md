# Crossed neutron beta-decay opacity in the allowed-current ansatz

Research checkpoint, 15 October 2026. This directory completes the neutron-decay channel absent from the capture-only optical checkpoint. All earlier directories remain unchanged. It uses the same common-kinetic-mass, free-FD, nonrelativistic baryon response and leading allowed `V0/Aspin` vertices. The new process is `n → p + e− + anti-nu_e`, with inverse `p + e− + anti-nu_e → n`. **This is not a relativistic weak-current calculation, a collision-width treatment or a GM1 opacity.**

## Crossing and angular factor

For an emitted electron and antineutrino, define `c=cos(theta_e,antinu)` from their physical three-momenta. The emitted leptons remove their **sum** of momentum and energy from the baryons:

```
omega_decay = −E_e−E_nu,
q_decay = |p_e+p_nu|,
q_decay² = p_e²+E_nu²+2 p_e E_nu c.
```

The inverse transfers `+E_e+E_nu` and `+p_e+p_nu` into the proton-to-neutron transition. Using the capture expression `omega=E_nu−E_e` or its difference-momentum q in this channel would violate its on-shell energy/momentum bookkeeping.

For the same unpolarized leading nucleon current, the symmetric emitted electron/antineutrino tensor gives `L00 ∝ E_e E_nu+p_e·p_nu` and `sum Lii ∝ 3 E_e E_nu−p_e·p_nu`. Its contraction therefore retains

```
W(c)=gV²(1+beta_e c)+gA²(3−beta_e c).
```

Equivalently the angular correlation coefficient is `(gV²−gA²)/(gV²+3gA²)`, multiplied by beta_e. This agrees with the leading beta-decay correlation shown by the primary [aCORN neutron-decay measurement](https://arxiv.org/abs/2012.14379), and with the independent [neutron-correlation discussion by Byrne et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4852819/). The checkpoint reproduces the correlation from its contraction; it does not borrow a lifetime normalization. The crossing relation used in neutrino-rate implementations is also discussed in [Chiesa et al., Appendix A](https://arxiv.org/html/2412.04570v2#A1). Sources were inspected on 15 October 2026.

The response Fourier convention, spin factor and lepton measure are identical to the previous allowed-opacity slice: S includes spin two and an explicit `2 pi delta(energy)`, in fm⁻³ MeV⁻¹; the lepton phase measure is `E_e p_e dE_e dc/(4 pi²)`. Crossing adds neither a spin factor nor a symmetry factor. The electron and antineutrino are different particles, and j is per one antineutrino occupation state, not integrated over its phase space.

## Spectral coefficients and chemical reciprocity

Write `S_np` with n as its initial species, p as its final species, and

```
K=GF² |Vud|² (hbar c)³/(4 pi² hbar).

j_nd(E_nu) = K int_(m_e)^infinity dE_e int_(-1)^1 dc
               E_e p_e W(c) [1−f_e(E_e)] S_np(q,−E_e−E_nu),

chi_inverse(E_nu) = K int_(m_e)^infinity dE_e int_(-1)^1 dc
               E_e p_e W(c) f_e(E_e) S_pn(q,+E_e+E_nu).
```

Thus electron **vacancy** enters beta emission, while electron **occupation** enters the three-body inverse absorption. `mu_e` is the total electron chemical potential in the same rest-mass convention as the earlier lepton reference. The coefficients are s⁻¹ and enter `C=j(1−f_antinu)−chi f_antinu`. The true inverse-absorption length coefficient is chi/c, while fixed-bath relaxation is `(j+chi)/c`.

Applying the chemical KMS relation to the reversed baryon response gives

```
S_np(q,−E_e−E_nu) / S_pn(q,+E_e+E_nu)
  = exp[(−E_e−E_nu+mu_n−mu_p)/T].
```

Multiplication by the electron vacancy/occupation ratio cancels E_e, so

```
j_nd/chi_inverse = exp[−(E_nu−mu_antinu)/T],
mu_antinu=mu_n−mu_p−mu_e.
```

The code calculates both integrals directly. It does not set one coefficient by imposing this identity afterward. The chosen `DecayState` orders its baryons n→p, the **spontaneous** direction, unlike `OpticalState`, which orders baryons in the absorption direction. These different contracts are documented deliberately to prevent a silent species reversal.

## Absolute static-target gate

For dilute targets at fixed density, the M→infinity response approaches `S_np=2 pi n_n delta(omega+Q)`, with `Q=O_n−O_p`. The energy delta in decay then enforces

```
E_e=Q−E_nu,
0<E_nu<Q−m_e.
```

The angular integral gives `2(gV²+3gA²)`. Consequently

```
j_nd_static = GF² |Vud|² (hbar c)³/(pi hbar)
                × n_n (gV²+3gA²) E_e p_e [1−f_e],
chi_inverse_static = same prefactor × n_p E_e p_e f_e.
```

These expressions reproduce the previous static neutron-decay coefficients with no fitted multiplier. The emitted volume number rate integrates `D E_nu² j dE_nu`; the neutrino-energy rate inserts one extra E_nu, where `D=1/[2 pi²(hbar c)³]` has no spin-two neutrino factor. Dividing the volume number rate by n_n yields the uncorrected decay probability per neutron.

The static limit is tested by raising the kinetic mass while keeping total target densities, T and Q fixed through the classical chemical potentials. It is a mathematical limit, not an alternative physical nucleon mass. The warm spectral tests approach static coefficients at the few-parts-per-million level for M=10⁷ MeV. The warm moments and the dilute-lepton uncorrected beta lifetime are checked separately in `results.json`. The restricted static lifetime is approximately 947.37 s, **not** the experimentally measured neutron lifetime: Coulomb, radiative and complete recoil-current corrections are absent and no empirical lifetime calibration is applied.

## Thermal support and composing channels

The static neutron-decay spectrum ends at `Q−m_e≈0.7823 MeV`; static positron capture begins at `Q+m_e≈1.8043 MeV`. Their combined static antineutrino coefficient has an exact intervening gap. In the finite-temperature quadratic-dispersion response, the neutron-decay contribution can be nonzero above the static endpoint because energy and momentum are exchanged with thermal nucleon motion. The new calculation therefore removes exact zero support of the neutron-decay contribution in that former gap at tested finite-T points. It does **not** use an ad hoc Q correction, artificial smoothing or a collision width.

An exponentially small coefficient need not equilibrate a mode on any relevant timescale. Removing a mathematical disconnection does not establish efficient spectral coupling or a well-conditioned closed-box evolution. Rates and timescales must be assessed on the chosen state and radiation grid before drawing that conclusion.

Furthermore, unbounded nonrelativistic dispersion formally permits very large-momentum tails outside p/M≪1. Such tails are not relativistic predictions. Generic offset choices can also make additional crossed processes relevant; this checkpoint adds the physically named free-neutron decay branch and does not claim a universal all-species weak-process inventory.

`combined_antineutrino()` sums positron-capture and neutron-decay emission/absorption only after the constituents have passed their individual gates. Both have the same matter-set antineutrino chemical potential, so their sum obeys the same reciprocity relation. Scattering, pair processes and higher-order current corrections remain excluded. The electron-neutrino capture channel is still the earlier capture reference; no production source-term replacement is made here.

## Numerical evidence and independence

Run

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/crossed_opacity/check_crossed.py
```

The suite checks spectral heavy-mass normalization, warm number/energy moments, dilute-lepton uncorrected lifetime, chemical KMS signs for both the crossed channel and its combination with positron capture, thermal support above the cold endpoint, exact angular correlation, and energy/angle/neutrino quadrature refinement. A separate `(q,E_e)` QUADPACK integration uses `c=(q²−E_nu²−p_e²)/(2 E_nu p_e)` and `dc=q dq/(E_nu p_e)`; its independent variable map/panel grid checks the optical integration at and beyond the static endpoint.

Both quadratures share the free response and current model. This is numerical and normalization evidence within one ansatz, not agreement with an independent physical interaction theory. Finite explicit electron and neutrino integration cutoffs are qualified only on the listed states. Log coefficients retain underflowed ordinary-value tails, but the panel scheme has not been qualified for arbitrary extremes.

`results.json` contains actual errors, states, masses, coefficients, moments, constants, timestamps, software versions and hashes of this slice and its imported dependencies. No dependencies were installed; no earlier prototype was edited. Research sibling imports do not prescribe NMB's eventual package layout.

An initial failed check is preserved in `initial_failures.json`: the independent adaptive quadrature's absolute tolerance was too loose for the small above-endpoint thermal tail, producing a 4.57×10⁻⁴ relative discrepancy at E_nu=1 MeV. Tightening that reference's absolute error control resolves the comparison; no physical prefactor or response was adjusted. This is a reference-integration qualification, not a fit to the desired result.

Additional degenerate-state tests then exposed a separate electron-mass endpoint convergence defect, preserved in `degenerate_endpoint_failures.json`. Direct Gauss-Legendre integration in electron energy converged too slowly when an above-endpoint thermal tail was weighted strongly near E_e=m_e. The production-side research quadrature now uses `E_e=m_e+x²` with its exact Jacobian on every energy panel. This removes the integrable square-root corner mathematically; it does not alter physical support or add a smoothing scale. Both the refined optical quadrature and independent adaptive reference must pass again after this repair.

The next gate is independent review of crossing signs and endpoint integration, followed by a controlled slab or fluid-element comparison of static and dynamical phase space using the same complete named channel inventory. Interacting/relativistic EOS inputs, complete current recoil terms and physical width corrections need their own derivations and qualification.
