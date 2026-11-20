# An allowed-current optical reference with recoil and Pauli phase space

Prepared 15 October 2026. **Research checkpoint, not a production opacity release.** This slice contracts the earlier dynamical density response with a fully specified leading allowed nucleon current, retaining finite electron mass, charged-lepton blocking, baryon recoil in energy/momentum conservation and baryon Pauli blocking. It supplies electron/positron capture inverses only. It does not include neutron beta decay and its three-body inverse, finite-recoil current corrections, weak magnetism, RPA, pairing or a relativistic GM1 current.

## Primary sources and approximation order

[Reddy, Prakash and Lattimer (1998), Eqs.5–16](https://arxiv.org/pdf/astro-ph/9710115) supply the golden-rule phase-space normalization and dynamic response conventions. [Vogel and Beacom (1999), Eqs.7–10](https://arxiv.org/pdf/hep-ph/9903554) give the finite-mass lepton angular factor at zeroth order in the nucleon-current expansion. We retain the angular term they display, with no radiative correction. The finite-mass lepton extension below follows from the symmetric lepton tensor. No published numerical table or external rate implementation was used in this checkpoint.

The vertices are the allowed current `J⁰=gV tau±`, `Jⁱ=gA sigmaⁱ tau±`; recoil and thermal velocities enter **phase space**, rather than a full order-1/M expansion of the current. This is a common controlled leading-current construction, but is not a complete order-1/M calculation. In particular it cannot predict the physical neutrino/antineutrino weak-magnetism asymmetry or a relativistic medium response. The two spin states are unpolarized, the spin transition response is free and identical to the density response, and residual spin interactions are absent.

## Angular contraction and absolute normalization

After spin averaging, the surviving hadronic contractions are `H00=gV²`, `Hij=gA² deltaij`, with zero mixed vector/axial spin average. The symmetric leptonic terms are proportional to

```
L00 = 8 E_nu (E_e + p_e cos(theta))
sum_i Lii = 8 E_nu (3 E_e − p_e cos(theta)).
```

Consequently the dimensionless angular weight is

```
W = gV²(1+beta_e cos(theta)) + gA²(3−beta_e cos(theta)),
beta_e=p_e/E_e.
```

The lepton antisymmetric tensor does not contribute to these static symmetric vertices. The same W therefore applies to electron-neutrino and electron-antineutrino absorption in this approximation. Its angular integral is exactly `2(gV²+3gA²)`, including the electron threshold where beta_e vanishes. At finite recoil one must retain the angular weight inside the response integral; its odd term need not integrate away because q depends on angle.

The response S is the earlier spin-two quantity with an explicit `2 pi delta(energy)`, in fm⁻³ MeV⁻¹. Set `omega=E_nu−E_e`, `q²=E_nu²+p_e²−2 E_nu p_e cos(theta)`, with energy transferred **into** the absorbing baryons a→b. The true occupation absorption coefficient is

```
chi(E_nu) = K int_(m_e)^infinity dE_e int_(-1)^1 dc
                E_e p_e W [1−f_e(E_e)] S_ab(q,E_nu−E_e),

K = GF² |Vud|² (hbar c)³ / (4 pi² hbar).
```

The factor `1/(4 pi²)` comes from `d³p_e/(2 pi)³ = E_e p_e dE_e dc/(4 pi²)` after its azimuthal integration. S already includes the baryon spin factor, so inserting another two would double count it. `GF` has units MeV⁻². The explicit `(hbar c)³` converts S's fm⁻³ measure to natural energy units; `hbar⁻¹` converts the remaining energy into inverse seconds. There is no incoming-neutrino spin average or degeneracy factor two.

The spontaneous inverse emission coefficient is computed **independently in the same integral**, replacing `[1−f_e]S_ab(q,omega)` by `f_e S_ba(q,−omega)`. The species labels in `OpticalState.baryons` always refer to absorption. For `nu_e+n→p+e−`, `(a,b)=(n,p)` and `mu_lepton=mu_e`. For `anti-nu_e+p→n+e+`, `(a,b)=(p,n)` and `mu_lepton=−mu_e`. Both electron chemical potentials include rest mass in the paired lepton convention.

## Detailed balance and transport meanings

The FD and response KMS identities imply pointwise

```
j/chi = exp[−(E_nu−mu_nu)/T],
mu_nu = mu_lepton + mu_b − mu_a.
```

The implementation computes both directions rather than imposing this identity after integration; their agreement is a check on signs, offsets and occupations. The collision term is `C=j(1−f_nu)−chi f_nu`. Returned `j` and `chi` are s⁻¹. True absorption is `chi/c` in cm⁻¹; fixed-bath occupation relaxation is `(j+chi)/c`. A gray opacity needs a supplied spectral weighting and is not yet delivered.

## The independent static-target gate

For classical targets with fixed initial density n_a and common kinetic mass M→infinity, the dynamical response tends to `2 pi n_a delta(omega−Delta)`, where `Delta=O_b−O_a`. Inserting that distribution into the contracted integral gives

```
chi_static = GF² |Vud|² (hbar c)³/(pi hbar)
             × n_a (gV²+3gA²) E_e p_e [1−f_e],
E_e=E_nu−Delta.
```

This agrees algebraically with the earlier static-capture coefficient, without any fitted normalization. Equivalently the bare single-target total cross section is

```
sigma_static = GF² |Vud|² (hbar c)²/pi
               × (gV²+3gA²) E_e p_e           [fm²],
chi_static = n_a[cm⁻³] c[cm/s] sigma[cm²] [1−f_e],
1 fm²=10⁻²⁶ cm².
```

The test obtains chemical potentials from the classical density formula at each kinetic mass, keeping target density and Q fixed. It never rescales K to reproduce a test point. M is varied from 939 to 10⁷ MeV as a mathematical convergence test, not as an alternative physical nucleon mass. At M=10⁷ MeV the spectral coefficients approach the prior static reference to a few parts per million, and independently integrated number and neutrino-energy moments approach its capture moments within 4×10⁻⁶ at the tested state. The measured neutron lifetime and corrected inverse-beta cross sections are not used to calibrate the bare prefactor.

## Numerical implementation and executed checks

Run

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/allowed_opacity/check_allowed.py
```

`allowed_capture.py` uses deterministic Gauss-Legendre lepton energy and angle integrals, with panels resolving the static energy, Doppler/recoil scales and lepton Fermi surface. Positive integrands accumulate via log-sum-exp; the underlying response retains log thermal tails. The electron-energy upper bound is explicit and varied in the validation. These panel choices are tested on the modest states recorded in `results.json`; they are **not a universal endpoint or arbitrary-degeneracy integration proof**.

The checks include chemical detailed balance, absolute spectral normalization against both `static_capture.py` and the independent bare cm² cross-section expression, integrated emitted number/energy normalization, inner and outer quadrature refinement, lepton-energy cutoff extension, exact angular integrals and unit conversion. A separate QUADPACK calculation in `(q,E_e)` variables checks two finite-mass FD states, using `dc = −q dq/(E_nu p_e)` and an independently chosen energy panel grid. That comparison shares S and the chosen current, so it checks optical integration rather than a separate microscopic theory.

The initial normalization/quadrature batch passed all checks; no failed calibration was discarded. Constants are imported directly from the previous static reference and serialized indirectly through the recorded source hashes; `K` and the tested states are written to the result file. No dependencies were installed. Research imports connect sibling checkpoint modules and are not a proposed production package layout.

## Scientific limits and next step

The common kinetic mass and independent total offsets must match the supplied baryon occupations. Prescribed offsets do not turn this into a conserving interacting response. NR unbounded dispersion has formal all-energy thermal tails, including regions outside p/M≪1; a relativistic restriction cannot be pasted over them without rederiving the response. This checkpoint does not impose hidden cold thresholds or numerical smoothing.

Neutron beta decay requires a crossed charged-lepton channel and its three-body inverse, not another label for the outgoing-positive-lepton integral. Thus the antineutrino spectrum here is incomplete as a total weak collision source. No transparent equilibrium, cooling or trapped-box calculation should replace the complete earlier static channel inventory with this partial inventory without explicitly retaining that limitation.

The next useful step is independent review of current normalization and mathematical support, followed by the crossed neutron-decay channel in the same allowed-current ansatz. Then one can compare finite-recoil/Pauli and static results on a controlled fluid element or spectral slab. A physically broader opacity needs a separately qualified relativistic dispersion/current treatment, weak magnetism and consistent EOS inputs; this calculation alone does not provide those capabilities.
