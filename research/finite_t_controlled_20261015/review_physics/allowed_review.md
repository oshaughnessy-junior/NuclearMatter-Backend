# Independent physics review: allowed-current capture opacities

15 October 2026. Reviewed `allowed_opacity/allowed_capture.py`, technical note, check source and recorded 51-check payload. Reviewer changed only `review_physics/`. Campaign account/window matched at review start (4% used). No full author test-suite replay was needed; new targeted probes execute independently.

## Disposition

**Accept the restricted electron/positron capture and inverse-absorption coefficients as an allowed static-current model with finite-mass lepton kinematics, nonrelativistic baryon recoil phase space and Pauli blocking. No blocking angular, normalization or reciprocity error found.** The coefficients have an absolute bare weak normalization, not a calibrated multiplicative fit.

**Do not replace a complete total weak collision source or cooling/composition source with this partial channel inventory.** Crossed neutron beta decay and its three-body inverse are absent. The new kernels may be used for explicitly named capture-only fixed-state comparisons; they are not a complete finite-recoil Urca backend or a full order-1/M weak-current calculation.

## Angular current and approximation order

The spin-averaged static current has `H00=gV²`, `Hij=gA² deltaij`, and no mixed vector/axial expectation for unpolarized free spin states. Contracting it with the symmetric lepton tensor gives

`W = gV²(1+beta_e c)+gA²(3−beta_e c)`.

The electron mass is retained through `beta_e=p_e/E_e`. The mass term supplies no additional symmetric trace contribution at these V−A vertices. The antisymmetric lepton tensor has no antisymmetric static hadronic tensor to contract with, so the same angular factor applies to neutrino and antineutrino absorption in this approximation. No physical weak-magnetism asymmetry can follow from it. Keeping W inside the dynamic response integral is correct because q depends on c; its odd term generally cannot be integrated away at finite recoil.

The underlying response already includes the two occupied initial spin states. Its scalar and individual axial spin correlators have the same free-gas phase-space factor in this spin-symmetric model. No additional baryon spin degeneracy should multiply the optical prefactor.

Retaining the quadratic dispersion in phase space while retaining only allowed static vertices is a coherent **specified leading-current ansatz**, with exact phase space for that dispersion. It resums kinematic recoil without supplying all current corrections at the same nominal inverse-mass order. It must not claim complete order-1/M accuracy, exact finite-q vector-current conservation, weak magnetism, a relativistic Ward identity, or a relativistic EOS response. Those require compatible convective, axial-charge, magnetic and possibly interaction/vertex terms. Where p/M or lepton energies/M cease to be small, this approximation loses physical control; the numerical quadrature tolerance is not its truncation error.

## Fourier factors and absolute units

With the reviewed spin-two `S` carrying explicit `2pi delta(energy)`, the lepton azimuth and momentum measure give `E_e p_e dE_e dc/(4pi²)`. The code's

`K=GF² |Vud|² (hbarc)³/(4pi² hbar)`

is consistent. K has units `fm³ MeV^-2 s^-1`, while `dE_e E_e p_e S` has `MeV² fm^-3`; the result is **s^-1**. Dividing true chi by c in cm/s gives cm^-1. The incoming neutrino has one helicity, with no factor-two spin average or occupancy degeneracy. Its density-of-states factor belongs in subsequent emitted number/energy moments, not in j or chi.

In the dilute infinite-common-mass limit `S→2pi n_a delta(omega−Delta)`, the angular integral `2(gV²+3gA²)` gives

`chi = GF² |Vud|² (hbarc)³ n_a (gV²+3gA²) E_e p_e(1−f_e)/(pi hbar)`.

This is the earlier static coefficient and the bare single-target cross section multiplied by `n_a c`, with no free normalization. The recorded M=1e7 MeV spectral discrepancies are at most **4.30e-6**; emitted capture number/energy moments differ by at most **3.66e-6** at the tested state. Mathematical heavy-mass convergence is an appropriate normalization gate, not a physical alternative nucleon model. The note correctly excludes radiative/Coulomb/neutron-lifetime calibration. In particular published inverse-beta normalizations that include inner radiative corrections must not be compared without removing that correction.

## Channel signs and chemical reciprocity

For absorption, `omega=E_nu−E_e` and `q=k_nu−p_e` into the baryons. `OpticalState.baryons` consistently orders the **absorbing** species:

- nu_e absorption: n→p with outgoing e−, `mu_lepton=mu_e`.
- antinu_e absorption: p→n with outgoing e+, `mu_lepton=−mu_e`.

Emission uses reversed baryons at `(q,−omega)` and an occupied charged-lepton state. Combining response KMS with `f_e/(1−f_e)` gives `j/chi=exp[−(E_nu−mu_nu)/T]`, where `mu_nu=mu_lepton+mu_b−mu_a`. Rest energies/offsets and chemical potentials have consistent zeros. This remains true at chemical imbalance; no beta-equilibrium substitution is necessary. Integrating both positive contributions instead of imposing the ratio afterward is a useful sign check.

The alternative `(q,E_e)` integration in the recorded suite uses the correct Jacobian `|dc/dq|=q/(E_nu p_e)`. Its agreement at two tested FD states is strong optical-integration evidence for the chosen S/current, not independent many-body theory. The panel/cutoff choices remain qualified on selected states; the q→0 distributional limit, very narrow Fermi edges, extreme thresholds and nonrelativistic tail contamination require state-specific convergence in wider use.

## Independent targeted checks

`allowed_probes.py` writes `allowed_probe_results.json`; all four aggregate checks pass:

1. Nine explicit **4×4 Dirac gamma-matrix traces and 2×2 Pauli spin traces**, including near electron threshold and signed angles, reproduce the finite-mass W within **6.22e-15 relative**. This derives the contraction independently rather than numerically integrating the assumed angular formula.
2. Absolute optical coefficients are invariant under a simultaneous common shift of both baryon offsets and chemical potentials at the selected state.
3. A selected arbitrary chemical-imbalance state satisfies integrated KMS.
4. Reversed baryons with opposite lepton chemical potential satisfy the corresponding KMS relation.

The latter three residuals are zero at the displayed floating-point precision. They are selected checks, not a domain sweep. A first reviewer-probe execution contained a NumPy operator-precedence error in its newly written spin trace; adding explicit matrix parentheses repaired that reviewer script before results were produced. No author code or physics gate was changed.

Primary angular-reference source inspected: [Vogel and Beacom, Eqs.8–10](https://arxiv.org/pdf/hep-ph/9903554), whose zeroth-order angular form supports the stated contraction; their quoted corrected normalization includes a radiative factor that this bare reference deliberately omits. [Reddy, Prakash and Lattimer](https://arxiv.org/pdf/astro-ph/9710115) supplies the response/phase-space framework. No external source implementation was executed.

## Gate for crossed neutron decay

Derive the crossed current and phase-space measure explicitly, rather than relabeling outgoing positive-lepton absorption. For inverse neutron decay `p+e−+antinu→n`, the transfer **into** baryons is `omega=E_antinu+E_e` and vector `q=k_antinu+p_e`. Absorption therefore uses p→n with an occupied electron, while spontaneous decay uses reversed n→p response at negative transfer with an electron vacancy. The combined KMS factor must yield `mu_antinu=mu_n−mu_p−mu_e`. Establish the angular contraction by crossing or an explicit trace, including its angle convention; do not assume a minus-sign replacement is sufficient.

Require: positive crossed j/chi, its pointwise and integrated detailed balance, independent phase-space integration, dilute heavy-mass recovery of the **unmodified bare** static neutron-decay spectrum/lifetime and its number/energy moments, and separate electron-blocked and near-endpoint tests. Preserve capture/crossed channel labels in totals. Assess kinematic regions requiring p~M before interpreting nonrelativistic all-energy tails as physical; completing the channel inventory does not repair the dispersion outside its domain.

Only after that channel gate and independent review should the complete new collision inventory enter one existing fluid-element or slab benchmark. A hybrid of finite-recoil capture with static neutron decay can be a labeled approximation study, but it must not be presented as a uniformly finite-recoil total source.
