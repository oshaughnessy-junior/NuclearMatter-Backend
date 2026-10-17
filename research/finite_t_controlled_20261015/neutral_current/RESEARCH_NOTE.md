# Free-nucleon neutral-current redistribution reference

Author checkpoint, 2026-10-15. **189/189 author checks pass** in 5.33 s after the numerical repair below. Independent review is required. This research leaf supplies a physically normalized selected-energy scattering kernel and conservative discrete collision example. It does not certify an interacting-matter opacity, continuum energy-grid evolution, or a production backend.

## Model and primary provenance

Use uniform, free, unpolarized nonrelativistic nucleons with dispersion ε(p)=offset+p²/(2M), Fermi–Dirac occupations, spin degeneracy two, and a prescribed thermal bath. Obtain each species from the frozen `nr_baryon_eos/free_nucleons.py`; use its chemical potential and kinetic mass without an additional neutrality or beta-equilibrium condition. Nuclear Pauli blocking and quadratic-dispersion recoil are retained. The weak current is leading allowed order: no weak magnetism, recoil corrections to the current, relativistic correction, RPA, interaction/spin correlations, or in-medium axial renormalization. Independent neutron/proton kernels may be added for a free mixture.

The differential normalization and density/spin convention follow equations 1–2 of [Bedaque, Reddy, Sen & Warrington, 2018](https://arxiv.org/html/1801.07077). The independent elastic total/transport checks use equations 9, 14, 18, 20 and 23 of [Burrows, Reddy & Thompson, 2006](https://arxiv.org/html/astro-ph/0404432); its equation 33 also fixes the collision occupation measure. Both primary sources were read on 2026-10-15. The interaction calculations in those papers are **not** implemented here.

## Response, couplings and units

Let E and E′ be incoming/outgoing neutrino energies, u=cosθ, ω=E−E′ (positive into matter), and q²=(E−E′)²+2EE′(1−u). Energies, masses, T and q are MeV; densities fm⁻³. With a momentum measure in physical fm units, the imported spin-summed response is

S(q,ω)=2 ∫d³p/(2πℏc)³ f(p)[1−f(p+q)] 2πδ[ω+ε(p)−ε(p+q)].

Thus S has units fm⁻³ MeV⁻¹. The leading free density response equals each Cartesian Pauli-spin response: the unpolarized spin trace is Tr(σaσb)=2δab. There is no extra factor three multiplying S; the angular coefficient below already contracts the three components. This equality fails in general interacting matter.

Declare GF=1.1663785×10⁻¹¹ MeV⁻², gA=1.2723, sin²θW=0.23129, ℏc=197.3269804 MeV fm. Couplings are (CV,CA)n=(−1/2,−gA/2), (CV,CA)p=(1/2−2sin²θW,+gA/2). There is no CKM factor. Relative signs do not affect this unpolarized leading-order kernel, which has no vector–axial interference. Neutrinos and antineutrinos of all active flavors share this approximation; precision electroweak corrections are outside scope.

The bare kernel per occupied incident neutrino is

K(E→E′,u) ≡ d²Γ/(dE′du)
= GF²(ℏc)³/(4π²ℏ) E′² [CV²(1+u)+CA²(3−u)] S(q,ω).

Its units are s⁻¹ MeV⁻¹. It includes the outgoing E′² phase space and outgoing azimuth integration. Multiply by [1−fν(E′)] once for final neutrino blocking; initial fν is separate. Do not multiply this kernel by another outgoing density of states. Exact q=0 is distributional and explicitly rejected; Gauss angular nodes exclude that isolated forward-elastic point.

The neutrino number measure for **one helicity** is g(E)dE=E²dE/[2π²(ℏc)³]. In an isotropic collision integral expressed using this bare K, gain has the measure ratio E′²/E²:

C(E)=∫dE′du {(E′²/E²)K(E′→E,u) fν(E′)[1−fν(E)] − K(E→E′,u)fν(E)[1−fν(E′)]}.

Nuclear chemical potentials cancel from neutral-current detailed balance, but remain in the response amplitude and nuclear blocking:

S(q,−ω)=exp(−ω/T)S(q,ω),

K(E′→E,u)/K(E→E′,u)=(E/E′)²exp(−ω/T).

Consequently FD radiation at the bath T is stationary for **any** neutrino chemical potential. Neutral-current scattering conserves neutrino number and cannot by itself enforce μν=0. This is redistribution rather than an emission/absorption scalar relaxation opacity.

## Independent normalization checks

In the dilute heavy-static limit S→2πnδ(ω). Integrating the angular weight gives 2(CV²+3CA²), or (4/3)(CV²+5CA²) after multiplying by 1−u. Dividing Γ by n c, converting fm²→10⁻²⁶ cm², yields

σtotal=GF²E²(ℏc)²/π × (CV²+3CA²),

σtransport=2GF²E²(ℏc)²/(3π) × (CV²+5CA²).

At E=10 MeV the declared couplings give neutron (2.46865894, 2.55560184)×10⁻⁴² cm² and proton (2.04947681, 2.27614709)×10⁻⁴² cm² for (total, transport). Independent literature formulas expressed through σ0 reproduce these values. Direct finite-mass kernel integration at M=10⁷ MeV agrees within 6.54×10⁻⁶; this is a limiting experiment, not a physical nucleon mass. Pure-vector σtransport/σtotal=2/3 and pure-axial=10/9 independently test angular weighting.

## Selected finite-mass experiments

Run `python outputs/nmb-finite-t-20261015/neutral_current/validate_nc.py` from the workspace root. It writes every gate and raw value to `validation_results.json`. No dependency installation is needed. The frozen response/EOS hashes are recorded there.

Three single-neutron baths have (T [MeV], n [fm⁻³])=(2,0.00007), (2,0.002), (0.5,0.005); their typical pF/M values are approximately 0.027, 0.082, 0.112. Use these only within the free NR ansatz. The final bath is more degenerate and more sensitive to a coarse energy grid. At incident 10 MeV:

| T | n | Γ [s⁻¹] | ∫(1−u)dΓ [s⁻¹] | ∫ωdΓ [MeV s⁻¹] |
|---:|---:|---:|---:|---:|
| 2 | 0.00007 | 4998.61172 | 5142.91423 | −105.21425 |
| 2 | 0.002 | 96684.10306 | 99462.94824 | −2945.86711 |
| 0.5 | 0.005 | 49785.87771 | 52126.54269 | 23773.93589 |

These are bare incident-event moments without a radiation occupation. A single-energy energy-transfer moment may have either sign; equilibrium requires the complete occupied and blocked spectrum. Γ/c is an event inverse mean free path. The (1−u) moment is an angular transport diagnostic, not a complete inelastic transport closure or thermalization rate.

For E=5,10,20 MeV at each bath, angular orders 24→48, energy relative tolerance 2×10⁻⁹→2×10⁻¹¹ and high-energy tail 40T→50T leave all 27 moments unchanged within 2.1×10⁻¹⁵ by the recorded scaled comparison. This selected convergence result does not establish a universal parameter-domain guarantee. Thirty-six pointwise reverse-kernel KMS checks, including nearly equal energies, agree within 3.6×10⁻¹⁵. A separate q-variable angular integral supplies a coordinate-level cross-check. Final neutrino FD blocking is tested explicitly.

## Closed-grid conservation demonstration

For a deliberately closed eight-energy grid, set Aij=wj∫du K(Ei→Ej,u), gi=Ei²wi/[2π²(ℏc)³], and pair flux Jij=gi Aij fi(1−fj). The returned bin number source is dNi/dt=Σj(Jji−Jij), and the occupation source is this divided by gi. Pairwise antisymmetry conserves number independently of equilibrium. Direct contraction of (Ei−Ej)Jij gives the bath energy exchange; independently contracting Ei dNi gives its opposite. Tests include arbitrary-μν FD stationarity, off-equilibrium number conservation, energy accounting and nonnegative radiation-plus-bath entropy production.

The grid intentionally excludes transitions outside its nodes. It is a reversible finite collision system, **not** a converged continuum evolution. In particular, a coarse grid can miss narrow energy redistribution in a degenerate bath while satisfying conservation exactly. Its numerical energy exchanges must not be quoted as converged physical heating/cooling rates. No time integrator or coupled material evolution is supplied.

## Integration recommendation and remaining gates

Keep this kernel as a separate neutral-current scattering operator beside charged-current j/χ. Use EOS-derived occupations consistently, retain the neutrino phase-space ratio in gain, and use pairwise exchange accounting when coupling a fixed-bath experiment to a material energy ledger. For anisotropic transfer, retain angle dependence rather than replacing all scattering with the selected (1−u) diagnostic. Next steps are independent prefactor/response review, a genuinely resolving energy discretization with tails and leakage, and a bounded continuum stationary/non-equilibrium experiment. Interacting nuclear response and relativistic/current corrections need separately sourced models and validation.

The initial 167 author gates passed, but independent review exposed missing invalid-input contracts and a cancellation failure on an additional irregular grid. The repaired suite now passes 189 gates, including the original 167. See the repair record below; independent rechecks are pending. This is not external physical validation. No frozen upstream file, production source, dependency pin, or git state was modified by this leaf.


## Numerical repair after independent review (2026-10-15 12:52 UTC)

The complete original author source, results, note and checkpoint, together with the independent failure reports, are preserved under `pre_repair_20261015T1250/` with a SHA256 manifest. No frozen upstream source was edited.

The numerics reviewer demonstrated that negative energy-tail controls, negative radiation temperature, and NaN grid weights/occupations could be accepted. The repaired `moments` entry validates finite positive incident energy, tail, integration relative tolerance and radiation temperature, finite radiation chemical potential, and an integer angular order at least two. The discrete entry validates finite nonempty one-dimensional energy/weight/occupation arrays, positive energy/weight, occupations in [0,1], matching shapes, and the same angular-order contract. Eighteen new rejection regressions include the actual four reviewer counterexamples plus related finite/positive controls.

The physics reviewer supplied T=1.3 MeV, n=0.0013 fm⁻³, offset=17 MeV, energies [3.5,5,7.5,11,18] MeV, weights [0.3,1.3,0.8,2.1,4] MeV, occupations [0.91,0.07,0.63,0.21,0.005], and couplings (0.2,0.7). The original column-minus-row summation included large self-transition fluxes that must cancel exactly. Gross transition flux was approximately 0.0397 fm⁻³ s⁻¹ while the redistribution entropy signal was approximately 1.20×10⁻⁸ fm⁻³ s⁻¹. The original entropy identity missed the 2×10⁻¹² relative gate with a 2.17×10⁻¹⁰ error.

The repair accumulates each unordered **off-diagonal** pair directly: add Jji−Jij to bin i and its negative to bin j; diagonal self-transitions contribute exactly zero. It does not modify transition rates, physics, weights or acceptance tolerances. On the counterexample the entropy identity error returns to roundoff. Four additional regressions check the actual entropy identity, energy exchange, number conservation and entropy sign. `validation_results.json` retains both original cancellation replay and repaired numbers. The original 167 checks remain in the suite unchanged; total 189/189 pass. Independent rechecks must confirm the repaired hash before acceptance.
