# Minimally gauged static OPE charged vector interaction current

**Authored research freeze pending different independent review:123/123 selected checks PASS.** This adds a source-labelled seagull plus pion-in-flight vector current to the accepted off-pole external-leg algebra. It cancels that model's measured finite-Q divergence. It does **not** complete the physical transverse or axial current, implement PCAC, give an absolute modified-Urca amplitude/rate, supply a width or qualify interacting finite-T matter.

## Declared reduction and primary anchors

Use the same common-M939 MeV, zero-species-potential, static unregulated OPE model as the frozen external-leg reference: fπNN1.01,mπ140 MeV and C=f²/mπ². Spin/isospin matrices, labelled plane-wave states and unit n→p weak charge are unchanged. All nucleon energies in the external resolvents remain M+p²/(2M). No strong/weak source is modified.

The independently derived prescription is **minimal gauging of an instantaneous auxiliary-pion Hamiltonian**, with no pion retardation or extra two-body charge at this static order. It is not a wholesale insertion of relativistic nuclear currents into a different NR potential model. [Savage Eqs11–15](https://archive.int.washington.edu/users/mjs5/Class_560/lec560_3/node2.html) fix the derivative pion coupling and potential sign/Fourier convention. [Ruiz Simo et al. Eqs13–27](https://arxiv.org/pdf/1604.08423) separately identify seagull and pion-in-flight vector operators. The following Hamiltonian variation, momentum routing, normalization and tests are local derivations under the stated static reduction.

Let S_a^i=N†σ_i τ_a N, g=f/mπ, and let an external isovector source v_i^b couple to the one-nucleon generator t_b=τ_b/2. Introduce

```
D_i N=(∂i−i v_i^b t_b)N
(D_i π)_a=∂i π_a+epsilon_abc v_i^b π_c
Hπ[v]=∫d³r {[(D_iπ)_a² + mπ² π_a²]/2 + g S_a^i(D_iπ)_a}.
```

The nucleon Hamiltonian is `(p−v^b t_b)²/(2M)+v0^b t_b`. Its conventions give physical J_i=−δH/δv_i and rho=δH/δv0. At v=0, minimize Hπ over its nondynamical pion field:

```
(−∇²+mπ²)π_a=g ∂i S_a^i.
```

Eliminating π yields `V(k)=−g² (σ1.k)(σ2.k) τ1.τ2/(k²+mπ²)`, exactly the accepted full momentum OPE operator. No central/contact subtraction or regulator is introduced. The stationarity/envelope identity permits varying the Hamiltonian at its stationary π without an additional δπ term. The two vector pieces are then

```
J_sea,i^b = −g epsilon_abc S_a^i π_c
J_flight,i^b = −epsilon_abc (∂iπ_a)π_c,
```

including both source assignments1↔2. The first follows from gauging the derivative pion-NN vertex; the second from gauging pion gradient energy. They are not inferred by solving one continuity equation for a longitudinal field.

## Explicit charged momentum operators

Set k1=p3−p1,k2=p4−p2. Weak four-momentum is outgoing, so Q=−(k1+k2), with the inherited free-NR energy conservation. A labelled source matrix element carries e^(−ik_i.r), hence its induced pion field is `π_i=−i g (σ_i.k_i)τ_i/D_i`, D_i=k_i²+mπ². Define Ai=σi.ki and

```
R1=tau1_minus (or plus for raising)
B=[R1,tau1.tau2]
  =−i(tau1×tau2)^− for lowering,
(tau1×tau2)^−=(tau1×tau2)x−i(tau1×tau2)y.
```

The literal matrix commutator is used for both charge directions, avoiding a guessed spectator/isospin multiplier. The spatial matrices are

```
J_sea = C gV B [sigma1 A2/D2 − sigma2 A1/D1]
J_flight = −C gV B (k1−k2) A1 A2/(D1 D2)
J_int = J_sea+J_flight,       rho_int=0 at this static order.
```

Spin and isospin factors commute; matrix products above mean their tensor product in `(s1,s2)⊗(t1,t2)`. For lowering these are respectively `−i C(tau1×tau2)^−` times the seagull spin bracket and `+i C(tau1×tau2)^−` times the flight bracket. Each has units MeV⁻³, matching the stripped external-leg current. GF,Vud,lepton current, covariant state conversions and event phase/factorials remain absent.

Final exchange is **coherent**: J_A(12;34)=J(12;34)−P J(12;43), with the same full spin/isospin permutation as the external amplitude. There is no sqrt2 or spin/isospin average added. Both pieces obey the appropriate incoming/outgoing fermion exchange and charged Hermitian-adjoint reversal.

### Charged normalization caution

The one-body generator has unit n→p matrix element. For Cartesian t_a=τ_a/2, the charged operator is t_x−i t_y=τ_minus, so the corresponding current is J_x−iJ_y, **without another1/2**. Ruiz Simo's `(I_V)^±=i[(tau1×tau2)x±i(tau1×tau2)y]` convention is compatible after the stated signs/reduction. [Van Cuyck et al. Eq4](https://arxiv.org/pdf/1702.06402) instead defines a half-normalized charged isovector combination. Its charged one-body/prefactor conventions must be rematched before copying a coefficient. No claim is made that a source convention is wrong or that an arbitrary half-factor can be selected by fitting divergence. Here normalization is fixed by the explicitly gauged t_a and the inherited unit weak charge.

## Why the inherited defect cancels

For q_in=k1+k2, direct contraction of the two gauged terms gives

```
q_in.J_sea = C gV B {A1A2(1/D2−1/D1)
                     +A2²/D2−A1²/D1}
q_in.J_flight = −C gV B A1A2(D1−D2)/(D1D2).
```

The mixed products cancel because D1−D2=k1²−k2². Thus

```
Q_out.J_int = C gV B [A1²/D1−A2²/D2]
            = gV{[R1,V(k2)]+[R2,V(k1)]}
            = Delta_external.
```

The accepted external legs satisfy `omega A_ext^0−Q.A_ext=Delta_external`. With rho_int=0, their sum therefore has zero vector divergence **within this specified static Hamiltonian model**. The identity holds before and after the identical external antisymmetrization. Neither piece alone cancels the finite defect. No Q_i Delta/Q² expression, normalization fitted to a defect or conditional Ward-zero operation appears in the implementation. In particular, the formulas are finite at Q=0 without a separate longitudinal limit.

## Independent algebra and primary Dirac limit checks

`check_current.py` compares the compact commutator current with an explicit epsilon_abc variation of the two original Hamiltonian terms and separately calculated pion-source fields. That construction uses its own Pauli/isospin factors and never calls the defect to set a coefficient. Both terms are checked for lowering and raising at three finite-Q points, including a new (pCM128.7,omega7.3,Q=(2.7,.4,−1.9)) MeV point alongside the two independently reviewed external-leg points.

| omega,Q | Original external defect max (MeV⁻²) | Combined divergence max (MeV⁻²) |
|---|---:|---:|
|6.7,(−1.7,2.9,.8)|2.02240181689e−6|3.5364e−20|
|2.8,(1.4,.7,−2.1)|1.29416735104e−6|1.0588e−20|
|7.3,(2.7,.4,−1.9)|1.93328632261e−6|1.7062e−20|

Maximum scaled combined residual is1.36×10⁻¹⁴. Tests retain separately nonzero seagull, flight and transverse norms, and show that deleting either term leaves a finite mismatch. This verifies a sourced transverse component exists; it does **not** establish that the physical transverse current has been completely determined. Incoming/outgoing exchange, adjoint reversal, Q=0 finiteness, dimensional scaling, finite-real coupling and malformed-momentum rejection also pass. The107 checks complete in0.031s; related permutations/charge checks are not107 independent physical benchmarks.

`relativistic_limit.py` supplies a second source-sensitive test: independently defined Dirac matrices and ubar u=1 spinors evaluate the **spatial vector** bilinears of Ruiz Simo Eqs17,21,26 with physical on-shell free common-M energies, unregulated pion vertices and F1V=1. It does not call the compact/static/gauged-vertex evaluator. The spatial NR limits are `VπNN→−sigma.k/D` and `ubar gamma5 gamma^i u→−sigma_i`; these give the same two static signs, coefficient and charged operator without fitting a scale.

All external momenta are successively scaled1,.5,.25,.125 with fixed M,mπ. For each charge direction, relative seagull errors versus the static expression are .004238,.001028,.0002540,.00006329; flight errors are .002269,.0005980,.0001543,.00003897. The decreasing O(p²/M²) behavior passes16 limit checks in0.009s. This is a controlled leading-static normalization/shape check. It is not a finite-momentum reproduction of the entire relativistic Simo model, its chosen external-state conventions, form factors or response normalization. No empirical nuclear benchmark or physical uncertainty bound follows from numerical agreement.

## Reduction limits and remaining physics

The static auxiliary-pion theory sets pion-energy dependence and a two-body charge to zero at this declared order. It does not reproduce retardation, relativistic two-body charges or a systematically complete recoil expansion. Combining the reduced current with a retarded/regulated potential, mean-field self-energy, Dirac/Landau-mass prescription or physical neutron/proton mass splitting requires a fresh matched derivation. The unregulated operator retains the contact-distribution contribution inherited from its derivative potential.

Van Cuyck et al. Eqs1–6 explicitly show the low-energy seagull and flight structures and a nontrivial consistent regulator treatment; their cutoff and form factors are not copied here. Their coupling set also differs. Sources motivate and cross-check the model operator; they do not substitute for its explicit momentum/current/state normalization derivation.

The minimally gauged pion Hamiltonian chooses a **particular** transverse contribution. Additional separately gauge-invariant short-range, magnetic, Delta or higher-order operators can change it without changing this continuity identity. No unique physical transverse completion or complete model uncertainty is claimed. Axial pion-pole, two-body axial charge/current, pion production/absorption and PCAC are not implemented. Neither a vector Ward check nor an external axial commutator proves them.

No thermal occupations, five-body shell integral, absolute weak matrix element, modified-Urca emissivity/opacity, collision width or DU/MU/NWA matching is added. External on-shell poles still reject under the inherited exclusion rather than being regularized. This advances only the static vector-continuity consistency gate for a controlled two-nucleon ansatz; it does not close the broader finite-T interacting-rate gates.

## Reproduction and provenance

Run `python check_current.py` and `python relativistic_limit.py` in fresh processes with the campaign NumPy environment. Only pinned function modules are loaded; no upstream driver executes. `current.py` checks the exact accepted external source d8400295… and strong source a4d25710… hashes before import and every direct evaluation. Source controls, separate term norms, all selected external momenta and numerical residuals are in the raw reports. All123 selected gates pass; no failed physical probe is concealed. Invalid-input rejections are expected controls, not physics failures. No upstream source, production or dependency file is edited. Independent review remains required before admitting this new research slice.
