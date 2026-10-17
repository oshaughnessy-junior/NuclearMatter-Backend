# Independent review: off-pole external-leg charged weak OPE algebra

**Accept the finite-input external-leg algebra at the selected new points, with TWO helper-input failures preserved. The current is physically incomplete at finite momentum transfer and supplies no modified-Urca rate.** Independent reports contain419 passing checks and2 failing helper-control checks. The original author source reviewed is SHA256 `372ae513793b7b2c359cdf57ed0566faec2af259e33683d4f52c68d882541484`.

The author has been commissioned separately to repair the helper input guards **after** this initial evidence freeze. That repair is not reviewed by this report. Copies of exact original source,199-check author report and checkpoint are in `reviewed_initial_author_*`; `INITIAL_EVIDENCE_PIN.json` pins all initial review evidence. Do not replace these two failures with a later repaired result.

## Independent construction and new forward points

Only the pure `external_legs.py` is imported under a unique explicit module spec; no author driver or validator executes. The independent construction uses the different interleaved basis `(s1,t1,s2,t2)`. Strong OPE entries are reconstructed with explicit two-spinor components and the isospin Fierz identity `Σa τa_31 τa_42=2δ32δ41−δ31δ42`. Weak insertion is performed as a literal action on one nucleon's spin/isospin state, with a sum over its actual allowed intermediate states, rather than multiplying the author's weak or strong matrices.

For each emitting line I independently compute kinetic energies εkin(p)=p²/(2M) and resolvents **by energy differences**:

```
D_in=ε(pi)−omega−ε(pi−Q)
D_out=ε(pf)+omega−ε(pf+Q).
```

The common rest energy cancels. No chemical potential, mass Q offset, imaginary width or Fermi-surface approximation enters. Independent recoil kinematics solve p1+p2=p3+p4+Q and ε1+ε2=ε3+ε4+omega. The two new points are

| Incoming pCM | omega | Outgoing Q | final relative direction |
|---:|---:|---|---|
|117.3|6.7|(−1.7,2.9,.8)|(.24,−.61,.75)|
|83.1|2.8|(1.4,.7,−2.1)|(−.46,.32,.83)|

All momenta/energies are MeV, common M939 MeV. Resolvents have minimum magnitude2.618 MeV, well separated from the declared.05MeV pole exclusion. That exclusion remains an admission gate, not a physical width. Matrix comparisons cover all7 supplied components and both lowering/raising weak charges. Maximum relative forward-amplitude disagreement is1.04×10⁻¹⁴. The many repeated forbidden-block checks contribute to the total count;419 does not mean419 independent kinematic benchmarks.

The incoming one-body action occurs **before** strong scattering in V G J; the outgoing action occurs **after** it in J G V. For line1 the spectator strong transfer is p4−p2; for line2 it is p3−p1. These transfers cannot be identified at finite Q. All four direct terms and the four final-exchanged terms are combined coherently before any spin square. The np charge-exchange pieces of OPE are retained; replacing the strong operator by a spectator number would fail these checks. All fixed-charge blocks forbidden by a single n→p or p→n transition vanish in the natural interaction/resolvent scale, without dividing by an exactly zero reference.

## External states, dimensions and reversal

I explicitly evaluated the four contractions between normalized two-sided antisymmetric states:

```
<AS34|B|AS12> = [B − B_in_swap P − P B_out_swap
                 + P B_both_swap P]/2.
```

The two1/sqrt2 normalizations together give1/2, and the literal particle-permutation identities reduce this to the labelled `B−P B_out_swap` amplitude. All7 components match. This independently confirms the absence of an extra sqrt2 factor. For a specified nn→np channel I sum all initial/final spin states and compare the complete sum, without averaging isospin or inserting an event symmetry factor.

The elementary operator has V∼MeV⁻² and resolvent∼MeV⁻¹; supplied density/axial/convection one-body operators are dimensionless. Therefore A∼MeV⁻³ and its spin sum squared∼MeV⁻⁶. Fresh simultaneous scaling of every momentum, omega, pion mass and common nucleon mass by2.17 recovers that law for all7 components. Convection p/M remains dimensionless under this scaling. A separate reversed-state test uses `(12;34,omega,Q,lowering)` versus `(34;12,−omega,−Q,raising)` and verifies the full amplitude's Hermitian adjoint, including final exchange.

These are **stripped NR amplitudes**. GF Vud/sqrt2, external lepton current, covariant state conversion, phase measures, occupations and event factorials are absent. A future labelled modified-Urca event integral needs its own initial/final identical-particle counting (the stated inventory has one identical pair and factor1/2), separately from the coherent exchange already present. No absolute rate normalization or physical opacity follows from these spin squares.

## Vector continuity defect and the axial soft limit

With outgoing K=(omega,Q), the incoming convection numerator gives `omega rho−Q.j_in=−D_in rho`, while the outgoing one gives `omega rho−Q.j_out=+D_out rho`. Cancelling each resolvent therefore gives

```
omega B0−Q.BV = [R1,V(p4−p2)]+[R2,V(p3−p1)].
```

I reconstruct the RHS with independent Fierz matrices and literal charged actions and antisymmetrize it with the same external-state prescription. It matches the contraction of all four vector components. At the two new finite-Q points its largest absolute entries are respectively **2.02240181689×10⁻⁶ and1.29416735104×10⁻⁶ MeV⁻²**, for both weak charge directions. The nonzero divergence is actual output, not a failed numerical cancellation silently floored to zero.

Thus **this external-leg current is not conserved at finite Q**. The missing interaction current must cancel this defect in the same Hamiltonian/reduction convention. Adding a manufactured longitudinal expression would not determine the missing transverse current or establish a consistent microscopic interaction current.

At the new Q=0,pCM96.4,omega4.1 point, the denominators become−omega,+omega and both strong transfers reduce to the same even OPE operator. I independently construct total charged isospin and the three charged axial generators. The vector commutator with the antisymmetrized strong operator vanishes; each axial commutator is nonzero and reproduces `[J_total,Astrong]/omega`. This is a long-wavelength isospin-symmetric algebra limit. It does not prove finite-q vector suppression, PCAC, an axial continuity identity or a complete weak current.

## Primary-source and physical scope audit

[Shternin, Baldo & Haensel Eqs3–5 and Fig1](https://arxiv.org/pdf/1807.06569) support a leading NR V−A insertion and the before/after resolvent ordering. They use a medium-dependent scattering matrix/propagator and a specified degenerate treatment; those inputs are not reproduced by this free common-mass Born calculation. Their numerical weak and pion conventions must be rematched before a normalization comparison. The convection term here is a declared free-NR continuity test beyond their displayed leading density/axial vertex, not evidence that a full recoil expansion has been implemented.

[Ruiz Simo et al. Eqs12–31](https://arxiv.org/pdf/1604.08423) explicitly separate seagull, pion-in-flight, pion-pole and Delta two-body currents. Their relativistic current prescription cannot be inserted wholesale into a static potential/resolvent scheme without matching reductions, charge flow and regulators. [Dmitrašinović & Sato](https://arxiv.org/abs/nucl-th/9806103) require associated axial and pion production/absorption operators for nuclear PCAC; an external-leg axial commutator is not that construction. These primary sources substantiate missing consistency work, not completed coefficients in this leaf.

The strong operator is the separately reviewed f1.01,mπ140 MeV common-M939 unregulated model, not an empirical NN amplitude. The SOM23 printed transfer ambiguity remains unresolved; inheriting a separately sourced static OPE convention does not certify that paper's full two-loop diagrams or normalization. No finite-T material root, GM1 quasiparticle spectrum, weak lepton contraction, on-shell-pole treatment, physical width, detailed balance integral, thermal modified-Urca rate or NWA/DU subtraction is supplied here.

## Concrete failed helper controls and evidence preservation

The main `direct/amplitude` path rejects nonfinite weak couplings and malformed raising flags, but the public denominator-free helper `vector_defect_direct` does not apply the same guards:

- `gv=NaN` returns a NaN matrix instead of rejection.
- `raising=NaN` is treated as true by `rho` and returns a finite raising-current matrix instead of rejection.

Both failures use the new valid off-pole point and are saved in `extra_results.json`. They do not invalidate the finite-input matrix comparisons or measured Ward defect, but helper admission/finite-output qualification is **not accepted at the reviewed source**. No author code was repaired during this review. The parent commissioned a separate narrow helper repair after I copied/pinned the exact original source/report/checkpoint; its recheck must be a new result.

All original author artifacts and original complex-input failed regression remained hash-identical throughout the probes. The subsequent authorized author repair is intentionally distinguished from that frozen before/after interval. `probe.py` supplies389 passing physical/algebra checks; `extra_probes.py` adds15 passing reversal/dimension/source checks plus2 failed helper controls; `normalization.py` adds15 passing Slater/spin-sum/source checks. The last two scripts import only the local review probe, so they replay/write their own report; no upstream driver is involved. All numerical batches finish in less than0.2s.

This initial review accepts only the declared finite-input external-leg research rung. Interaction currents, physical strong-model uncertainty and absolute modified-Urca observables remain open even after fixing helper input guards.
