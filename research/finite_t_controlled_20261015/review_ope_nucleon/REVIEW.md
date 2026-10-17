# Independent review: elementary strong static OPE Born reference

**Accept the selected static OPE algebra and convention-matched LO Born normalization.** Independent probes pass173/173 checks (162 main +11 additional limits, less than0.2 s numerical work). The author operator and every author artifact remain byte-for-byte unchanged. This is an interaction-model building block, not empirical NN scattering accuracy, a weak modified-Urca amplitude, a width or a collision rate.

## Independent basis and exchange audit

I imported only the pure `ope.py` function module under a unique name. No author driver runs. I reconstructed its amplitudes in the **different** interleaved basis `(s1,t1,s2,t2)`, using explicit two-component spin entries

```
σ·q = [[qz, qx−i qy], [qx+i qy, −qz]]
Σa τa(t3,t1) τa(t4,t2)
   = 2δ(t3,t2)δ(t4,t1) − δ(t3,t1)δ(t4,t2).
```

No author Pauli matrices, Kronecker products, isospin matrix or analytic spin-average formula enter that reconstruction. For each labelled matrix element, I construct the direct term with q and subtract the literal final-spin/final-isospin-swapped term with k. Four fresh transfer pairs include generic noncoplanar values, q=0 with k≠0, and q=k. An explicit permutation maps this result to the author's spin-pair/isospin-pair ordering. Maximum full-amplitude disagreement is2.61×10⁻¹⁶ in units of C=f²/mπ².

All16 initial/final charge-label combinations are checked at each point. np→np and np→pn retain both allowed charge exchange and coherent interference; forbidden total-charge changes vanish. Their errors are measured against the **natural nonzero interaction coefficient C**, never relative to the zero amplitude. Outgoing and incoming fermion permutations give the minus sign with their correct q/k mappings, and the complete antisymmetrized operator is Hermitian. No negative interference is clipped.

Fresh coupled spin/isospin states independently show isotriplet eigenvalue+1, isosinglet−3, and spin eigenvalues ±q² along a selected momentum. At q=k the amplitude is `(1−PsPt)V`: total-exchange-even states cancel and exchange-odd states receive twice the labelled direct matrix element. This is a Pauli/permutation algebra limit, not an assertion that arbitrary q,k lie on an elastic energy shell. A fresh axis/angle SO3 rotation and explicit spin-1/2 unitary in the interleaved basis verify rotation covariance.

The fixed-charge coefficients agree with d,e=(1,1) for nn/pp, (−1,2) for np→np, and(2,−1) for np→pn. Isospin is **not averaged** in a fixed-charge cross section; only the four incident spin states are averaged. Summing np→np and np→pn over their full outgoing spheres would double-count a relabeling of the same physical final-state set.

## Absolute Born normalization and new energy/angle anchors

For the declared plane-wave normalization, the relative-coordinate first-Born transition rate divided by incident flux yields

```
dσ/dΩ = [2π/(2π)³] [µred p/(p/µred)] |A|²
       = µred² |A|²/(4π²)
       = M² |A|²/(16π²),   µred=M/2.
```

Here the radial δ(E−p′²/(2µred)) gives µred p, and the incident relative speed is p/µred. The four initial spin states are averaged once. There is no incoming identical-pair1/2 for a beam/target cross section. An nn/pp full-sphere final-state integral needs1/2, or one hemisphere with no such factor. An unordered many-body collision event rate has separate initial-pair counting. The LO external-state conversion `Mcov≈4M²A`, with s=4M² at the same order, gives the same prefactor; mixing exact s with a LO amplitude would not improve physical accuracy.

I derived **closed angular integrals**, independently of author angular quadrature. On an elastic CM shell q·k=0, let a=2p², b=mπ² and L=log(1+2a/b). Then

```
I=∫dc q⁴/(q²+b)² = 2+2b/(b+2a)−(2b/a)L
J=∫dc q²k²/[(q²+b)(k²+b)]
 =2−b(2a+b)L/[a(a+b)]
σtotal = M²(ħc)² C²/(8π) [(d²+e²)I+d e J]
```

with final identical factor1/2 only for nn/pp. The interference sign is retained. Fresh fixed p and angles c=−.873,.147,.964 reproduce the absolute unpolarized differential trace; analytic totals agree with independently integrated author differentials within1.67×10⁻¹⁴ relative.

| pCM(MeV) | nn total(fm²) | np total(fm²) |
|---|---:|---:|
|37.4|0.175883662813485|0.531964190402347|
|92.1|2.07625521142911|5.37463832096219|
|165.3|5.05724055128707|11.5109280103731|

pp equals nn in this common-pion-mass, Coulomb-free model. Full-sphere factorial and hemisphere results agree. New p=.35,.7,1.4 MeV limits recover the derivative-coupling σ∝p⁴ scaling within the explicitly measured O(p²/mπ²) corrections. Zero coupling and zero direct momentum give exact zeros. These are absolute **mathematical model** anchors; no empirical phase-shift comparison is claimed.

## Operator, contact and primary-source checks

The selected model parameters f=1.01,mπ140 MeV,M939 MeV are consistent throughout. The declared equal-M on-shell NR pseudoscalar conversion gπNN=2Mf/mπ is checked; it does not identify the dimensionless pion-NN coupling with a pion decay constant.

[Savage Eqs11–15 and20–21](https://archive.int.washington.edu/users/mjs5/Class_560/lec560_3/node2.html) give the derivative pion coupling, static momentum transfer, negative momentum-space potential and Fourier convention. His F=132 MeV convention has coefficient gA²/(2F²), distinct from gA²/(4F92²). His coordinate derivation identifies the contact distribution and separately discusses its absorption into short-distance operators. The author's decision to retain the **entire unregulated momentum operator** is explicit; it does not claim to implement that later subtraction or a fitted contact interaction.

Independently, `(σ1·q)(σ2·q)=q²[σ1·σ2+S12(qhat)]/3`. Averaging three orthogonal momentum directions gives the central part proportional to `q²/(q²+mπ²)=1−mπ²/(q²+mπ²)`. Fresh Q83.6/147.9 MeV projection checks verify that decomposition. Its position-space central term includes `−C(τ1·τ2)(σ1·σ2)δ³(r)/3` plus the Yukawa term. Opposite triplet m=0 versus m=±1 eigenvalues confirm that the anisotropic tensor operator was not replaced by only its central average.

[Friman–Maxwell SecIII Eqs10–12 and SecVI](https://adsabs.harvard.edu/pdf/1979ApJ...232..541F) support the uncorrelated transfer operator, distinguish correlations/short-range terms, and discuss antisymmetrization. Their pion-pole f²=4π×.08 is not exactly this model's1.01². Consequently the elementary operator does not reproduce their full strong-interaction prescription or weak-rate diagrams. The incomplete Eq10 OCR is supplemented by Savage's explicit derivative derivation; no OCR reconstruction is asserted as an independent byte-pinned equation.

[SOM23 Eqs23–24](https://arxiv.org/html/2308.09819v2) use the chosen f/mπ values and isospin convention but retain the displayed pl−pl numerator ambiguity. The current separately declared static OPE operator does **not** resolve or silently correct that source transcription. No claim is made to reproduce SOM's entire spin/isospin tables, two-loop normalization or widths.

## Qualification, preserved failures and open gates

The author's seven old relative-to-zero validator failures, SyntaxError and download failure remain unchanged and excluded from the250 accepted author gates. My initial extra tensor check also had a SyntaxError before imports or numerical execution; its source/message are preserved locally. The corrected natural-C zero tests do not loosen finite-amplitude relative comparisons or change the physical operator.

`probe.py` and `limits.py` provide independent matrix, coupled-basis, trace, angular-integral and limit checks. `limits.py` imports only the local review probe and therefore replays/writes its own local report; it never imports an author driver. Before/after hashes include all author raw evidence, not only `ope.py`; all match. Browser primary references were checked on2026-10-15, with no claim of locally archived paper bytes.

Still absent: weak insertion before/after each strong scattering leg, consistent intermediate propagators, coherent weak diagram/current interference, interaction currents if required, phase-space integration, strong widths, finite-T collision rates and matching to dressed/direct descriptions. Born numerical precision does not quantify strong-interaction uncertainty or prove a current-conservation identity. The next rung remains off-pole, source-labelled weak amplitudes with independent diagram algebra and external-state normalization; no absolute modified-Urca result follows from this review.
