# Independent review: relativistic crossed vacuum contract

**Accept the declared vacuum/current/normalization contract and selected numerical probes. 62 independent checks pass; no author files were changed. This does not qualify a finite-temperature crossed response or opacity.**

Reviewed `relativistic_crossed_contract/beta.py` SHA256 `15fd96d5cbf17adf71f39563a15f2746b03bfecae0cf5ba93bc95c0e10a56971` and all frozen author reports, including the original 66 author checks. Raw `results.json` records before/after hashes for every author file and the shared event and static-rate sources. No author validator was run. The new review numerical experiment took 0.0731 s; no failed new gate or interrupted experiment occurred.

## Independent amplitude and spin normalization

The review constructs its own gamma matrices in the **Weyl representation**, independently of the author's Dirac-basis spinors and vertices. With metric (+---), the direct trace uses

\[
L^{\mu\nu}=\operatorname{tr}[(\slashed l+m_e)\gamma^\mu(1-\gamma_5)\slashed k\overline{\gamma^\nu(1-\gamma_5)}],
\]
\[
H^{\mu\nu}=\operatorname{tr}[(\slashed p_p+m_p)\gamma^\mu(1-g_A\gamma_5)(\slashed p_n+m_n)\overline{\gamma^\nu(1-g_A\gamma_5)}].
\]

The normalization is \(\overline{|M|^2}=G_F^2V_{ud}^2 L^{\mu\nu}H_{\mu\nu}/4\): coupling factor 1/2 times **initial-neutron spin average 1/2**. The neutrino spin sum contains only one weakly active massless state after the chiral projectors; there is **no additional antineutrino average**. A final-electron spin multiplicity must not be applied again.

The independent scalar reduction is

\[
16G_F^2V_{ud}^2\{(1+g_A)^2(p_n\cdot k)(p_p\cdot l)
 +(1-g_A)^2(p_n\cdot l)(p_p\cdot k)
 +(g_A^2-1)m_nm_p(l\cdot k)\}.
\]

At fresh total electron energies 0.543, 0.702, 0.973, 1.271 MeV, independent mass-shell events, scalar reduction, Weyl traces and the author's crossed external-spinor point-V/A amplitudes agree within 4.45e-16 relative. These checks directly test the incoming/outgoing `u_e`/`v_anti` assignment and the factor of two; they do not obtain a fitted normalization. The full-current reciprocal amplitude and independent common boosts agree at the same points within 6.67e-16. Full-current checks share the already reviewed upstream vertex implementation; they are not a fresh independent derivation of every weak-magnetism/pseudoscalar term.

The physical channel and crossing prescription agree with [Guo et al., Sec. III.2, Eq. (36)](https://arxiv.org/html/2006.12051v2): crossing changes the charged-lepton momentum, and inverse neutron decay contains two incoming fermion occupations. The author uses the reciprocal current for the true inverse process rather than relabeling a positron-capture tensor.

## Independent absolute phase-space measure

Starting from invariant three-body phase space, integrate the proton momentum with the spatial delta function. In the neutron rest frame let \(E=E_e\), \(v=E_{\bar\nu}\), \(p=\sqrt{E^2-m_e^2}\), and \(c=\hat l\cdot\hat k\). The remaining energy delta has derivative \(|\partial F/\partial v|=(m_n-E+pc)/E_p\). After the global 4π orientation and relative 2π azimuth,

\[
\frac{d\Gamma}{dE\,dc}=\frac{\overline{|M|^2}pv}{64\pi^3m_n(m_n-E+pc)},\qquad
v=\frac{m_n(E_{\max}-E)}{m_n-E+pc}.
\]

Consequently \(|dc/dv|=(m_n-E+pc)/(pv)\) gives the independently integrated **constant Dalitz measure**

\[
\boxed{\frac{d\Gamma}{dE\,dv}=\frac{\overline{|M|^2}}{64\pi^3m_n}},\quad
v_-\le v\le v_+,\quad
v_\pm=\frac{m_n(E_{\max}-E)}{m_n-E\mp p}.
\]

This agrees with [PDG Kinematics, Eqs. (49.19)–(49.22)](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-kinematics.pdf) after angle integration. The review integrates E and v adaptively, using the independent invariant amplitude and an endpoint-regularizing E map. It does not use the author's pair-angle or electron-angle phase integrator for its reference.

| Quantity | Independent point-V/A reference | Relative author difference |
|---|---:|---:|
| Decay rate | 0.001053374266304405 s^-1 | −5.625e-12 |
| Antineutrino energy rate | 0.0005049905751787911 MeV/s | −5.085e-12 |
| Electron total-energy rate | 0.0008569964013256237 MeV/s | −5.922e-12 |

Fresh differential electron spectra at E=0.589, 0.863, 1.157 MeV agree within 1.214e-11. A separate constant-amplitude comparison of direct Dalitz area and recursively factorized two-body phase space passes for four mass choices, including physical neutron decay. The fully massless analytic value \(\Gamma=M/(512\pi^3)\) for unit squared amplitude is recovered to 2.23e-16. Thus the absolute measure is anchored independently of the weak amplitude.

## Lifetime and static limit

An independent analytic dimensionless Fermi integral

\[
f(W)=\frac{\sqrt{W^2-1}(2W^4-9W^2-8)+15W\log(W+\sqrt{W^2-1})}{60}
\]

with W=(mn−mp)/me gives the matched allowed static lifetime **947.3710531904904 s**, agreeing with the author's static integral to 6.67e-16 relative. The independent exact-recoil point-V/A lifetime is **949.330197241613 s**, compared with author 949.330197246953 s. The author's full-current **949.32086547 s** remains its selected tree-level model result, not a measured lifetime calibration.

[Ando et al., Eq. (5) and discussion of Eqs. (8)–(10)](https://arxiv.org/html/nucl-th/0402100v2) provide the allowed normalization and explicitly distinguish recoil, Coulomb and radiative effects. This review does not add the omitted Coulomb/radiative corrections or estimate their uncertainty from coordinate agreement. The author's heavy-mass approach to the matched static limit is consistent with the exact point-current and analytic static anchors; the full form-factor current is not silently replaced by a contact model.

## Thermal signs and invariant exclusions

For neutron decay and its inverse, respectively,

\[
j\ \propto f_n(1-f_p)(1-f_e),\qquad
\chi\ \propto f_pf_e(1-f_n).
\]

Actual log-occupation products at the four fresh events and two independent baths (including T=0.017 MeV) verify

\[
\log(j/\chi)=-[E_{\bar\nu}-(\mu_n-\mu_p-\mu_e)]/T.
\]

The baryonic ordered response for inverse decay is p→n with positive pair transfer Q=k+l, while its reverse uses the reciprocal current. With W containing unaveraged target spin sums and invariant baryon measures, the external-lepton measure after azimuth integration is \(p_e\,dE_e\,dc/(8\pi^2)\). Multiplying the coupling factor \(G_F^2V_{ud}^2/2\) and incoming factor \(1/(2E_\nu)\) yields the contract's \(G_F^2V_{ud}^2p_e/(32\pi^2 E_\nu)\) prefactor before ħ conversion. This is an algebraic normalization check, not an implemented finite-T kernel test.

A future-directed free final-state sum requires **m_parent ≥ sum(final masses)**. Free neutron→proton+muon+antineutrino and free proton→neutron+positron+neutrino violate that invariant inequality, regardless of the parent's boost or thermal distribution. The author rejects both branches. This exclusion applies to the one-baryon free decay process; collisions with spectators and medium dispersions require a new derivation. A boosted allowed neutron decay must not be clipped at the neutron-rest-frame neutrino endpoint.

## Remaining gates

The selected vacuum contract is accepted. No finite-T crossed kernel, moving thermal neutron decay rate, inverse three-body integral, complete relativistic capture+decay inventory, GM1 crossed implementation, external matched EOS comparison, or physical uncertainty envelope is qualified here. Those gates remain open. Both independent numerical methods can share the same physics omissions, so their agreement is a numerical result rather than a physical error bar.

Reproduce from the repository root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_relativistic_crossed/probe.py
```
