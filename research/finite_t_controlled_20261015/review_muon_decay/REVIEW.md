# Independent muon-decay vacuum contract review

**Accept the selected exact-mass vacuum benchmark and derived inventory/occupation contract. 108 independent checks pass in 0.014 s. No finite-T collision kernel or medium rate is qualified.**

Reviewed `muon_decay_contract/decay.py` SHA256 `a05e3db13d57ba3e7890fc58aa3e6e57342cea6ced95f2f77efd1c9dac1f441a`, the 158 author checks and their reports. Every author file, the preserved initial boost failure and the shared upstream event source have identical before/after hashes in `results.json`. No author validator was executed and no author file changed.

## Independent amplitudes and absolute normalization

The review implements fresh gamma matrices in the Weyl basis. It does not use the author's external spinors, gamma matrices, invariant-amplitude function or phase constructors on its reference side. For mu−(P)→e−(p)+antinue(a)+numu(n), it explicitly contracts

\[
\frac{G_F^2}{4}\operatorname{tr}[\slashed n\gamma^\alpha(1-\gamma_5)
(\slashed P+m_\mu)\overline{\gamma^\beta(1-\gamma_5)}]
\operatorname{tr}[(\slashed p+m_e)\gamma_\alpha(1-\gamma_5)
\slashed a\overline{\gamma_\beta(1-\gamma_5)}].
\]

The factor 1/4 contains 1/2 from the squared weak coupling and 1/2 from averaging the initial muon spin. There is **no CKM Vud**, neutrino-helicity average or identical-final-particle factor. The chiral projectors eliminate the inactive massless states. The independent trace agrees with both the scalar reduction

\[
\overline{|M|^2}=64G_F^2(P\cdot a)(p\cdot n)
\]

and the author's spinor amplitudes within 6.67e-16 relative at four new kinematic points. These include physical masses and hypothetical daughter/parent ratios .37 and .987; the latter are phase-algebra diagnostics, not a medium mass model. Independent mass shells, four-momentum closure and reciprocal currents pass. For a thermal event measure with per-spin material occupations, the common unaveraged spin sum is **2 times this vacuum spin-averaged amplitude**. The author's reciprocal check retaining the same charged-spin averaging is not mistaken for a pre-normalized inverse thermal opacity.

## New one-dimensional phase reduction

A new reference uses the **electron-antineutrino energy** A in the parent rest frame. Set S=m²−2mA and r=me²/m². The invariant amplitude is independent of the electron energy at fixed A:

\[
\overline{|M|^2}=32G_F^2mA(S-m_e^2).
\]

After the global orientation integration, invariant three-body phase space gives dΓ/(dEe dA)=Mavg²/(64π³m ħ). The Ee interval has width A(1−me²/S), so its integral is analytic:

\[
\boxed{\frac{d\Gamma}{dA}=\frac{G_F^2}{2\pi^3\hbar}
A^2\frac{(S-m_e^2)^2}{S}},\qquad
0<A<\frac{m^2-m_e^2}{2m}.
\]

The conditional electron-energy distribution is uniform over that interval, with center (m−A)(S+me²)/(2S). Its variance is the interval width squared divided by12. Thus total rate, all three flavor energy moments, second moments and mixed moments reduce to independent one-dimensional quadratures. This uses neither author phase coordinate, numerical event generator nor amplitude implementation.

For physical me, zero me, and heldout ratios .37 and .987, author pair-invariant results agree with this independent reduction within **6.56e-15** relative for rate and all first energy moments. The analytic total phase factor agrees within **1.06e-14**. The physical independent rate is **457089.0316305327 s^-1**, corresponding to a tree-level lifetime about **2.18775759381666 microseconds**. The physical energy-rate outputs and their units are preserved in the raw report.

In the massless limit, the antinue spectrum is 12x²(1−x), while the numu and electron spectra are 2x²(3−2x), with x=2E/m. Independent integration reproduces first moments (.30,.35,.35)m, antinue second moment .10m², electron/numu second moments (2/15)m² and mixed <Eanti Ee>=.10m². The flavors cannot be assigned a common emitted spectrum.

## Primary-source and coupling convention check

[PDG Electroweak Model, Sec.10.3.2, Eqs.10.22–10.24](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-standard-model.pdf) explicitly separates the exact electron-mass phase factor from QED terms in the muon-lifetime relation used to extract GF. Its quoted GF=1.1663785e−5 GeV^-2 matches the declared input after unit conversion. Reusing that input in a calculation with the QED factor omitted gives a **tree-level convention**, not a new extraction or a measured-lifetime fit.

[PDG Muon Decay Parameters, Eqs.57.1–57.4](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-muon-decay-params.pdf) supports the endpoint and unpolarized V−A Michel distribution, including the stated neglect of radiative corrections. Neither those distributions nor the numerical method comparisons supply a missing-radiative-effect uncertainty estimate. No bare-theory/measured-lifetime calibration is accepted or required here.

## Flavor, energy and thermal contracts

The review independently assigns electric charge, Le and Lmu to every species, then reconstructs all16 directed reaction rows from the saved author inventory without executing its code. All three charges are conserved. Supported free decay and the three2↔2 crossing classes have the stated flavor assignments. Other1→3 crossings must independently pass the parent-mass≥sum(final masses) condition; the table is not evidence that a light electron or neutrino can spontaneously produce a free muon.

For one mu− decay, material ΔNe=+1 and ΔNmu=−1 leave ΔYp=Δ(Ye+Ymu)=0. Radiation carries antinue Le=−1 and numu Lmu=+1. Charge-conjugate decay reverses both flavor ledgers. Independent event energies verify that thermalized matter loses Ea+En; Ee includes its physical rest energy. In a matched EOS, composition work uses (u_Ye−u_Ymu) times the event rate per baryon. A separate manually added muon-rest Q would double count the EOS composition contribution.

Actual separate log occupation products at four fresh events verify

\[
F=f_\mu(1-f_e)(1-f_a)(1-f_n),\quad
B=(1-f_\mu)f_ef_af_n,\quad
\log(F/B)=(\mu_\mu-\mu_e-\mu_a-\mu_n)/T.
\]

Singling out numu gives affinity mumu−mue−muanti, and singling out antinue gives mumu−mue−munumu. The partner-neutrino occupation remains in each coefficient. These algebraic checks do not validate a coupled finite-T collision integral or imply a universal affinity for a nonthermal partner.

## Scope and failure preservation

The original generic pair boost's near-lightlike closure failure remains unchanged in `initial_general_boost*`; the final specialized longitudinal boost is consistent with the independent energy-coordinate reference. No new failed review gate arose. Agreement of numerical methods is not a physical uncertainty estimate.

Moving thermal parents, actual inverse phase-space rates, partner blocking, two-flavor spectral conservation/entropy evolution,2↔2 crossed kernels, polarization and QED corrections remain open. The accepted result is a vacuum benchmark plus a checked contract for subsequent finite-T work.

Reproduce from the repository root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_muon_decay/probe.py
```
