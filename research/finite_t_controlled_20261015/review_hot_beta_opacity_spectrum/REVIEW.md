# Independent hot beta spectral-opacity review

**ACCEPTED at selected fixed T2 MeV,nB1e−7 fm−3,Ye.3 free-npe beta/inverse-three-body spectral scope, including thermal support above the neutron-rest endpoint.** Current corrected review runs have **216 passing checks and no failed physical gate**. A retained initial reviewer entropy-report FAIL, two reviewer harness errors and a redundant reviewer quadrature-panel warning are explained below; they are not erased or reclassified as passing original records. Counts include repeated moments, hashes and replay checks, not216 independent physical benchmarks. No PC source is joined and no broader opacity inventory is accepted.

The reviewed spectrum SHA256 is `e5b67fd4206bf3f766225e7030864729213b29944deda17fbdb606a9630957ff`; author manifest is `6f7cc97e9d5820da6ff7d0592691b5f438a7292daf98eb6fe295aea7660c9af8`. All52 pinned author/upstream/review files remain unchanged. In particular original `hot_beta_moments/results.json` remains exactly `57e2e89e820b0e80b9d0cd4956d67c54e208cd78d5e2609edcd8e1ef54f94d88`. The author note/qualification, accepted hot-beta/crossed reviews, source/current and phase contracts were read before probes. No generic driver/main was imported or executed.

## Independent amplitude and conditional phase

My new reference evaluates **Dirac traces** rather than the author's explicit external-spinor sum. Three fresh physical rest events compare with both literal decay and reciprocal inverse spinors before thermal integration. The baryon tensor is Tr[(slash p+mp) Gamma_mu (slash n+mn) adjGamma_nu]; the lepton tensor is Tr[(slash e+me) L_mu slash k adjL_nu]. The factor GF² Vud²/4 includes the initial neutron-spin average2, and V−A selects the single active antineutrino helicity. No further antineutrino spin average is inserted. The same physical-q² vertices/form factors, physical masses and bare weak mass are shared; this is numerical/current-contraction independence within that declared model, not independent many-body physics or measured-rate validation.

The new phase rule uses **parent rapidity eta**, direct flat rest-neutrino/electron Dalitz energies, and the lab-energy delta integrated over the rest-neutrino **orientation**. With p=mn sinh eta and gamma=cosh eta,

```
E_lab = gamma E_star (1+beta cos_theta_star)
dcos_theta_star/2 * delta(E_lab-...) = 1/(2 sinh(eta) E_star)
```

Rest E_star lies between E_lab exp(−eta) and min(E0,E_lab exp eta). The rest electron interval is independently recovered from the original on-shell Dalitz constraint, using stable factored discriminants; daughter lab energies then enter the actual FD factors. The electron azimuth is integrated. Rest electron energy and the author's electron–proton CM cosine are affine-equivalent at fixed rest-neutrino energy, so those particular Gauss rules are not claimed fully independent. Independence comes principally from the trace, rapidity measure, rest-neutrino map and the different delta reduction.

The spin-two parent measure is mn³ sinh²(eta) cosh(eta) deta/[pi²(hbarc)³], multiplied by 1/gamma and the orientation-delta factor above. The rest Dalitz measure is Mavg2² dE_star dEe_star/(64pi³mn hbar). For inverse3→1 the same invariant measure supplies the final-neutron factor, with actual neutron vacancy. Separate products are fn(1−fp)(1−fe) for j and (1−fn)fp fe for chi. No KMS expression manufactures chi. The complete conditional event density is divided by **one** active-helicity DOS E²/[2pi²(hbarc)³] to give s−1 coefficients; reconstructed volume moments multiply that DOS once.

## New energies and moving-parent support

Fresh independent rules (parent,rest-nu,electron,phi,tail)=(8,10,10,8,65T)→(12,16,16,12,85T) compare at **new .31 and .835 MeV**. The latter exceeds the exact physical rest endpoint **.7820077025790402 MeV**.

| Lab energy [MeV] | Independent j [s−1] | Independent true chi [s−1] |
|---|---:|---:|
| .31 | .09700539300716648 | .02780134748373826 |
| .835 | .0004250232758043399 | .0001583743998504818 |

Maximum independent control change is **9.99e−7**, and maximum fresh comparison with the author's distinct cone-delta integral is **7.38e−7** relative, both below the unchanged3e−6 gate. The .31 comparison is about1e−14; the new above-endpoint point is less tightly resolved and is not presented as machine-precision agreement. KMS is checked only after both independent positive branches are computed.

Above E0, the minimum parent momentum is independently solved from the original Doppler condition E0(sqrt(mn²+p²)+p)/mn=E. It matches mn(E²−E0²)/(2EE0). A separate scalar infinite-p FD integral reproduces the reference's allowed-parent density within7.67e−15 relative. That above-endpoint density is an allowed-domain density, not the full EOS neutron density. The real FD parent occupation and relativistic energies are used, with no Maxwell replacement, inserted width or rest-endpoint lab cutoff. A same-rule tail85→100T experiment at .835 leaves both logs unchanged at printed precision; this is selected tail evidence, not an analytic bound.

## Fresh spectral reconstruction and occupied radiation

Every independent incident node is a **new actual trace/FD/phase calculation**. My incident panels differ from the author's: [0,.15,.35,.6,E0,.88,1.,1.15,1.4,1.6] MeV, with6→10 Gauss nodes per panel (54→90 fresh nodes) and the material rules above. The source is neither interpolated nor read from saved author nodes. Coarse/fine collection takes9.50 and33.70s; every complete experiment stays below120s.

For empty, LTE, emission-dominated mu_anti=mu_eq−1.6 MeV and inverse-dominated mu_anti=mu_eq+1.6 MeV radiation, I reconstruct positive F=j(1−f) and B=chi f number/energy moments separately. Fine results agree with the separately accepted hot-beta event totals within **1.39e−9 relative**. The incident/material refinement is within the unchanged3e−6 gate; it is a joint change, not an isolated estimate of every error source. Positive LTE F and B are nonzero. Direct (F−B)log(F/B) entropy is nonnegative and agrees with the signed affinity relation in both directions; near-zero LTE net is scaled by gross F+B, not by a cancelling net. Empty radiation has no finite entropy derivative assigned.

Fine empty moments are **R=4.2587395037795795e−11 fm−3s−1** and **Q=2.0237742396163444e−11 MeVfm−3s−1**. Above E0, independent actual emission contributes **4.4026908034200123e−13** and **3.5381925676597135e−13** in those units: **1.033801386% of number and1.748313867% of energy**. This confirms finite free-particle thermal Doppler support above the rest endpoint; it is not a collision width or an interacting-Urca threshold claim.

The independent1.4→1.6MeV extension contributes about3.79e−51 in number and5.32e−51 in energy at this bath. It supports selected truncation numerically; it does not provide an analytic omitted-tail bound beyond1.6 or extend the author's guarded API beyond1.4. Global event-total agreement supplies an additional independent numerical check. No uniform energy/state domain is certified.

Actual integrated net beta counts conserve baryon, electric charge and combined electron Le under (n,p,e_net,anti)=(−1,+1,+1,+1). Rest-inclusive material loss is the emitted antineutrino energy with no added beta-Q. This review reconstructs the radiation number/energy spectra; it does not newly reconstruct every material species-energy spectrum. That separate material ledger remains the accepted event model's scope.

## Admission, soft representations and reviewer records

Thirty-seven controls cover incident/order/tail/budget/nonfinite/Boolean guards, the original-invariant parent lower edge, the independent FD density and isolated tail, ordinary underflow/overflow representation, false-empty saved-node evidence and source identities. For selected saved-node admission only, I execute the exact literal `contract`/`validate` function ASTs in a private namespace; `total.py` is not imported and its collector/combine main never executes. Genuine saved-node validation and tamper rejection are **evidence/API checks, not fresh physical calculations**.

False-empty j and chi, Boolean indices, changed energy/bath/source/control records are rejected. Finite log−1000/1000 coefficients retain underflow/overflow flags with ordinary value None, rather than exact zero. Soft E1e−8 and1e−100 calls return finite logs, but **their accuracy is unqualified**. E1e−160 explicitly raises `ArithmeticError: Positiveeventphase unresolved`; this soft extension is excluded rather than called empty support. These observations do not qualify arbitrary soft logs or a general numerical rectangle. True absorption is chi and fixed-bath relaxation is j+chi; neutrino occupations enter only the subsequent collision operator.

Three reviewer implementation issues are fully preserved:

- The initial coarse entropy reporter reused the previous scenario's mu in one expected-value calculation, causing one retained emission-entropy FAIL. All fresh spectra and F/B moments were already correct. The reporter alone was fixed; a fresh coarse replay has bitwise identical physical phase records at all54 nodes. `initial_entropy_reporter.py.txt` and `initial_coarse_*` preserve the original failure.
- The literal consumer-AST harness initially omitted `leggauss`, then `EDGES`. Both exact sources/errors are retained. No author code or physical rule changed; the complete required private namespace was supplied.
- My first rapidity rule inserted a mathematically redundant above-endpoint switch. `abs(log(E0/E))` and `log(E/E0)` can differ by an ulp, creating12 microscopic panels (widths about1e−17) and a log-zero warning. The physical switch exists only below E0. The original reference/raw fine results/warning and exact micro-panel diagnosis are retained; the corrected own rule inserts the switch only below E0. Fresh fine selected moments differ by at most the1e−13 replay gate, and the warning disappears. No tolerance was relaxed.

The original author manual-plan endpoint typo and corrected descriptive literal remain untouched. The physical driver used exact S.ENDPOINT before collection; this review does not rewrite that history. All author, upstream, original hot totals and prior review bytes remain unchanged.

Acceptance excludes PC/full-antineutrino joining, complete flavor opacity or material-energy inventory, nonthermal/other-temperature radiation, broad EOS/state domains, mean fields/interactions, Coulomb/radiative corrections, transport/cooling evolution, physical uncertainty and all AD/derivatives. It also excludes relative accuracy of the recorded extreme soft extension. No production, paper, Git or dependency operation occurred.

## Reproduction

With an unfrozen copy of this reviewer leaf beside the pinned source leaves, from Genesis root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_hot_beta_opacity_spectrum/pilot.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_hot_beta_opacity_spectrum/moments.py coarse
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_hot_beta_opacity_spectrum/moments.py fine
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_hot_beta_opacity_spectrum/controls.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_hot_beta_opacity_spectrum/compare.py
```

Current checks are24 pilot+26 coarse+26 fine+37 controls+103 comparison/replay=216. All raw coefficients, positive branches, old failures and hashes are retained. Fresh account/window checks throughout matched account82f4cfd9-9e20-47cf-9997-761328d89f60, reset1792652325,23% used; no reset credit was consumed.
