# Relativistic crossed neutron beta decay and inverse-three-body contract

15 October2026. **66/66 selected author checks pass; independent review required.** This leaf derives the missing crossed semileptonic branch and implements exact **vacuum neutron decay** plus small current/thermal-product probes. It does **not** implement a finite-temperature spectral crossed rate or complete antineutrino opacity. It preserves all other campaign sources unchanged.

## Primary sources and scope

[Guo et al.2020, Sec.III.2, Eq.36 and AppendixC.2](https://arxiv.org/html/2006.12051v2) explicitly treats inverse neutron decay and its crossed capture matrix element. That primary source was checked for this contract. [Ando et al., neutron beta decay in EFT](https://arxiv.org/abs/nucl-th/0402100) treats radiative, recoil and weak-magnetism corrections; it makes clear why a source-defined tree-level calculation is not a measured-lifetime calibration. We retain the previously reviewed Guo full V/A/F2/GP current and its timelike form factors, with physical free baryon masses, bare Mweak=(mn+mp)/2=938.91875434MeV, GF=1.1663785e−11MeV^-2, Vud=.974, ga=1.2723. No Coulomb, radiative, finite-density correlation or nuclear mean-field correction is included. Electron and muon mass replacements alone cannot supply omitted current physics.

`beta.py` reuses our separately authored, frozen explicit-spinor/gamma/vertex primitives from `relativistic_capture_oracle/events.py`, while constructing new crossed lepton/baryon currents. This is honest source reuse, not a claim of a second independent implementation of those primitives. Within this leaf, two independent exact three-body phase-space reductions and a scalar point-V/A invariant test supply numerical forward checks.

## Ordered currents and the physical transfer

For neutron decay write

`n(pn) -> p(pp) + e−(l) + antiνe(k)`, `Q=k+l=pn−pp`, `Omega=Eanti+Ee>0`.

Both leptons have positive physical energy. The forward hadronic n→p transfer is **−Q**, while the ordered inverse reaction has `p+e−+antiνe -> n` and hadronic p→n transfer **+Q**. Define the n→p vertex Gamma(q) with the same source convention as the vacuum capture current. The decay amplitude is

`Mdec = GF Vud/sqrt(2) [ubar_e(l) gamma_mu(1−gamma5) vanti(k)] [ubar_p(pp) Gamma^mu(−Q) u_n(pn)]`.

The inverse amplitude uses its reciprocal factors

`[vbaranti(k) gamma_mu(1−gamma5) u_e(l)] [ubar_n(pn) barGamma^mu(−Q) u_p(pp)]`.

The explicit spin sums agree at selected events. Both full baryon spin sums have2 states; for a vacuum lifetime, average only the initial neutron spin by1/2. No antineutrino spin average is inserted. `amplitude(...,inverse=True)` uses the reciprocal current explicitly, rather than deriving a numerical inverse squared amplitude by KMS.

This is **not** the unchanged antineutrino-capture tensor: the electron in inverse neutron decay is an incoming **u** spinor, whereas positron capture has a **v** charged-lepton spinor. Their completeness signs are slash(l)+me and slash(l)−me respectively. The physical transfer also changes from k−l to k+l. Blindly feeding a positive capture El into an unchanged capture tensor loses that crossing. Guo's four-momentum crossing substitution has its specified particle/current convention and should be checked against these positive-energy external spinors.

## Invariant support and inventory

For free physical particles, Q is future timelike, with `sQ=Q²=me²+2Eanti(Ee−pe cos(theta_eanti)) >= me²`. The ordered inverse on-shell condition `(pp+Q)²=mn²` gives

`pp dot Q=(mn²−mp²−sQ)/2 >= mp sqrt(sQ)`.

Consequently the necessary and sufficient invariant mass range is

`me² <= sQ <= (mn−mp)²`.

The neutron proper-rest electron endpoint is1.2925811676748322MeV; the proper-rest antineutrino endpoint is.7820077025790402MeV. These are **rest-frame decay endpoints**, not hard thermal-bath neutrino-energy cutoffs: a moving neutron Doppler shifts its decay spectrum. A finite-T implementation must resolve the restricted **pair invariant mass and baryon support**, not impose the rest endpoint on bath-frame Eanti.

The same invariant argument forbids **free physical neutron muon beta decay** at every boost and temperature because mmu>mn−mp. A thermal occupation cannot change an individual free-particle invariant mass. It also forbids free physical proton beta+ decay `p->n+e++νe` and its inverse-three-body branch at every boost. An interacting quasiparticle model can change these conclusions only through its explicitly declared dispersions, vector shifts and new support derivation; no blanket vacuum-mass conclusion is imposed on GM1 or other media.

The resulting free one-baryon electron-flavor inventory is:

| radiation | capture pair | crossed pair |
|---|---|---|
| νe | νe+n ↔ p+e− | p ↔ n+e++νe: forbidden in this physical free model |
| antiνe | antiνe+p ↔ n+e+ | n ↔ p+e−+antiνe: allowed |

Free muon-flavor capture can occur above its two-body thresholds, but its analogous one-baryon beta-decay pairs have no support here. Pure muon decay, flavor-exchange leptonic reactions, pair production and nucleon bremsstrahlung remain distinct inventories. This table does not claim all neutrino processes have been implemented.

## Finite-T coefficients and signs: derived, not implemented

Use per-spin particle occupations fp,fn and pair-equilibrated electron occupation fe(Ee;mu_e), with total chemical potentials and the same physical energy zeros in the EOS. For antiν occupation evolution `dfanti/dt=janti(1−fanti)−chianti fanti`, the common invariant measure and fully summed matrix element multiply

`chianti: fp fe (1−fn)` for inverse three-body absorption,

`janti: fn (1−fp)(1−fe)` for neutron decay emission.

Electron blocking is therefore **1−fe for emission**, **fe for absorption**. There is no positron occupation substitution and no extra electron spin2 outside the common full lepton spin sum. The baryon response for ordered p→n uses Q=(Omega,qvec), full hadronic trace, and fp(1−fn); the inverse response uses fn(1−fp). Its natural W units remainMeV², with no GF. After the electron azimuth is integrated, the formal coefficient is

`dchianti_s/(dEe dc)=GF²Vud²/(32pi² hbar) * pe/Eanti * fe * Lcross_mu_nu Wpn^mu_nu(Omega,q)`.

For j replace fe→1−fe and Wpn→the inverse-occupation response with its reciprocal index/current orientation, retaining the same positive microscopic contraction. The angular coordinate is now the angle between k and l in their **sum**: q²=Eanti²+pe²+2Eanti pe c. A tensor implementation must preserve all complex components and reject an unsupported current/transfer domain rather than borrowing a spacelike capture support solver.

The independently evaluated occupation products give

`log(janti/chianti)=−[Eanti−(mu_n−mu_p−mu_e)]/T`.

Thus `mu_antiν,eq=mu_n−mu_p−mu_e=−zeta_e`, with zeta_e=mu_e+mu_p−mu_n. The sign reverses relative to electron-neutrino capture. This is trapped-equilibrium detailed balance, not transparent source balance. The kernel must integrate j and chi independently in log space before using this relation as a check. A finite log must be retained with ordinary representability flags in suppressed tails; no floor, source hole or inferred zero is justified by underflow.

## Exact vacuum phase-space anchors

The vacuum rate is `Gamma=(1/2mn) integral dPi_p dPi_e dPi_anti (2pi)^4 delta4 Mavg²`, with invariant dPi=d³p/[(2pi)³2E]. Two separately ordered reductions are implemented:

1. Factorize with the timelike lepton-pair invariant s and CM electron angle c*. After unpolarized overall4pi orientation and pair azimuth2pi integrations,

   `dGamma/(ds dc*) = pQ pe* Mavg²/[128pi³ mn² sqrt(s)]`.

   Here pQ=sqrt([Qmass²−s][(mn+mp)²−s])/(2mn), Qmass=mn−mp, and pe*=(s−me²)/(2sqrt(s)). A sin² map resolves both physical s endpoints. The electron and antineutrino are boosted from the pair rest frame into the neutron rest frame.

2. Integrate neutron-rest electron energy Ee and electron/antineutrino angle c directly. The original proton-shell delta fixes

   `Eanti=mn(Ee,max−Ee)/(mn−Ee+pe c)`,

   and its exact Jacobian yields

   `dGamma/(dEe dc)=Mavg² pe Eanti/[64pi³ mn(mn−Ee+pe c)]`.

   A square-root electron threshold map resolves pe. This path does not call the pair-invariant phase reduction or its boost construction.

The two complete phase-space integrations agree for total rate and both lepton energy moments at16/32/48 orders with unchanged3e−9 coordinate-agreement and3e−10 refinement gates. Proton energy is integrated as well, closing the exact full rest-energy ledger including proton recoil. Selected full-current result:

| quantity | tree-level free neutron |
|---|---:|
| decay rate |.0010533846209152071s^-1|
| proper lifetime |949.3208654699883s|
| antineutrino energy rate |.0005048801786060273MeV/s per neutron|
| electron total-energy rate |.0008571201859495994MeV/s per neutron|

The corresponding exact point-V/A recoil lifetime is949.3301972469558s. The static allowed expression is GF²Vud²(1+3ga²)/(2pi³hbar) integral_me^Q dEe Ee pe(Q−Ee)², giving947.3710531904898s. Its integral agrees with the older frozen static weak reference to1.34e−15 when constants/Q and negligible electron blocking are matched. A separate mathematical common-bare-mass point-V/A sequence1e4→1e5→1e6MeV at fixed physical Q approaches this static rate under unchanged limits; it is not physical dense matter. Full q²-dependent/GP current and contact allowed current should not be silently equated as M grows.

These tree-level numbers are **not fitted to the measured neutron lifetime**. They omit Coulomb and radiative effects. The small full-current versus point-V/A integrated difference is a diagnostic of the chosen current convention, not a complete uncertainty estimate. Spectra at total Ee=.56,.75,1.0,1.24MeV are saved with angular refinement; no spectral fit was introduced.

## Evidence and remaining implementation gates

`python probes.py` gives44PASS in~.35s; `python check_contract.py` gives21PASS; `python check_static.py` gives1PASS. Additional probes test reciprocal crossed amplitudes, a separate scalar point-V/A invariant, timelike four-momentum closure, baryon and complete anti-flavor KMS occupation products, forbidden free branches and finite/positive/control guards. No failed numerical gate arose in this bounded batch. All references remain unchanged; CHECKPOINT hashes this leaf and its reused primitives.

Next implementation should: (1) construct the full crossed lepton tensor and p→n current at positive timelike Q; (2) implement exact support without sampling across its narrow invariant interval; (3) integrate separate positive logj/logchi at selected matched free-EOS states; (4) test dilute moving-neutron spectra/total emission against Lorentz-time-dilated proper lifetime and Doppler-boosted rest spectra; (5) check direct inverse-three-body phase space and anti-KMS independently; (6) combine with capture only after named inventory/units checks; and (7) qualify energy moments, convergence and EOS feedback before using a complete anti-flavor source. GM1, general medium support, broad spectral tails, rates in slabs/boxes and derivatives remain separate gates. No thermal crossed opacity is claimed from the vacuum lifetime alone.
