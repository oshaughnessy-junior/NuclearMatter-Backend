# Massive-lepton vacuum charged-current normalization benchmark

15 October 2026. **Author-frozen reference, independent review pending.** This leaf calculates exact relativistic vacuum two-body phase space for `nu_l+n→l−+p` and `antinu_l+p→l++n`, with an explicit point V−A nucleon current. The lepton mass is an immutable channel parameter. It is a normalization/support benchmark before a warm muon response calculation. It supplies no finite-temperature rate, matter blocking, inverse/crossed medium inventory or production replacement.

## Primary sources and model boundary

[Guo et al.2020](https://arxiv.org/html/2006.12051v2), Eq26, separates the vector/axial, weak-magnetism and pseudoscalar vertices. Equations47a–c give an invariant VV,VA,AA decomposition used here as a separately coded check of the matrix trace. Thermal mean fields and their modified four-momenta are absent in this vacuum benchmark. The omitted pseudoscalar/interference contributions cannot be justified by their suppression for electrons.

[Strumia & Vissani2003](https://arxiv.org/html/astro-ph/0302055), Eqs3–6 and11–13, provides the tree-level invariant cross section, current, unequal-mass threshold and CM kinematics. An independent code path implements their Eq6 at **equal nucleon masses and f2=g2=0**, with `f1=gv`, `g1=−ga`; six amplitude comparisons test both neutrino signs and several energies/angles. Their displayed reduced matrix-element normalization differs from our physical amplitude: `Mphysical²=8 GF² Vud² Mreduced²`. This comparison is an external published algebra anchor, not an executed external software comparison or validation of their complete corrected cross sections.

The model uses constant `gv=1`, `ga=1.2723`, `Vud=.974`, `GF=1.1663785e−11MeV^-2`, `hbarc=197.3269804MeVfm`, and bare masses `mn=939.56542052`, `mp=938.27208816`, `mmu=105.6583755`, `me=.51099895069MeV`. These are declared benchmark conventions, consistent with the existing campaign; no lifetime fitting or precision CKM inference is performed.

Write the general vacuum current schematically as

`Gamma^mu=gamma^mu(GV−GA gamma5)+i F2 sigma^(mu nu) q_nu/(2M)−GP gamma5 q^mu/M`.

**Implemented:** constant GV=gv, GA=ga, exact spinors and full relativistic two-body kinematics. **Explicitly zero:** F2 and GP. No q² dependence, electroweak/QED radiative corrections, Coulomb distortion, second-class currents, structure uncertainty or nuclear target effects are included. “Point V−A” describes a current ansatz, not a systematic complete expansion in E/M. Exact recoil kinematics alone does not qualify the missing current terms. No comparison with a full-current physical prediction is claimed.

## Amplitude, spin conventions and dimensions

Metric `g=diag(+1,−1,−1,−1)`, `gamma5=i gamma0 gamma1 gamma2 gamma3`, slash p=`p0 gamma0−p·gamma`. The physical weak amplitude is

`M=GF Vud/sqrt(2) [ubar_l gamma^mu(1−gamma5)u_nu] [ubar_b gamma_mu(gv−ga gamma5)u_a]`.

Average the initial nucleon spin with **1/2**; sum final lepton and nucleon spins. The incident neutrino has one active helicity and receives **no spin average**. Using slash k for a massless two-spin completeness relation does not introduce a second interacting helicity: the V−A projectors kill its inactive component. For antineutrinos, the lepton tensor is transposed, reversing its antisymmetric part; the outgoing antilepton spin sum has slash l−ml. Target/final baryon masses are interchanged for the charged-conjugate nucleon reaction.

`L^(mu nu)=Tr[(slash l+ml) gamma^mu(1−gamma5) slash k bar{gamma^nu(1−gamma5)}]`,
`H^(mu nu)=1/2 Tr[(slash b+mb) Gamma^mu (slash a+ma) barGamma^nu]`,
`average|M|²=GF² Vud²/2 L^(mu nu) H_(mu nu)`.

The gamma-matrix implementation contracts both metric factors explicitly and checks that the result is real and nonnegative. Its independent scalar-product form is, with `X=(k·a)(l·b)`, `Y=(k·b)(l·a)`, `Z=ma mb(k·l)`,

`average|M|²=16GF²Vud²[(gv²+ga²)(X+Y) ±2gvga(X−Y)−(gv²−ga²)Z]`,

where + is neutrino and − antineutrino. This agrees with the vacuum reduction of Guo47a–c. The physical squared amplitude is dimensionless: GF² has MeV^-4 and each invariant product has MeV^4.

## Exact vacuum support and two-body measure

For an initial target at rest and incident energy E>0,

`s=ma²+2ma E`,
`Ethr=max(0,[(mb+ml)²−ma²]/(2ma))`,
`pi=(s−ma²)/(2sqrt(s))`,
`pf=sqrt([s−(mb+ml)²][s−(mb−ml)²])/(2sqrt(s))`.

The code evaluates the first threshold factor from `2ma(E−Ethr)` in endothermic channels, avoiding a large mass-square subtraction close to threshold. Below or exactly at a positive threshold the integrated two-body cross section is zero. Just above threshold it is positive and scales as sqrt(E−Ethr) for these massive final states. The exothermic electron-neutrino/neutron channel instead has threshold zero; its positive E→0 limit must not be treated as equality `s=(mb+ml)²` at E=0.

CM four-vectors conserve energy/momentum at every angle and satisfy all external mass shells. Azimuthal symmetry gives

`d sigma/d cos(thetaCM)=average|M|²/(32pi s) * pf/pi`.

This is MeV^-2. Multiply by `(hbarc)² *10^-26` for cm². There is no additional target spin degeneracy, neutrino DOS or factor c in a cross section. A later stationary dilute-target absorption rate would contain `n_target c sigma`; that is not a finite-T/degenerate matter rate.

Vacuum muon thresholds are **110.161373013MeV** for neutrino/neutron and **113.047310595MeV** for antineutrino/proton. In comparison, the static neutrino estimate `mmu−(mn−mp)=104.36504314MeV` misses target recoil; it cannot determine vacuum two-body support. Neither vacuum threshold is a dense-matter thermal support prescription.

## Forward values and independent limits

Selected point-current results, not full physical uncertainty predictions:

| incident energy [MeV] | sigma(nu_mu+n) [cm²] | sigma(antinu_mu+p) [cm²] |
| --- | ---: | ---: |
| 150 | 1.00984167787319e−39 | 8.11862155917659e−40 |
| 300 | 4.45904278188420e−39 | 3.11233847477011e−39 |

`validate.py` reports **477/477 author checks pass in0.244s**: Clifford algebra, independent invariant/trace amplitudes, four-momentum and mass-shell conditions, threshold support, positivity, angular-order comparison, source-defined Strumia polynomial, electron and muon heavy-target limits and massless-lepton continuity. Maximum trace/invariant or Strumia relative discrepancy is6.57e−16. The mass-shell tests normalize residuals by s; they are float64 consistency tests rather than arbitrary-mass exact-arithmetic certificates.

For common M→infinity, the independently imposed static angular and total limits are

`d sigma/dcos c = GF²Vud²/(2pi) El pl [(gv²+3ga²)+(gv²−ga²)beta_l c]`,
`sigma=GF²Vud²/pi (gv²+3ga²) El pl`,

with `El=E+ma−mb` and beta_l=pl/El. Electron E20MeV and muon E150MeV use M=1e6→1e7MeV; errors decrease by about10, consistent with recoil corrections. At M1e7 the muon total errors are4.38e−5 (nu) and6.34e−5 (antinu), while electron errors are4.47e−6 and7.95e−6. Finite-M corrections are not quadrature errors. Explicit lepton masses make this a trace/phase-space check, not electron-rate rescaling or global mass mutation.

For a point current the angular dependence is low-order polynomial, so Gauss8 and24 agree near roundoff. That does not forecast convergence for q²-dependent form factors, matter response or blocked thermal tails. Algebraic and quadrature errors here are tiny compared with omitted physical contributions; no numerical error bar should be presented as the physical uncertainty of a muon opacity.

## Preserved test-design failures

The first run's sole failed check incorrectly demanded the threshold invariant be zero for exothermic nu_e+n when its physical threshold had been clamped to E=0. Its raw474-check report and driver are preserved as `initial_exothermic_threshold_failure.json` and `initial_validate.py`. The check now tests allowed support there, while retaining equality for positive thresholds.

A second run tested sqrt-threshold asymptotics at deltaE1e−3→1e−5MeV with a3e−4 tolerance. Electron antineutrino production has a smaller mass scale; its finite-distance correction was2.414e−3, so the asymptotic approximation had not reached that tolerance. `initial_threshold_asymptotic_failure.json` and `initial_threshold_validate.py` preserve it. Moving to deltaE1e−5→1e−7 retains the same3e−4 gate and passes. **The physical kernel was unchanged through both test repairs.** No integration floor, smoothing width or fitted normalization was introduced.

## Next physical gates

Independent review should verify spin averaging, antineutrino transposition/sign, invariant normalization and the external Strumia reduction before accepting the leaf. Then add Guo source-defined weak magnetism, pseudoscalar and q²-dependent factors with separate interference comparisons and matched forward external-code points. After that, embed the validated current into a relativistic finite-T response with EOS-consistent dispersions, lepton blocking, inverse/crossed channels and flavor-resolved number/energy moments. A vacuum cross section alone closes none of those warm or dense-matter gates.

Reproduce with the existing `junior_tools` Python: `python outputs/nmb-finite-t-20261015/muon_vacuum/validate.py`. Only this new directory is authored; other contributors' frozen sources remain untouched.
