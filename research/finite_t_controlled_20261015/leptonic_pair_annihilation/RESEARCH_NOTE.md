# Flavor-resolved free e−e+ neutrino-pair reference

**Frozen author prototype; independent review pending.** This adds the named Fermi-limit reaction e−+e+↔nu_a+anti-nu_a, separate from nucleon charged-current beta/positron capture. Exact physical electron mass, flavor-dependent couplings, spin normalization, CM kinematics/boosts and actual FD occupations are retained.339 selected author checks pass. No production NMB, source-join or coupled evolution is changed.

## Primary convention and an explicit source conflict

The [Misiaszek–Odrzywolek–Kutschera primary paper](https://arxiv.org/html/astro-ph/0511555), Eqs6–11, specifies the Fermi amplitude, electron-flavor CC+NC interference, mu/tau NC couplings and initial spin average4. Its independently separable massive FD formulas20–25 supply an additional numerical normalization anchor. A conflict in its printed integrated formula17 is preserved below. The [CPC2022 paper](https://cpc.ihep.ac.cn/article/doi/10.1088/1674-1137/ac62ca), Eq16 with nonstandard terms absent, independently states the massless FD total-energy limit used here. Neither paper's fitting functions, new electromagnetic-neutrino terms or older parameter estimates are imported.

Our fixed illustrative tree input is sin²thetaW=.23129, **not a newly inferred precision low-energy mixing angle**. For a=e, CV=.5+2sin²thetaW andCA=.5; for mu/tau, CV=−.5+2sin²thetaW andCA=−.5. Electron neutrinos include the charged-current amplitude before squaring. There is no universal flavor multiplier applied to an electron result. GF=1.1663785e−11MeV−2; **no Vud/CKM factor** occurs. Running, EW/QED uncertainty, finite-W/Z effects, plasma correlations and initial-state Coulomb distortions are omitted.

The amplitude is GF/sqrt2 times [ubar_nu gamma^mu(1−gamma5)v_anti] [vbar_e+ gamma_mu(CV−CA gamma5)u_e−]. Initial charged spins are averaged by4; final neutrino and antineutrino each have one active helicity. The implementation separately computes a fresh Dirac trace, direct external spinors, reciprocal spinors and a scalar invariant expression. All agree at new massive CM points, including near threshold. The reciprocal common-phase-space kernel retains the same charged-spin-average convention; an inverse physical cross section with one active helicity per incoming neutrino instead uses the unaveraged amplitude4times larger. The thermal common measure already restores that factor through its two charged spin2 measures.

The derived exact CM total is

    sigma = GF² s/[12π beta] * [CV²(1+2me²/s)+CA²(1−4me²/s)],
    beta=sqrt(1−4me²/s).

Its angular integral matches the trace/scalar calculation. The threshold limit is sigma*v_M→GF²me²CV²/π; the axial term vanishes at threshold. The massless limit is GF²s(CV²+CA²)/(12π). me=.51099895069MeV; generated particles satisfy exact two-body support.

**Source-conflict diagnostic:** formula17 as rendered in both primary HTML and PDF text does not reproduce angular integration of that same paper's formula11. At electron CM energy.6MeV and the same declared couplings, its sigma*v differs by−12.56% for electron flavor and+87.04% for mu/tau. `vacuum_results.json` preserves the literal calculation; it is not labeled a passing normalization anchor. The derived angular result agrees with the fresh trace and the paper's separate massive FD formulas20–25. We do not calibrate away the disagreement or silently use the conflicting formula. Independent review should assess this conflict explicitly; no external correction/erratum is asserted. PDF screenshot access failed, so the literal PDF check is through extracted equation text.

## Covariant thermal measure and actual matched matter

The selected physical bath is identical in its lepton sector to the accepted charged-current campaign state: T=2MeV,nB=1e−7fm−3,Ye=.3,muons absent. Fresh matched free npe roots give mu_e=.17446333200826397MeV. Electron and positron particle densities are2.034286367843283e−7 and1.7342863678432833e−7fm−3. Their difference,3e−8fm−3, is the neutral proton density. Physical e± particle distributions, not net charge or classical populations, enter the reaction.

For electron momentum fixed along z, positron relative cosineci and outgoing CM neutrino solid angleOmega,

    dR = [p− E− dK−/π²] [p+ E+ dK+ dci/(2π²)]
         * v_M (d sigma/dOmega) dOmega /(hbar hbarc³) * occupations.

Both incoming measures contain spin2. v_M=pi sqrt(s)/(E−E+), d sigma/dOmega=Mavg4² pf/(64π² s pi), and massless-neutrino pf=sqrt(s)/2. Their product is Mavg4²/(128π² E−E+), used without a small-pi division. Initial electrons retain exact massive energies. Neutrino directions are sampled relative to the incoming electron CM axis; all four particles are boosted to the isotropic bath before moments or final blocking are evaluated. There is no stationary electron approximation.

Two radial square-kinetic-energy maps resolve the massive thresholds. Actual FD logarithms are evaluated at every node. hbar=6.582119569e−22MeVs and hbarc=197.3269804MeVfm. The measures and cross section give naturalMeV4; division byhbar hbarc³ givesfm−3s−1. No neutrino DOS is added to an already complete final-state phase integral. R counts **events**, so each flavor's nu count and anti-nu count both increase byR; the total emitted particle count is2R.

## Empty-radiation results and independent limits

With both neutrino occupations exactly empty, physical T2 results are

| Flavor | Event rate,fm−3s−1 | Nu energy,MeVfm−3s−1 | Anti-nu energy,MeVfm−3s−1 |
|---|---:|---:|---:|
|e|1.8942019361363643e−7|1.5577892728725297e−6|1.5553673088914602e−6|
|mu|4.0010658306063156e−8|3.2971386326860854e−7|3.296197101673893e−7|

Tau couplings equal mu couplings exactly in this electron-pair Fermi model; no separately different tau mass/coupling or thermal integration is asserted. Nu/anti energies differ in the charge-asymmetric matter bath. A direct mu_e→−mu_e run exchanges their energy moments, exchanges electron/positron energy moments and leaves the event rate unchanged. This checks flavor-energy assignment rather than forcing equal output energies.

Empty rules radial/incident/outgoing/phi8/4/4/8→12/6/6/8→20/8/8/8 converge by4.67e−11 at most across the five moments. The independently integrated final-angle cross section agrees with total number and summed energy. Isolated tail40→60 passes. Independently adaptive **one-dimensional** massive FD moments inserted into primary formulas20–25 reproduce both actual flavor event rates and summed energies, independent of the CM event angles/radial quadrature. They anchor normalization and physical mass effects without using printed formula17.

For the separate exact massless, mu_e=0 FD diagnostic,

    R = [49π³/64800] GF²(CV²+CA²) T⁸ /(hbar hbarc³),
    Q_nu+Q_anti = [7 zeta(5)/(12π)] GF²(CV²+CA²) T⁹ /(hbar hbarc³).

These follow from the independently angular-integrated kernel and I3=7π⁴T⁴/120,I4=45zeta(5)T⁵/2. Tests atT=1,2 reproduce the absolute coefficient, powers and equal nu/anti energies. This is a normalization limit, not replacing the physicalme calculation by a massless fit.

## Both partner occupations and direct inverse: selected electron flavor

For common material/radiationT, the actual products are

    F = fe− fe+ (1−fnu)(1−fanti),
    B = (1−fe−)(1−fe+) fnu fanti.

Both partner occupations and both electron holes are evaluated directly. B is independently summed, not manufactured from KMS. Stable signed sums use expm1. The chemical identity is log(F/B)=−(mu_nu+mu_anti)/T because mu_e−+mu_e+=0. The code rejects mixed radiation temperatures. Positive finite logs preserve underflow/overflow flags; they are not silently returned as physical zeros. Empty backward occupation is separately represented as an exact absent input.

Selected electron-flavor commonT FD cases use mu_nu=.5MeV andmu_anti=−.5,−2.1,1.1MeV:

| Case | Forward events | Inverse events | Net events | Entropy production |
|---|---:|---:|---:|---:|
|LTE|1.70994595559271e−7|1.70994595559271e−7|roundoff|roundoff|
|Affinity/T=+.8|1.746439899253392e−7|7.847260308245123e−8|9.617138684288732e−8|7.693710947431052e−8|
|Affinity/T=−.8|1.6418028102631143e−7|3.6538993507545104e−7|−2.012096540491387e−7|1.6096772323931298e−7|

Event units arefm−3s−1; entropy unitskBfm−3s−1. Radial/angular16/8→24/12 (phi12) changes separate positive number/energy moments by at most1.76e−11. An isolated FD tail50→70 at identical controls passes. The cutoff includes the inverse chemical sum, not only electron density. These are selected commonT electron-flavor inverse checks; arbitrary spectra, temperatures, degeneracy rectangles or mu/tau finite-radiation coefficients have no new certificate.

## Energy, species and entropy accounting

Each forward pair event removes one electron and one positron and creates one nu_a and one anti-nu_a. Material net charge/Le and radiation flavorLe each remain zero; baryons do not react. This is exact reaction stoichiometry, not an extra independent numerical benchmark. In later pair-equilibrated FD material feedback, net electron charge remains fixed while total pair energy changes withT and the charge-constrained mu_e.

The code separately accumulates initial electron/positron and final nu/anti energy moments. Their sum ledger passes with exact rest-inclusive energies. Material energy source is−(Qnu+Qanti); adding2me separately would double count. Both chemical material changes cancel because mu_e++mu_e−=0. Thus matter entropy derivative is−Qtotal/T, and radiation entropy derivative is[Qtotal−(mu_nu+mu_anti)R]/T. Their sum matches the separately integrated(F−B)log(F/B), is nonnegative in either affinity direction, and cancels at LTE relative to nonzero gross events. These are prescribed FD-reservoir entropy derivatives, not a spectral-grid H theorem or coupled trajectory. Empty radiation has no finite initial radiation entropy derivative claim.

## Records, runtime and remaining gates

The selected count339 is164 vacuum/coupling,66 empty/massless,90 direct-FD/inverse,15 CP/tail and4 massive separated-FD checks. Related angular/moment/control checks contribute; it is not339 independent physical benchmarks. No numerical physics gate failed in this batch. The explicit primary formula17 discrepancy remains an **excluded external-anchor diagnostic**, not a toleranced passing result. Pre-FD source/raw bytes and pre-cleanup FD raw evidence are preserved. Six redundant algebraic stoichiometry assertions were removed from the driver, and the final90 FD checks were freshly rerun without changing physical integrals. The empty source is bitwise replayed at its previous selected controls after the FD extension. All individual experiments are below20s; no install, heavy build or production edit occurred.

Independent physics/numerics review is still required, especially the massive source discrepancy, common inverse normalization and flavor-energy assignments. No rates are combined with the existing two-channel anti-nu_e join: pair production requires both neutrino distributions and has zero netLe, unlike the named nucleon CC reactions. Running/EW/QED corrections, plasma correlations, photon-mediated thermal processes, scattering, oscillations, coupled cooling/transport feedback, hydro and derivative qualification remain open. There is no full neutrino inventory or blanket state-domain accuracy certificate.
