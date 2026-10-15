# Hot physical free npe neutron decay/inverse total moments

**Frozen author prototype; independent review pending.** Actual moving-neutron three-body decay/inverse event-number and separate antineutrino/electron/proton/neutron/direct material-loss moments are calculated for the same pair-rich neutral bath as the positron-capture pilot: T=2 MeV,nB=1e−7fm−3,Ye=.3,muons absent.412 selected author checks pass. No positron-capture total data were read or combined. There is no full antineutrino inventory, transparent equilibrium, coupled evolution or all-neutrino-energy tensor-integral qualification here.

## Exact material match, energy and species

Fresh physical free FD roots give nn=7e−8 andnp=3e−8fm−3; mn=939.56542052,mp=938.27208816,me=.51099895069MeV. Total chemical potentials are mu_n=919.8287578100662,mu_p=916.8449303084382,mu_e=.17446333200826397MeV. Pair electrons have n_e−=2.034286367843283e−7 andn_e+=1.7342863678432833e−7fm−3, so neutrality is n_e−−n_e+=np=3e−8fm−3. Using net electron density as its particle density would miss the actual blocking distribution.

The electron blocking factor uses the actual FD occupationf_e−(Ee;mu_e,T). Its accompanying equilibrium positron population enters thermodynamics and charge neutrality but **positron capture is a separate channel** and is not included in these neutron-decay moments. Muons are absent by declared npe ansatz. The inherited EOS carries antinucleon log diagnostics and rejects its pair-exclusion gate; baryon antiparticles do not contribute to the source or EOS. All baryon and lepton energies and chemical potentials include rest mass. There is no medium shift, effective Q, Coulomb correction, radiative correction or measured lifetime calibration.

## Full current and common covariant event measure

`hot_moments.py` is an explicit local adaptation of frozen `crossed_thermal_moments/moments.py`, SHA55a84c8eb1d6bf78e3f6247c9f814f1cf72471c92a21c09bbd4028df26dba405. The new finite direct-parent momentum rule and new hot qualification do not modify the frozen old leaf or the existing low-temperature integrated spectral API.

The exact rest-event primitive contains n→p+e−+anti-nu_e physical three-body recoil and the full massive V/A/weak-magnetism/pseudoscalar current with physical timelike q²-dependent form factors. Its n→p transfer isq=p_p−p_n, and the reciprocal amplitude is its Dirac adjoint with the reversed current orientation. The bare weak-current mass is(mn+mp)/2; no effective mass is inserted. Current sources/conventions are the frozen [Guo et al.](https://arxiv.org/html/2006.12051v2) implementation and its existing independent vacuum interference checks. The same declared GF,Vud and form-factor parameters are used; numerical agreement is not a current-model uncertainty estimate.

The neutron rest measure dΓ*=spin-averaged|M|² dΦ3/(2mn) is multiplied by

    [2 d³pn/(2π hbarc)³] (mn/En).

This represents the fully summed spin amplitude with per-spin FD occupations in either direction. There is no second proton/electron spin factor or incoming-neutrino average. The Haar rest-event orientation is integrated relative to the moving parent, and all daughter four-vectors are boosted into the bath frame before computing blocking, energies and emission/inverse products. No extra neutrino DOS enters a total event integral, since the complete three-body phase space already contains that neutrino.

Actual positive products are

    F = fn (1−fp) (1−fe−) (1−f_antinu),
    B = (1−fn) fp fe− f_antinu.

Separate positive/log F andB moments are summed directly. Stable local signed differences use expm1. KMS does not manufacture the inverse. Event scenarios are: unblocked final particles and empty radiation as a vacuum diagnostic; actual electron/proton blocking with exactly empty radiation; and commonT FD radiation at mu_antinu=mu_eq ormu_eq±.8T, where mu_eq=mu_n−mu_p−mu_e=2.8093641696198066MeV. Mixed radiation temperature is rejected. Empty radiation hasB=0 exactly; this is not a finite-temperature equilibrium.

## Forward gate before blocked results

A separately adaptive direct-p FD integral gives the actual neutron density and proper-time density. The computed mean mn/En=.9968154832083826. The current model's rest decay rate isGamma*=.001053384620915211s−1, with rest antineutrino/electron/proton energy rates .0005048801786060281/.0008571201859496017/.9883617639549442MeV/s. These are bare tree-level model outputs, not a measured-neutron-lifetime fit.

The unblocked hot number rate is nn<mn/En>Gamma*=7.350210699312923e−11fm−3s−1. An isotropic daughter energy boost contributesgamma, which cancels the proper-time1/gamma, so each thermal daughter energy rate is nn times its rest energy rate. The parent energy moment is nn mn Gamma*. These independent time-dilation/Doppler identities pass before accepting blocked moments; the largest proper-lifetime relative discrepancy among the selected rules is2.52e−9. The neutron FD density forward gate agrees within1.6e−14 relative. No Maxwell-Boltzmann target or static neutron replacement supplies these results.

## Actual blocked total moments

For actual electron/proton blocking and **empty** antineutrino radiation, the selected refined results are

| Moment | Value |
|---|---:|
| Event number |4.2587395065654314e−11fm−3s−1|
| Antineutrino energy |2.0237742420493317e−11MeVfm−3s−1|
| Electron energy |3.5010549837197257e−11MeVfm−3s−1|
| Proton energy |4.0086272791382174e−8MeVfm−3s−1|
| Neutron energy |4.0141521083639844e−8MeVfm−3s−1|

For commonT radiation with mu_antinu=mu_eq−.8T, forward and inverse number rates are1.7435278060521023e−11 and7.834175430029598e−12fm−3s−1. The net rate is9.601102630491467e−12; net antineutrino/electron/proton/neutron energy moments are4.640433726941096e−12,7.816482717951132e−12,9.037245036318882e−9,9.049701952763774e−9MeVfm−3s−1. The independently accumulated direct material energy loss equals the antineutrino energy moment. The commonT inverse-dominated case has net rate−6.418748543757059e−12 and positive entropy production. LTE has nonzero separate forward/inverse event rates≈1.01234829562496e−11 each, with net cancellation at numerical roundoff.

## Numerical controls and new selected hot spectral gate

The initial sequence increases rest/parent/orientation/phi orders16/12/6/8→24/20/10/12→32/28/14/20. Both separateF/B moment branches and every energy moment pass the unchanged3e−6 refinement gate. A distinct electron-energy rest parametrization agrees with the lepton-pair-mass parametrization. Low versus refined orientation checks pass. An independent finite direct-p parent rule, split into kinetic-energy-scale panels, agrees with the generalized-Laguerre parent rule. Finite parent tail50→80 is checked; its chemical center includes both mu_n andmu_p+mu_e+mu_antinu, as required by direct and inverse products at commonT. The largest moment disagreement across alternate phase, parent coordinate, tail and orientation controls is1.40e−9. All experiments are bounded individually; the largest event run takes9.65s.

The frozen `relativistic_crossed_thermal.kernel.spectral` deliberately rejectsT2. This guard remains intact and was tested. It is not called accepted at hotT. `hot_spectrum.py` instead contains a **new local selected-state reference**, restricted to the exact chemical potentials above and Eantinu=.42MeV. Its numerical geometry is derived from the old integration body, with entirely new actual hot-bath forward/refinement checks. The differential full-current kernel itself already has the exact invariant three-body support and does not impose a neutron-rest endpoint on moving particles.

The independent conditional-event spectrum fixes lab antineutrino energy through an isotropic rest-angle delta: dcos(theta*)/2 becomes1/(2gamma v Eantinu*). It integrates a new conditional Dalitz measure and only then divides by the one-helicity DOS Eantinu²/(2π² hbarc³) to obtain j,chi. Refining conditional rest/parent/phi12/16/8→20/24/12 changes coefficients by less than9e−13. New hot-reference angular/lepton/baryon-energy/width controls10/12/16/9→16/20/24/13 change coefficients by5.22e−10. AtEantinu=.42MeV, conditional j=.07083091288795236 andchi=.02144761484832465s−1 agree with that reference within1.21e−10. The final exact-state guard replays those selected coefficients bitwise and rejects changed chemical potentials/incident energy. This is one new selected hot spectral anchor, not an all-Enu tensor moment integration or a general extension of the frozen lowT API.

## Composition, total energy and entropy

Per signed forward beta event: neutron number−1, proton number+1, net material electron number+1, radiation antineutrino number+1. Hence dYe=dYp=R/nB, electric charge is conserved, and material/radiation electron lepton changes+R/−R cancel. Pair electron thermodynamics must update its **net** density in any later coupled evolution. These are instantaneous fixed-bath moments, not an evolved composition trajectory.

Material rest-inclusive energy change is Qp+Qe−Qn=−Qantinu. The direct per-event material loss En−Ep−Ee is independently summed and compared with Eantinu, avoiding reliance only on subtracting large integrated baryon moments. The largest normalized integrated energy-ledger residual in the main sequence is1.73e−12. Adding an extra betaQ would double count the rest-energy difference.

For commonT FD radiation, log(F/B)=(mu_eq−mu_antinu)/T is verified per event. Entropy production is directly integrated from(F−B)log(F/B), not assigned after the fact. Its matter and radiation reservoir decomposition is independently checked:

    T sdot_m = −Qantinu + mu_eq R,
    T sdot_rad = Qantinu − mu_antinu R.

Their sum agrees with the direct product-based entropy and is nonnegative in either affinity direction. For mu_eq−.8T, total production is7.680882104393242e−12kBfm−3s−1; the opposite affinity gives5.1349988350056005e−12. LTE signed net and residual entropy are numerical roundoff, not evidence for a physical finite source. Empty radiation has a singular initial radiation entropy derivative and receives no finite-reservoir entropy claim.

## Limits, preserved records and next gates

412 selected checks pass; repeated scenario/moment controls contribute to that count and are not412 independent benchmarks. No failed physics refinement gate occurred in this hot pilot. One initial syntax-only driver failure is retained with its exact source; no experiment ran before that fix. `CHECKPOINT.json` pins code, raw reports and all imported frozen physics sources.

The exact full-current event model, rest-frame invariant measure, actual FD distributions and physical boosts remain the physical chain. No width fit, phenomenological smoothing, static/classical target shortcut or surrogate replaces it. Shared-model method agreement does not estimate omitted Coulomb/radiative or medium/correlation uncertainty.

Independent review is required before use as accepted evidence. Broader hot/cold/degenerate state domains, arbitrary radiation temperatures, all-Enu tensor integration, coupled thermal/composition feedback and derivatives remain open. Do not combine this with positron-capture total moments until that independently authored channel has its own accepted same-bath normalization, tail and moment checks. The matching affinity identity alone does not qualify a complete antineutrino source or transparent kinetic equilibrium. No hydro, modified Urca, mean fields, muons or production NMB changes are supplied.
