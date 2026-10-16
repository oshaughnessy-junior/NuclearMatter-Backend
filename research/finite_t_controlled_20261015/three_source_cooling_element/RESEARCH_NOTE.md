# Short fixed-density cooling with three CC sources and trapped LTE photons

**Author-frozen research prototype; different independent review pending.** A homogeneous material element evolves rest-inclusive energy and composition through actual free neutron-beta, positron-capture and electron-capture phase integrals. It starts atT2MeV,Ye.3,nB1e−7fm^-3 with exactly empty incoming nu_e and anti-nu_e. Baryon density/volume are prescribed constant. Photons are trapped and in LTE with the equilibrium e± bath; there is **no external thermostat**, photon escape, hydro or photon transport. This is a short controlled thermalization ansatz, not a complete stellar cooling calculation or neutrino inventory.

## Physical state and EM closure

Material is the existing physical free n,p and charge-neutral e−/e+ EOS, plus a two-polarization ideal blackbody photon component at the SAME local temperature:

```
epsilon_gamma = pi² T⁴/[15(hbarc)³]
P_gamma=epsilon_gamma/3
s_gamma=4epsilon_gamma/(3T)
Cv_gamma=4epsilon_gamma/T.
```

The photon chemical potential is zero. Rapid electromagnetic thermalization, pair creation/annihilation and photon-number relaxation are assumed; their kinetic times are not calculated. This EM pair equilibrium is distinct from neutrino-pair annihilation, whose weak emissivity is excluded from the present three-channel inventory. Lepton pair number and photon number are not separately conserved. Fixed net charge n_e−−n_e+=nB Ye is enforced by fresh FD roots, rather than treating net charge as an incoming electron or positron population.

An independent Bose/Planck integral checks the two-polarization normalization before any weak trajectory. Full EOS finite differences check u_T and u_Y, photon Cv and the rest-inclusive Euler identity. All11 preflight gates pass. Initial u=979.828434137160MeV/baryon, u_T=76.6725216520573MeV/baryon/MeV and u_Y=−.767335768914664MeV/baryon. Photons contribute13.7015297674561MeV/baryon to u and27.4030595349122 to u_T; the photon-independent composition derivative is unchanged.

All matter chemical potentials and species energies include rest mass. Physical mn939.56542052,mp938.27208816,me.51099895069MeV and bare weak-current average938.91875434MeV remain unchanged. Actual full V/A/weak-magnetism/pseudoscalar form factors are the previously reviewed source convention from [Guo et al.](https://arxiv.org/html/2006.12051v2). No Dirac/Landau mass insertion, source fit, stationary-target replacement or measured-lifetime calibration is introduced by the photon EOS choice.

## Conserved-variable evolution and actual source calls

Evolve u=epsilon_total/nB and Ye with

```
u_dot=−(Q_beta+Q_PC+Q_EC)/nB
Ye_dot=(R_beta+R_PC−R_EC)/nB.
```

The Q are separate positive escaping-neutrino energy moments in MeVfm^-3s^-1; R are directed event numbers in fm^-3s^-1. Each step and Heun predictor inverts the actual photon+FD EOS for(T,Ye), using analytic u_T only for a Newton step and a2e−10MeV/baryon energy-residual stopping gate. Every resulting weak-source evaluation is a NEW actual phase calculation at those fresh chemical potentials. Saved fixed-bath totals, interpolated sources, opacity fits or state-rate caches never supply feedback.

Beta uses exact moving-neutron three-body events and final p/e blocking; PC uses the accepted invariant-amplitude, neutron-tangent/positron-log-momentum event reference; EC uses exact moving-proton/angle-dependent incident-electron thresholds and full CM events. The source profiles are the same independently accepted functions and current conventions as `transparent_capture_balance/` and its254-check review. That acceptance is pinned, while changing temperature and evolving photons/composition are new claims requiring this leaf's own evidence and independent review. Source densities, separate positive moments and event energy ledgers are checked before use. Emitted neutrino DOS is already contained in each complete event integral; no second DOS is added.

The fresh-process client launches the owned worker with `python -I -B`. It pins23 source/acceptance/interface files and requires the exact13 loaded campaign-Python paths before the EOS or source calculation. All frozen source bytes are checked before/after calls. Parent numerical packets check requested state, source controls, density/charge roots, photon-energy/Cv accounting, actual channel sums, derivative signs and species/lepton sources. The client imports only its owned stdlib protocol; no frozen driver executes in the parent or child. This is selected research isolation/admission, not a general untrusted-code sandbox or broad API certificate.

## Completed short trajectories and time error

The predeclared duration is.002s. EOS and source forward gates pass for all trajectories. Fifty main checks cover discrete energy/Le/EM-pair ledgers, event stoichiometry and time refinement.

| Method / steps | Final T [MeV] | Final Ye | Escaped energy [MeVfm^-3] | Runtime [s] |
|---|---:|---:|---:|---:|
| Euler1 |1.996190304691|.315729516671|3.031004126706e−8|2.83|
| Euler2 |1.996208013440|.315450211352|3.015442581083e−8|5.74|
| Euler4 |1.996216671046|.315315305963|3.007846213513e−8|9.77|
| Heun2 |1.996225280156|.315180918341|3.000290224722e−8|11.28|
| Heun4 |1.996225222711|.315182790590|3.000347851599e−8|19.90|

Euler successive-difference ratios are2.04855 for escaped energy and2.07038 for Ye, consistent with first-order convergence on this example. Heun2→4 changes escaped energy by1.921e−5 relative and the COMPOSITION CHANGE Ye−Ye0 by1.233e−4 relative, both below the preregistered.001 gates. The comparison does not merely normalize errors by the large rest energy, total Ye or tiny temperature difference. Two Heun resolutions do not establish an observed convergence order or an absolute continuum-time error bound.

The Heun4 initial source has net R=7.86475833528445e−7fm^-3s^-1 and Q=1.51550206335300e−5MeVfm^-3s^-1. Fresh final-state evaluations give R=7.32591236462818e−7 and Q=1.48518702578990e−5 in the same units: real source changes of−6.8514% and−2.0003%. This is changing microphysics feedback, not a constant-source tangent.

Seventy-nine additional endpoint/forward/accounting checks pass. A fresh initial source matches the previously accepted source comparison, whose raw hash is verified against the accepted review. At the actual changed Heun4 endpoint, all separate number/species-energy moments refine base→fine within5.230e−9 and fine→wider-tail within2.112e−11, against the unchanged3e−6 gate. Incoming FD particle densities independently pass3e−8. These are selected endpoint controls, not a whole-rectangle quadrature/error certificate.

## Rest-inclusive energy, lepton and EM pair bookkeeping

For a beta or PC event material net electron Le/proton number increases by1 and anti-nu_e carries−1. For EC those material changes reverse while nu_e carries+1. The evolving NB Ye increase cancels integrated escaping nu_e minus anti-nu_e lepton number. Baryon and electric charge conserve. Independent reconstruction from the literal Euler/Heun stage weights reproduces the recorded escaping energy and composition changes.

For Heun4, total material energy-density loss is3.000347850608e−8MeVfm^-3 and escaping energy3.000347851599e−8. The relative energy ledger is3.302e−10 of EXCHANGED energy, and Le residual is−1.241e−24fm^-3. Across the five trajectories the maximum energy-ledger residual is3.474e−10. These small discrete conservation residuals do not establish time-step or source accuracy; the separate time/phase/tail tests above provide that evidence.

Photons lose1.03147966610638e−8MeVfm^-3, while the n+p+e± energy reservoir loses1.968868184502e−8. Their sum accounts for the escape without heat from outside. Fixed density means no pressure-work term. There is no second reaction-Q, pair-rest-mass heating or photon-replenishment energy source: rest-inclusive total EOS inversion already handles those transfers.

The integrated WEAK particle changes are delta_nminus,weak=I_beta−I_EC and delta_nplus,weak=−I_PC. Their differences from the actual final FD particle populations require the SAME net EM pair creation count from either sign. At Heun4 this is4.2637121284566e−10pairsfm^-3, with electron-versus-positron reconstruction residual−5.997e−24fm^-3. This is a bookkeeping diagnosis of the rapid-EM-equilibrium assumption, not an explicit EM kinetic rate. Pair and photon thermal energies are included once in the EOS; adding2me times this inferred count would double count part of the same internal energy.

Using the SAME initial weak sources, the photon-ON temperature derivative is−1.89788065940MeV/s, versus−2.95345818074 without photons. This instantaneous comparison quantifies the reservoir/heat-capacity ansatz; no duplicate photon-OFF trajectory was calculated. Neither case uses an external thermostat.

## Provenance, guarded replay and limits

Before independent-review freeze, admission was strengthened to require a literal accepted-balance flag, exact mandatory source/review sets and loaded-Python coverage, and parent-side conservative-derivative/photon/count checks. The entire first worker/client/PINS plus trajectory/endpoint records are preserved in `pre_admission_guard/`. Main convergence evidence was produced by that exact pinned version. Physical `state()` and `source()` function ASTs and every numerical source control are unchanged. All27 admission/bitwise tests pass, including fresh initial/endpoint source packets. A complete FINAL guarded Heun4 replay takes20.68s and is bitwise identical in ALL physical trajectory fields after omitting timing fields. `final_replay.json` records the final source identity. No failed physical gate was relaxed or suppressed, and no physical trajectory was relabelled with a different execution source.

In total168 related assertions pass:11 photon/EOS preflight,50 trajectory/time,79 forward/endpoint/independent accounting,27 admission/replay/unchanged equations and1 complete guarded trajectory replay. Counts include correlated ledger and provenance controls, not168 independent physical measurements. No new numerical failure occurred. Prior beta/PC/EC coarse failures, review failures and qualifications remain untouched. All individual experiments stay below120s; source and EOS calls run in fresh child processes with no installs or files written outside this owned leaf.

Reproduce only into a fresh copy of the pinned relative dependency tree, preserving leaf names and relative paths as well as frozen outputs; moving this leaf alone does not provide its pinned dependencies. The initial execution sequence is make_pins.py,preflight.py,study.py,endpoint_checks.py; current guarded source checks and complete replay are check_admission.py,final_replay.py. The snapshot pins the exact original stage. `make_pins.py` refuses overwrite; PINS is not an automatic moving-source updater. Figure PNG/PDF shows the discrete Heun4 element and photon subset of the energy loss, not a spatial cooling profile; it was visually inspected.

This is fixed nB, empty incoming neutrinos, selected short free-npe three-channel cooling with an ASSUMED trapped LTE photon/EM-pair reservoir. It omits weak pair annihilation, bremsstrahlung, modified Urca, muons, nuclear interactions/mean fields, Coulomb/radiative corrections, rescattering/trapping/transport, photon escape and long-time/global feedback. Material entropy is an EOS property here; no finite initial empty-neutrino radiation entropy derivative, H theorem, full thermal/chemical equilibrium, steady state, asymptotic balance, full neutrino luminosity, physical-model uncertainty or AD fidelity is claimed. Broader use requires a different review of this coupled ansatz and numerical evidence.
