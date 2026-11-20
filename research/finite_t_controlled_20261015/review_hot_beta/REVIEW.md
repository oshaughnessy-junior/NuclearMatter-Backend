# Independent review: hot physical beta/inverse total moments

**Accepted at selected scope: 267/267 new checks pass.** No author or upstream file was changed. The author leaf was pinned before numerical work and verified unchanged afterwards. No new failed gates, incomplete runs or repaired tolerances occurred. The original syntax-only author driver failure remains preserved. Individual moments and scenario controls contribute to the check count; these are not267 independent physical benchmarks.

## Independence and forward physical gates

The frozen independently authored reference in review_crossed_moments/reference.py is reused openly at NEW HOT inputs. It uses flat electron/antineutrino Dalitz energies, the source's scalar invariant full-current amplitude, electron-axis Haar orientations, and a finite direct original-parent-momentum rule. The author uses explicit spinor amplitudes, pair-invariant rest coordinates, a different orientation construction and generalized-Laguerre parent integration for its principal results. Both methods implement the same declared physical current; numerical agreement is not independent measurement or an uncertainty estimate for omitted medium/radiative effects.

Two actual physical npe states were checked: the original T=2 MeV,nB=1e-7 fm^-3,Ye=.3 bath and a new T=2.4 MeV,nB=1.7e-7 fm^-3,Ye=.26 bath. Adaptive momentum integrals independently reproduce neutron/proton densities and net electron charge before comparing moments. Electron and positron particles are integrated separately, with pair ratios .8525281373 and .8732297021. Electron blocking uses actual electron particle FD occupations with total mu_e, while neutrality uses n_e−-n_e+. Muons and baryon antiparticles remain absent by the declared EOS ansatz; positrons enter EOS/charge but positron capture is a separate uncombined channel.

The independent parent measure is spin2 d³pn/(2pi hbarc)³ with actual FD occupation and mn/En proper-time conversion. The parent quadrature is in p/sqrt(2mnT), with panel endpoints [0,.7,1.7,3,5,sqrt(80)] and no classical PDF substitution. Rest Dalitz weights are spin-averaged full-current |M|²/(64pi³ mn hbar) dEe dEanti. The spin2 parent cancels that initial spin average; the same phase measure supplies the inverse final-neutron1/(2En). No extra proton, electron or neutrino spin factor is inserted. This is checked by independent absolute normalization, not only reciprocal ratios.

The true unblocked total rate is n_n<mn/En>Gamma_rest. Each isotropic daughter energy moment is n_n times its rest energy moment because Doppler gamma cancels proper-time1/gamma. The neutron energy moment is n_n mn Gamma_rest. These proper-time/Doppler checks pass independently before accepting blocked results. There is no stationary target or measured-neutron-lifetime calibration.

## Actual total moments and both affinity directions

For each state independent rest/parent/orientation/phi controls20/10/8/12→28/16/12/16 were used. All separate positive forward/backward number, antineutrino/electron/proton/neutron and direct-material-loss energy moments were compared, including empty radiation, nonzero LTE and an emission-dominated common-T affinity .8T. Maximum independent refinement is3.51e-14 relative; maximum absolute discrepancy with the author is8.99e-14 against the unchanged3e-6 gate. These are measured differences at selected smooth states, not guaranteed machine-level physical or global accuracy. The reference evaluates F and B directly with actual FD factors, never constructing B from KMS.

For empty radiation the independent refined total values are:

| State | Event number (fm^-3 s^-1) | Antineutrino energy (MeV fm^-3 s^-1) |
|---|---:|---:|
| T2,nB1e-7,Ye.3 |4.25873950656545e-11|2.0237742420493253e-11|
| T2.4,nB1.7e-7,Ye.26 |7.472613804432024e-11|3.559118113118266e-11|

The other separate species energies and direct material loss are retained in raw results. A second independent original-state probe at mu_anti=mu_eq+.8T tests inverse domination. Its net number rate is-6.418748543757127e-12 fm^-3 s^-1 and entropy production5.134998835005685e-12 kB fm^-3 s^-1; all positive F/B and signed energy moments reproduce the author to7.4e-14 relative. LTE cancellation is checked relative to the nonzero gross F+B rate rather than against a vanishing denominator.

At common T, log(F/B)=(mu_n-mu_p-mu_e-mu_anti)/T is checked after computing separate products. Direct entropy is integral(F-B)log(F/B), positive in both affinity directions and consistent with the signed affinity times net number rate. Electric charge, baryon number and Le signs were independently reconstructed for n→p+e−+anti. Signed material energy is Qp+Qe-Qn=-Qanti; direct event En-Ep-Ee was separately integrated. Largest scaled integrated energy-ledger residual is1.45e-12. An extra Q source would double count rest-inclusive energies. Empty-radiation entropy is not assigned a finite FD-reservoir derivative.

## Selected hot spectrum and its exact-state contract

This part is a fresh control ladder plus a source/normalization audit, not a newly independent conditional-spectrum implementation. The hot_spectrum reference is evaluated only at its exact declared original bath and Eanti=.42 MeV. New controls12angular/18lepton/20inner/phi4/width11/tail60→18angular/24lepton/28inner/phi8/width15/tail90 give3.48e-11 relative refinement; final j=.07083091287415232 and chi=.021447614844145972 s^-1. They agree with the frozen conditional-event values within1.95e-10, below the unchanged3e-6 gate.

The isotropic conditional rest-angle measure is dc*/2, so the lab-energy delta contributes1/(2gamma v Eanti_star). A numerical derivative of the original boost relation checks this factor at two independent cone parameters. The one-helicity DOS follows directly from d³k/(2pi hbarc)³ and angular integration, Eanti²/(2pi² hbarc³), without spin2. It divides dR/dEanti only when producing j/chi. The complete total event integral already contains neutrino phase space and receives no second DOS.

Fresh probes reject altered temperature, chemical potential, incident energy and malformed orders/tail/width/budget. The OLD frozen low-T spectral API still rejects T2. This source boundary is essential: the new hot local reference is not a generic extension of the old API. No claim is made for all-energy tensor-integrated beta moments or a uniform hot spectral domain.

## Physical interpretation and remaining scope

The [Guo et al. current conventions](https://arxiv.org/html/2006.12051v2), Eqs26–29 and Appendix B, remain the underlying model: physical masses, timelike q², full V/A/F2/GP and bare weak-current denominator mass. The independent scalar formula changes the numerical amplitude construction; it does not add omitted Coulomb/radiative corrections, interactions, modified Urca or correlations.

Selected hot total beta/inverse moments and the local .42MeV spectrum are accepted with the distinctions above. These are instantaneous fixed-bath physical sources, not an evolved cooling/thermal-box calculation. No positron-capture code or numerical result was used in this review. A full antineutrino inventory, transparent kinetic equilibrium, arbitrary radiation temperatures, broad high-degeneracy/temperature domains, coupled matter feedback, physical uncertainty and derivatives remain OPEN. Repeated numerical agreement cannot replace those gates. No source repair is requested.
