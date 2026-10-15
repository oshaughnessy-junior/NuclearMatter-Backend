# Independent numerical review: moving-neutron crossed total moments

**Accept the named total-event reference and selected conditional-spectrum scope. 230/230 new checks pass.** This review performs actual total-moment calculations; it does not inherit total validation from the older spectral review. It owns only this directory, preserves all author sources/raw failures and verifies their hashes unchanged. No physical numerical failures or incomplete experiments arose in this batch. The original24 author coarse-FD refinement failures remain present and explicitly checked.

## Common invariant measure and inverse normalization

For n→p+e−+antinue and its reciprocal, the common fully spin-summed covariant measure can be written as

`dW = [2 d³pn/(2pi hbarc)³] (mn/En) dGamma_rest`,

where dGamma_rest averages the **initial neutron spin only**. The factor2 converts that average back to the common full neutron-spin sum. Rest decay supplies1/(2mn); proper-time mn/En supplies exactly1/(2En). That same factor is required when the neutron is a **final** particle in the inverse-three-body event. Thus the inverse integral must range over an unoccupied geometric neutron momentum measure and apply(1−fn), not sample an occupied neutron PDF and somehow infer its inverse weight afterward.

The source does this correctly. Actual forward fn(1−fp)(1−fe)(1−fanti) and backward(1−fn)fp fe fanti remain separately evaluated FD/log products; the generalized Laguerre exp(x) compensates its quadrature weight rather than replacing fn by a Boltzmann gas. Electron/proton spin sums and the active antineutrino state already occur in the amplitude. There is no additional target/electron spin factor or antineutrino DOS in a **total** decay event. Adding one would violate the independent proper-lifetime anchor.

## New independent total integration

`reference.py` changes three numerical constructions:

- The rest phase is a flat **(Ee,Eanti) Dalitz integral**, with exact proton-shell energy bounds. The author totals use pair invariant/angle or Ee/electron-neutrino angle coordinates.
- Its matrix element comes from the frozen **published scalar full-current invariant equations**, rather than the author's explicit spinor contraction. Three new on-shell events first compare these scalar equations with the accepted explicit crossed spinors, agreeing within5.56e−16. Source/current physics is shared and not called an external physical uncertainty test.
- Parent momenta use direct p/sqrt(2mnT) panels with actual FD occupations and a finite80T-scale momentum extent, rather than generalized Laguerre. Rest orientation is averaged around the **electron axis**, with neutrino-cone azimuth, rather than the author's pair/electron-plane rotations. Both paths use the same isotropic unpolarized physics but different phase ordering and samples.

All boosts evaluate bath-frame proton/electron/neutrino occupations and energies. The reference accumulates each positive direction's number, antineutrino energy, electron energy, proton energy, neutron energy, and **direct material loss En−Ep−Ee** independently. Signed differences use paired expm1 without an equilibrium floor. The direct material observable is not assigned from Eanti, and the heavy material-energy combination is checked separately.

## Forward gates, fresh states and absolute moments

Both new physical free-npe baths have T=.055MeV,Ye=.31. Their nB values are2.3e−12 and2.3e−8fm^-3, with all material chemical potentials freshly solved. Before rates, independent momentum integrals recover neutron/proton particle densities and the electron **net pair charge**. Physical masses/total-rest conventions, antiparticle logs and explicit absence of muons are audited. Electron and positron pair populations enter EOS neutrality; this reaction includes actual electrons, not a net-electron density multiplier or positron capture.

For each bath, common-T antineutrino radiation has muanti=(mu_n−mu_p−mu_e)−.37T, giving positive chemical force A/T=.37. Separate unblocked, empty-radiation, LTE and finite-affinity scenarios are computed. These are prescribed radiation baths, not a closed trapped-composition solution. The denser new bath is a **held-out extension**, not a uniform extension of the author's original density domain.

| new bath nB [fm^-3] | forward finite-affinity count | inverse count | net count | net antinu energy |
|---:|---:|---:|---:|---:|
|2.3e−12|4.96519597015e−19|3.42963131493e−19|1.53556465523e−19|1.08196175131e−19|
|2.3e−8|1.13173354443e−13|7.81727212274e−14|3.50006332159e−14|6.02928844938e−15|

Count units are fm^-3s^-1; energy unitsMeVfm^-3s^-1. Actual forward/inverse values for all6 moments, not only net totals, are retained. Each state gives96 passing checks. The independent fine reference agrees with fresh author totals within9.08e−8 across both states and all positive moments. Independent20→32 rest/parent/orientation refinement passes the unchanged3e−6 gate; the largest difference is2.50e−6 in the denser LTE electron-energy moment, so that selected check has a limited margin. It is not reported as a stronger accuracy bound. No tolerance or physical model was adjusted.

The denser state's neutron same-mu Maxwell-Boltzmann density exceeds its FD density by **.00091413**. This is far above the forward-density gate, so a Boltzmann substitution would be detectable; neither numerical path makes it. The dilute state's corresponding correction is9.14e−8. These diagnostic classical densities are not used in the physical rates.

## Proper lifetime, boosts, ledgers and entropy

Unblocked actual-FD neutron totals independently satisfy `<mn/En> Gamma_rest`, using the direct parent integral. Each daughter energy rate is instead nn times its proper-rest energy moment: isotropic rest orientation gives Elab=gamma Erest after averaging, canceling proper-time1/gamma. Parent energy correspondingly gives nn mn Gamma_rest. It is incorrect to multiply the time-dilated number rate by a separately averaged parent energy and assume that accounts for the correlated boost.

These absolute lifetime/Doppler gates pass, in addition to agreement with the independently reduced full-current vacuum rate. They are not inferred from the algebraic energy ledger. The same bare physical weak mass, timelike full-current form factors, no Coulomb/radiative corrections and no measured-lifetime fit are retained.

Species sources are n:−R,p:+R,e_net:+R,antinue:+R. Independent audits confirm baryon number, electric charge and electron lepton number conservation. Material energy Ep+Ee−En and the separately integrated radiation energy cancel, as does the direct material-loss observable. These identities check accounting, not total-integral convergence. Since all rest energies are in the material convention, adding another Q-heating term would double count.

For common-T FD radiation, forward/backward log ratio is A/T at each exact event. Net count has the expected sign; entropy is nonnegative and equals netR*A/T within the stated numerical gate. LTE residuals are retained at working precision, with no artificial zero or cancellation floor. This qualifies reservoir-bath microscopic balance and ledgers at these states, not a complete evolution or entropy theorem for an arbitrary radiation closure.

## Conditional spectral check is a separate gate

I reviewed the source derivation of the cone condition Elab=gamma Eanti*(1+v cos(theta*)) and independently differentiated its isotropic delta measure: dcos(theta*)/2 produces1/(2gamma v Eanti*). It multiplies the separate proper-time1/gamma; those two gamma factors have different origins. Only after conditioning the total event phase on a lab energy should its volume spectrum be divided by the **one-active-helicity** DOS Elab²/[2pi²(hbarc)³]. The code applies that factor once in the right place.

At a **new** selected Elab=.365MeV in the T.055/nB2.3e−12 bath, the conditional event route and the frozen tensor spectrum give

| coefficient [s^-1] | conditional | tensor |
|---|---:|---:|
|j|3.203512569127092e−6|3.2035125691271257e−6|
|chi|5.538170486805377e−12|5.538170486805199e−12|

Separate refinement/coordinate differences are below2.07e−12;6 checks pass. This is a new selected spectrum/phase/DOS check with shared current physics. It is **not** the tensor spectrum integrated over all neutrino energies. The full-Enu tensor number/energy gate remains OPEN despite accepted total moments by the event method.

## Evidence and limits

`probe.py` and `probe_denser.py` each give96PASS in4.46/4.61s. `spectrum_check.py` gives6PASS in5.66s; `audit.py` gives32PASS. Completed panel checkpoints remain available; there were no timeouts or erased failures. The author's original24 coarse refinement failures, early driver/import errors and selected spectrum evidence are preserved, alongside original source hashes. Those failures concern insufficient quadrature, not repaired physical kernels.

Accepted: the original named moving-neutron event totals, their absolute unblocked limits, selected blocked FD/LTE/net/energy accounting, the two new held-out baths, and two selected conditional spectral anchors including the author's. No uniform density/T/degeneracy certificate follows. Stronger degeneracy, extreme underflow, broad spectrum/endpoint tails, full-Enu tensor integration, positron capture/complete anti inventory, muon channels, medium/GM1 crossed kernels, Coulomb/radiative effects, physical current uncertainty and closed matter/radiation evolution remain separate work. No author repair is required within the reviewed scope.
