# Moving-neutron total decay/inverse event and energy moments

**Author-frozen prototype; independent review required.** This implements actual total three-body neutron decay and reciprocal inverse events in a dilute, physical-mass free npe gas. It passes independent proper-lifetime/energy limits, two rest-phase parametrizations, and a new conditional event-spectrum comparison to the differential-tensor route. It does **not** evaluate the tensor spectrum integrated over every neutrino energy, and the preceding crossed-spectral review is not treated as total-moment validation.

## Event measure and spin normalization

The channel is `n -> p+e-+antinu_e` and its inverse. The frozen exact full-current vacuum three-body recipe supplies a neutron-spin-averaged rest measure `dGamma_rest`, with both charged-lepton spins and the one active antineutrino helicity already summed. It includes V/A, WM, PS and physical-q² form factors with physical bare masses mn939.56542052,mp938.27208816,me.51099895069MeV and distinct weak mass938.91875434MeV. No Coulomb/radiative correction or measured-lifetime normalization is added.

For each on-shell neutron momentum in the material rest frame, the common geometric measure is

`dW = [2 d³pn/(2pi hbarc)³] (mn/En) dGamma_rest`.

The spin-two parent measure cancels the neutron spin average in `dGamma_rest`. Every original covariant neutron factor is then `1/(2En)`, as required for both initial-neutron decay and final-neutron inverse phase space. There is no extra electron-spin factor and **no extra antineutrino DOS** in these total events: that phase is already included in `dGamma_rest`.

The two positive event products are evaluated separately:

`dR_F = dW fn (1-fp)(1-fe)(1-fanti)`,

`dR_B = dW (1-fn) fp fe fanti`.

The backward measure is not sampled from a neutron-occupied density PDF or derived by exponentiating detailed balance. The code integrates the **unoccupied geometric parent measure**, retaining the independent forward/inverse FD products in log space. The generalized-Laguerre change `x=(En-mn)/T` pulls out its mathematical `sqrt(x) exp(-x)` weight and multiplies the residual by `exp(x)`; it does not substitute a Maxwell-Boltzmann neutron distribution. Actual fn and1−fn remain present.

Global rest-event orientation is averaged with the Haar measure for an unpolarized parent. Exact Lorentz boosts then supply bath-frame energies and all proton/electron/neutrino occupations. Pair-invariant and electron-energy rest-phase coordinates are implemented separately, using the original exact recoil/current event recipes. Their total weights are checked against one another.

## States and scenarios

Both selected states have nB=1e-12fm^-3,Ye=.3 and T=.03 or.1MeV. `matched_npe` freshly solves the actual relativistic spin-two neutron/proton density roots and exact pair-electron net-charge root. Thermal positrons enter that EOS neutrality, but this **reaction** has electrons only. Muons are explicitly absent; the known cold full-matter muon-susceptibility underflow is not repaired or hidden here. Antibaryon finite-log diagnostics are retained from the particle-only EOS, and remain negligible for the selected states.

Three kinds of radiation/blocking scenario are distinct:

- **Unblocked diagnostic:** physical FD initial neutrons, with final proton/electron blocking disabled and empty antineutrino radiation. This is a normalization/relativity limit, not the matched material rate.
- **Empty radiation:** actual FD proton/electron blocking and fanti=0. The inverse event count is then physically zero because the required incoming antineutrino is absent.
- **Common-T FD radiation:** actual material blocking and `fanti=[exp((Eanti-muanti)/T)+1]^-1`. Use `muanti=mu_n-mu_p-mu_e` for LTE, or shift that value by−T/+T for emission/absorption cases. These are prescribed radiation baths, not solutions of a closed trapped lepton-number or thermal-evolution problem.

All chemical potentials include rest energy; no extra Q term is inserted. The common-T chemical force is `A=mu_n-mu_p-mu_e-muanti`. Independent occupations satisfy `log(F_event/B_event)=A/T` through exact four-momentum conservation. Both F and B moments are nevertheless summed independently. Stable paired differences are used for signed net moments, with no equilibrium floor. Logs and ordinary representation flags accompany positive moments; the current signed-ledger implementation is numerically qualified only where those selected ordinary moments are represented, not in every possible underflow regime.

## Independent proper-time and energy anchors

Direct momentum quadrature of the actual neutron FD gas supplies its density and proper-time-weighted density. In the unblocked limit:

`R_unblocked / nn = <mn/En> Gamma_rest`,

`Qanti_unblocked / nn = integral Eanti* dGamma_rest`,

with corresponding electron and proton energy identities. Rest-frame rotational isotropy gives `<E_lab>=gamma E_rest`; multiplying by the exact `mn/En=1/gamma` cancels the boost for each daughter **energy rate**, while the event number rate retains time dilation. The sum of daughter-energy rates per neutron is therefore `mn Gamma_rest`, rather than a guess based on a rate times an independently averaged parent energy.

The implemented unblocked total number gate agrees within7.77e-13 and the three daughter energy gates within7.92e-13 relative at the refined controls. These are actual total-event tests. They are not an assertion that the all-Enu tensor calculation has passed the same test. Gamma_rest is the declared accepted tree-current result, with no measured-lifetime fit.

## Computed selected totals and accounting

Values below use refined pair coordinates. Number rates are fm^-3s^-1; energy rates are MeV fm^-3s^-1.

| T [MeV] | Unblocked total F | Actual empty-radiation F | Actual emitted antinu energy |
|---|---:|---:|---:|
|.03|7.373339201314e-16|7.371749384312e-16|3.532988329257e-16|
|.1|7.372515305723e-16|7.365901725912e-16|3.529832421424e-16|

At T.03 LTE, the independently evaluated positive F/B counts are approximately6.81992636867e-20/6.81992636866e-20. Their small residual net count is numerical roundoff, explicitly retained. A radiation chemical potential lower by T gives net emission1.16915348306e-19; higher by T gives net absorption−4.31469059293e-20. Entropy production is nonnegative and agrees with `(R_F-R_B) A/T`, without using that expression to construct either event product.

Each positive direction retains separate number, neutrino energy, electron energy, proton energy, neutron energy and **material energy loss** moments. The last observable is independently accumulated as `En-Ep-Ee`, not assigned from the neutrino energy. Species number sources are `(-R,+R,+R,+R)` for neutron, proton, net electron and antineutrino. Material electron lepton change is+R and radiation electron lepton change−R. Signed material energy is the proton/electron gain minus the neutron loss, and radiation gains the separately integrated antineutrino energy. Both the heavy-species subtraction and directly accumulated material loss agree with total energy conservation.

Maximum normalized heavy-species energy-ledger residual is1.05e-12 and direct material/antinu residual is1.15e-13. These ledger identities verify accounting; they are not independent quadrature or physical-accuracy certificates. The EOS includes all nucleon/electron rest energies, so adding separate Q-heating would double count chemical/rest energy.

## New independently conditioned spectrum

`spectrum_anchor.py` uses another exact rest Dalitz phase with rest antineutrino energy and electron energy. Conditioning a moving event on a fixed lab antineutrino energy uses

`cos(theta_nu*)=[E_lab/(gamma Eanti*)-1]/v`,

and the isotropic cone delta gives `1/(2 gamma v Eanti*)`. This conditional event route differs from the tensor's incoming-neutrino q-axis/outgoing-electron integration. Actual bath-frame proton/electron blocking is evaluated after the conditioned boost. Its volume spectrum is divided by the **one-active-helicity** `E_lab²/[2pi²(hbarc)³]` only when converting to j/chi for comparison.

At a new selected E_lab=.42MeV in the T.03 matched bath:

- conditional event j=1.214485094485571e-6s^-1 versus tensor1.2144850944866972e-6;
- conditional event chi=2.5551940555525e-14s^-1 versus tensor2.5551940555538073e-14;
- discrepancies are below9.28e-13 relative for both separately evaluated products.

Both routes inherit the same declared full-current primitives, so this checks phase/spin/DOS normalization, boosts and occupation integration rather than independently establishing the weak-current physics. The new selected spectral anchor is not an all-Enu tensor integral, nor a domain certificate near Doppler endpoints.

## Numerical evidence and remaining gates

The total study has304 passing checks in10.17s and the new spectral anchor6 passing checks. The initial16→24 rest grid produced **24 failed FD-radiation refinement checks**, reaching1.18e-5. Its exact driver/raw report remain in `initial_study.py`/`initial_refinement_failure.json`. The unchanged3e-6 gates pass after24→32 refinement: maximum7.88e-9, with independent pair/electron parametrizations agreeing within9.61e-12. Coarse order8 results and early syntax/import-cycle failures are also retained. No physical kernel or tolerance was adjusted to hide those failures.

At refined controls, independently evaluated chemical-log residuals are below1.12e-11, LTE net count/(F+B) below8.91e-13, and entropy identity residual below8.23e-13 per positive flux. The new spectral refinement passes separately. The largest individual total integral is2.21s, well below120s. Actual positive F/B moments, signed net values, EOS metadata, raw failures, controls and source/environment hashes are pinned in `CHECKPOINT.json`.

Independent review is required. Acceptance should concern these event totals after their stated anchors, not the old crossed-spectral review by inheritance. Full-Enu tensor integration, broader temperature/density/degeneracy/radiation domains, extreme moment underflow, closed matter/radiation evolution, medium/GM1 crossed kernels, Coulomb/radiative corrections and quantified physical-model uncertainty remain open. Positron capture is absent, so this is still not a complete antineutrino inventory.

Reproduce with the existing base environment and no installs:

```bash
/Users/rossma/miniconda3/bin/python study.py
/Users/rossma/miniconda3/bin/python spectrum_anchor.py
```
