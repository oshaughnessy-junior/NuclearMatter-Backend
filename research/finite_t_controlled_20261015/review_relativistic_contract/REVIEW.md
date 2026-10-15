# Independent review of the relativistic response implementation contract

**Accepted as a convention/derivation contract, not an implemented finite-temperature tensor.** All193 new independent checks pass (108 normalization/kinematics/KMS,21 support-interval and64 active-helicity projector checks). Author files, equations, original30 checks and results remain unchanged. Experiments take milliseconds; no long numerical scan or tensor implementation is hidden in this review.

## Independent absolute normalization

Starting from the invariant probability with target-spin-averaged squared amplitude and per-spin occupation fa,

    chi = [2/(2Enu)] integral dPi_a dPi_l dPi_b (2pi)^4 delta4 Mavg² fa(1−fb)(1−fl),

the physical target density is n_a=2 integral d³pa fa/(2pi)^3. With a full baryon trace, Mavg²=(GF²Vud²/4) L T, whereas the fully target-spin-summed amplitude is (GF²Vud²/2) L T. Thus the leading2 and target average cancel exactly. There is no additional final-baryon or final-lepton spin factor outside the traces.

A new explicit gamma-matrix calculation splits the massless incoming spin density into PL slash(k) and PR slash(k). Across two massive outgoing-lepton states and all16 lepton-tensor components, the active projector reproduces the full slash(k) trace and the inactive projector vanishes, to3.62e−16 scaled error. Averaging those two helicities would introduce an erroneous1/2.

After integrating the baryon momentum delta, its invariant measure is d³pa/(2pi)^3 ×2pi delta(omega+Ea−Eb)/(4Ea*Eb*). The charged-lepton azimuth gives dPi_l=pl dEl dc/(8pi²). Combining GF²Vud²/(4Enu) with this measure yields exactly

    dchi/(dEl dc) = GF²Vud² pl/(32pi² Enu) (1−fl) L_munu W^munu.

No further2pi, target-spin average or neutrino average belongs here. W has MeV², L has MeV² and chi has MeV in natural units. For q-axis integration, the energy delta Jacobian is Eb*/(p q), leaving W=[1/(16pi²q)] integral dEa* dphi T fa(1−fb), equivalently1/(8piq) with an azimuth-averaged trace. Changing the outer lepton variable to q gives the positive measure GF²Vud² q/(32pi² Enu²) dEl dq.

The independent absolute witness integrates the final two-body Lorentz phase space in the **lab** using its radial charged-lepton delta root, rather than the contract's initial-baryon q-axis root. Its integrand is p_l²/[8pi El Eb |d(El+Eb)/dp_l|] after lepton azimuth. Three unequal-mass/electron/muon/moving-target cases reproduce Phi2=pstar/(4pi sqrt(s)) to2.67e−15 relative. For a unit target-averaged amplitude, sigma=Phi2/[4(k·pa)] and v_Moller=(k·pa)/(Enu Ea). The contract's invariant opacity per physical target density independently matches v_Moller sigma to2.67e−15. This is an absolute measure/spin witness, not a new weak-current or finite-T rate calculation.

Gaussian energy deltas integrated in the original baryon angle, with three widths, match the reduced1/(4Ea p q) angular invariant measure to2.43e−12 relative. Thus the q-axis Jacobian check does not rely only on differentiating the same symbolic root.

## Support, KMS and NR closure

Reconstructing original mass shells gives x0=(C+2wEa*)/(2pq), C=ma*²−mb*²+w²−q². Squaring the angle condition gives the stated quadratic inequality; positive initial and final energies must still be imposed. New on-shell points test spacelike and timelike transfers, with polynomial errors below1.97e−11 relative and original energy-shell residuals below1.14e−13MeV.

A separate interval witness tests whole sampled energy ranges and exact endpoints. For ma700,mb810,q20,w105MeV, support is the finite interval Ea*=713.77541353–822.68341000MeV. Its reversed transition has the shifted finite interval. Equal-mass timelike samples have no particle-hole support. Null transfer q=|w|30MeV yields a linear lower threshold; for ma700,mb810,w30 it is1472.66756974MeV. These independently confirm why a generic lower-threshold-only or |omega|<q solver would be insufficient. They do not implement a robust production interval solver or antiparticle cuts.

Pointwise occupation ratios at new unequal vector shifts recover j/chi=exp[−(Enu−mu_l−mu_b+mu_a)/T] to1.78e−15 absolute log error. The exponent uses total chemical potentials and physical energy conservation; inserting another potential shift would double count. Tensor-transpose reverse balance assumes the explicitly chosen Hermitian forward/reverse current prescription; the future implementation still needs actual integrated forward/reverse component tests.

The full density trace tends to spin2 after division by4Ea Eb. Independently contracting L00=8Enu(El+pl c) and sum_i Lii=8Enu(3El−pl c) gives the allowed angular weight with beta_l=pl/El. It closes the rate coefficient to GF²Vud²/(4pi²) El pl, and restoration of (hbarc)^3/hbar exactly matches the frozen NR optical coefficient. This is a matched common-mass/offset limit; physical unequal bare masses are not silently identified with the earlier common-mass NR gas.

## Primary-source verification and GM1 restrictions

[Guo et al., Eqs23–28](https://arxiv.org/html/2006.12051v2) were checked directly. Their invariant normalization uses kinetic baryon energies; the vertex has physical-q² form factors, kinetic q* in WM/PS vertices and a bare weak normalization mass. These source-specific choices support the declared prescription, not a unique conserving interacting model.

[Roberts–Reddy, Eqs3–14 and81](https://arxiv.org/html/1612.02764v2) independently anchor spin degeneracy, phase measures and the32pi² differential-rate denominator. They also explain why charged-current vector conservation fails generically for unequal masses/self-energies. The contract correctly keeps all tensor components and does not use a generic CVC reduction.

For the archived common-bare-mass GM1 Hartree state, mD=939−Phi, nu_i=mu_i−Ui includes mD, physical q0=omega and kinetic q0*=omega−Ub+Ua. These must remain distinct. Mweak belongs to the weak-current convention and must not be replaced by mD or a Landau mass. Contract reuse of the existing vacuum trace without separating Mweak would therefore be invalid; the document explicitly blocks that shortcut. Its common-mass GM1 vacuum limit also cannot be compared directly with the physical neutron-proton mass-split vacuum channel. No external matched GM1 forward comparison has been supplied by this contract.

## Scope and preserved failure

The contract correctly leaves scalar/vector EOS energy bookkeeping, field self-consistency, anti-baryon diagnostics, crossed-decay inventory, actual tensor positivity, KMS integrations and independent dilute moving-target matching as implementation gates. Its30 author witnesses and these193 review checks cannot certify a tensor that does not yet exist. The new constant-amplitude phase-space witness checks normalization separately from the author's full-current c-sigma benchmark; neither is physical uncertainty estimation.

An initial review-only import failed because the NR module exposes GF,Vud,hbarc through its `constants` namespace. The original script and failure record are preserved; only review imports were corrected before replay. No author fix, equation change, calibration or source mutation occurred.
