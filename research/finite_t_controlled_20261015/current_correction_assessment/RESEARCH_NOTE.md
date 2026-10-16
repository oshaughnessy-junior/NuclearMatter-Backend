# Controlled F2-only current switch in free EC/PC thermal moments

**Author-frozen research assessment; different independent review required.** At one preregistered actual free-FD bath, removing only the transverse Pauli/weak-magnetism term lowers EC energy emission5.83% and raises PC energy emission6.22%. This does not identify weak magnetism with the entire previously measured stationary-allowed discrepancy. There are149 passing author checks (84 point-current controls plus65 phase/forward/provenance checks), no failed gate, no rescaling and no upstream source change.

## Locked physical experiment

LOCK.json was written2026-10-16T02:36:49.074852Z before physical current-switch data. The bath is T3MeV,nB1e-7fm^-3,Ye=.5490065797957709, the previously independently reviewed detailed transparent composition root. Every child solves fresh actual free relativistic n,p densities and charge-neutral pair-equilibrated e±, with physical masses and total chemical potentials. Muons are absent, incoming nue/antinue are empty and beta is outside this assessment. Temperature/density/composition are prescribed; no hydrodynamics or thermal trajectory is evolved.

The only change is F2 ->0 in the source-defined one-body current

`Gamma^u=F1 gamma^u - GA gamma^u gamma5 + i F2 sigma^(u v) q_v/(2Mweak) - GP q^u gamma5/Mweak`.

Here sigma^(uv)=i[gamma^u,gamma^v]/2, metric(+---), physical q=pb-pa in the ordered absorption current, t=q², Mweak=(mn+mp)/2=938.91875434MeV. Capture uses the original reciprocal/crossed orientation and initial-spin average4, with no neutrino spin average or extra target factor.

The baseline coefficients follow the already qualified [Guo et al. source current/form factors, Eqs27 and47](https://arxiv.org/html/2006.12051v2). Write den=(1-t/(4Mweak²))*(1-t/840²)². Hold exactly fixed

* F1=(1-4.706*t/(4Mweak²))/den;
* GA=1.2723/(1-t/1000²)²;
* GP=2Mweak²*GA/(139.57²-t).

Change only F2=3.706/den to0. In particular F1 retains its original4.706 numerator. Varying the common xi/Sachs parameter everywhere would change F1 too and would be a different experiment. Here this is a controlled algebraic current slice at fixed other form factors, not a new fitted electromagnetic form-factor family or a physical uncertainty model.

Recoil, moving-target relativistic phase space, all FD populations/vacancies, EOS roots, physical rest energies, all nonmagnetic form factors and normalization constants remain fixed. No stationary-target substitution, density rescaling, inverse-KMS substitution or surrogate supplies the thermal rates.

## Clean isolation and forward gates

The vacuum current already exposes `FormFactors(weak_magnetism=False)`, which is used for direct-current controls. The thermal EC/PC scalar implementations hardcode their coefficient expressions and do not expose that hook. `build_configured.py` therefore extracts the exact pinned scalar function ASTs into owned configured_currents.py. For the switched functions the sole physical substitution is the single f=3.706/den assignment becoming f=0. Original EC E.GF/E.VUD names are bound to identical literal constants in this owned module. CONFIGURATION_PROVENANCE.json records each original source SHA, exactly one assignment location and generated-code identity. Nothing is written to upstream code.

Each variant/profile runs in a fresh child `worker.py` process. Its deliberate function injection is local to that isolated process. The parent orchestration imports no mutable physical module. Raw phase output retains its original internal scalar label; the enclosing result explicitly records the configured variant, owned current hash, worker hash and phase/source identities. It must not be read as an unmodified-current certification for F2zero.

Before any switched thermal run, both baseline profiles pass the locked3e-6 absolute number/energy forward gate against immutable accepted detailed state_1.json. Fine baseline values match that reference exactly; the largest coarse discrepancy is4.61e-10. Both variants then pass separate-profile refinement, maximum4.74e-10 against3e-6. Actual proton/neutron FD integrals pass3e-8 and EOS n,p/net-charge roots pass1e-10. Baseline and switched bath/EOS dictionaries are identical.

Profiles are EC20/32/8/8/8 tail50 ->28/48/12/12/12 tail60 and PC28/44/10 ->32/52/12 at tail60. There are4 combined EC+PC physical calls, serial and subprocess-isolated, taking9.45s total. No integral or tolerance is retuned after observing differences.

The84 current controls cover six on-shell vacuum kinematic points: nu/antinu at energies4.7,17.3,48.9MeV with cosines-.61,.22,.78, for both current variants. Owned baseline scalar functions match original scalar functions; each avg4 configured scalar amplitude agrees with the independent literal Dirac trace divided by2, maximum4.45e-16. This division converts its initial-baryon spin average2 to the reciprocal original charged-capture average4, whose charged-target spin density is already included in the phase. Positivity, unchanged V/A/P vertices and coefficients, exact F2zero, magnetic q-contraction/transversality and Dirac-adjoint reversed-q identities pass. Only the magnetic term is asserted transverse; no conservation claim is imposed on the full unequal-mass weak current.

## Selected absolute results

Number is fm^-3s^-1 and neutrino energy MeVfm^-3s^-1. “Removal” means F2zero/baseline-1, so its denominator is always the positive full-current result.

| Channel/moment | Baseline full F2 | F2=0 | Removal fraction |
|---|---:|---:|---:|
|EC number|5.010099210059188e-6|4.763749866209821e-6|-.0491705520|
|EC nue energy|7.318736618959663e-5|6.892283477173827e-5|-.0582686827|
|PC number|5.010072918472533e-6|5.273642674319890e-6|+.0526079680|
|PC antinue energy|7.811780798644817e-5|8.297870370641985e-5|+.0622251935|

Equivalently adding F2 to this otherwise fixed current enhances EC number/energy by5.17%/6.19% and suppresses PC number/energy by5.00%/5.86%, with F2zero as those denominators. These signs/sizes refer to the selected thermal moments, not a uniform all-energy assertion. The event measures, weak prefactor and current normalization are unchanged, so the difference is a conditional F2-only current effect within the stated free model.

## Relation to the earlier joint allowed-model discrepancy

At this SAME state the stationary allowed model changes many ingredients together. Its allowed/full differences were EC number/energy -3.198%/-3.397% and PC number/energy +10.388%/+12.823%. Those assessment failures remain unchanged. Comparing the allowed moments to the new F2zero reference gives EC +1.808%/+2.580% and PC +4.871%/+6.214%.

Thus the PC energy discrepancy remains6.214% after the controlled F2 removal; assigning the full12.823% discrepancy to weak magnetism would be incorrect. Ratios multiply rather than add: allowed/full=(allowed/F2zero)*(F2zero/full). The remaining discrepancy jointly involves stationary kinematics/recoil/moving distributions, baryon vacancies and the remaining current/form-factor approximations. This experiment does not isolate any of those, nor assign additive fractions uniquely in the presence of current interference. It is not a model-uncertainty or empirical-error estimate.

## Physics scope and reproducibility

EC is p+e- -> n+nue; PC is n+e+ -> p+antinue. Their emitted number/energy moments remain separate. One active emitted helicity and the original unpolarized target conventions are already included. EOS/events are rest-inclusive: no additional reaction-Q or pair-rest source is introduced. No beta channel, combined net-source/root accuracy, full inventory, evolution, AD, interacting/mean-field/RPA correction, Coulomb/radiative effect or broad domain is qualified by this one-bath current switch. Method agreement is algebraic/numerical evidence inside the common model, not an independent physical model.

Run from Genesis, with existing interpreter and no installs:

```
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/current_correction_assessment/build_configured.py
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/current_correction_assessment/current_controls.py
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/current_correction_assessment/run_study.py
```

The builder and completed study refuse overwrites. Reproduce in a distinct authorized sibling output directory with exact LOCK/source files rather than modifying this frozen record. Full raw moments, partial checkpoints, child stdout/stderr, generated-current provenance, current controls and comparison.csv are retained. All pinned external physics/EOS/reference/review files verify unchanged before/after. Current source/evidence execution identities are recorded separately from upstream scientific-source pins. No parent global, accepted upstream byte, dependency, production, paper or git state changed. Different independent review is required before promoting these results.
