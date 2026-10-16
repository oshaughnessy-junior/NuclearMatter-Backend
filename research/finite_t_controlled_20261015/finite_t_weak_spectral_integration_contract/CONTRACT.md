# Stage B portable free charged-current spectra: direct j and true absorption chi

**DESIGN ONLY, awaiting DIFFERENT independent design review. No implementation or new physical call is authorized by this note.** Stage A `production_weak_port/` is frozen and has now received DIFFERENT selected numerical/physics acceptance after its narrow signed-log repair (417new passing rechecks; initial1378PASS/2FAIL preserved); packaging/normal CI/minimum-runtime gates remain separate. That acceptance does not qualify Stage B or authorize this implementation. The repaired accepted Stage A design contract remains authoritative. Future implementation requires a separate explicit commission after prerequisite review/admission, a fresh exact upstream commit/source pin and a numerical LOCK written before any candidate validation.

Scope: three named electron charged-current channels EC/PC/NEUTRON_DECAY, explicit fixed free chemical bath, physical masses/full Guo current, isotropic material and direct per-occupation spectral coefficients. No EOS/neutrality/beta factory, RMF, cooling evolution, AD, fitted spectral surrogate, complete weak/flavor inventory or transport closure. Detailed quadrature remains the default. This design uses only frozen SAVED physical evidence and source reading, not fresh rates, roots or derivatives.

## Exact proposed changed paths: eleven

The future Stage B change, after Stage A has passed its prerequisite gates, consists of:

1. MODIFY `src/nmbackend/reference/weak_reference.py`: replace the spectral-unavailable stub with explicit named dispatch, frozen spectral result/control types exported from the common helper, separate reconstruction entrypoint and units. Existing total API/constructors/semantics remain regression-gated.
2. NEW `src/nmbackend/reference/_weak_spectral_common.py`: spectral-only coefficient/support/control/result records, positive-log relaxation/unit conversions, one-helicity DOS and bounded reconstruction quadrature/admission. No runtime fixture/qualification-file reads.
3. NEW `src/nmbackend/reference/_weak_spectral_capture.py`: EC/PC original exact q-axis shell, actual lepton-energy/angular integration and direct material occupations. Shared geometry explicitly receives the correct physical species ordering.
4. NEW `src/nmbackend/reference/_weak_spectral_decay.py`: moving-parent conditional decay/inverse phase at fixed lab antinu energy; no histogram or rest-endpoint truncation.
5. NEW `tests/test_weak_reference_spectra.py`: currents, support, direct j/chi, guards and independently implemented phase/trace anchors.
6. NEW `tests/test_weak_reference_reconstruction.py`: one-helicity DOS, positive F/B R/Q reconstruction, independent event totals, bin ledgers and falsezero/partial-result adversaries.
7. NEW `tests/data/free_weak_spectral_reference.json`: source-labelled INTERNAL saved reference state/points/totals, no claim of external calibration.
8. MODIFY `examples/free_weak_reference.py`: explicit selected spectra/true absorption/relaxation units and a small bounded optional reconstruction example; retain Stage A total example.
9. MODIFY `docs/weak_reference.md`: source conventions, exact support/units/limits, selected evidence, failure and qualification scope.
10. MODIFY `README.md`: narrow discovery text; existing cold/public exports remain unchanged.
11. NEW `docs/weak_spectral_validation.md`: reproducible numerical profiles, isolated tails, first-domain/heldout LOCK, qualification ledger and unresolved gates.

Do NOT modify `_weak_quadrature.py` to admit arbitrary empty-support zero into its Stage A positive-total type. The new common helper owns spectral-only support records. `_weak_current.py`, `_weak_two_body.py`, `_weak_decay.py` and existing total tests/fixtures remain unchanged dependencies unless a separately preserved failed review explicitly authorizes a narrow prerequisite repair. If implementation cannot fit this honest eleven-path boundary, stop for a reviewed scope revision; do not bury extra runtime modules/dependencies in data/example files. No unmerged component dependency or campaign-directory loader is allowed.

The source graph is in DEPENDENCY_GRAPH.md. New runtime helpers use package-relative imports, immutable state/current/controls and existing fixed current primitives. Capture can reuse `_weak_current.scalar_capture` plus its already independently gated literal contraction; decay shares `_weak_current.decay_amplitude` and boost mathematics, not a total-event grid pretending to be conditional phase. `_weak_decay.pair_events` parametrizes a different rest lepton-pair invariant and is NOT the electron–proton pair needed by the conditional algorithm. No generic aliases, model-global injection, runtime source JSON, driver execution or result cache.

## Public API and first request domain, to be locked before tests

Proposed signatures, not implemented here:

```
spectral_coefficients(channel, bath, E_MeV, controls=SpectralControls(...))
capture_differential_coefficients(channel, bath, E_MeV, El_total_MeV, cosine, controls=...)
reconstruct_spectral_moments(channel, bath, radiation, energy_rule, controls=...)
```

`channel` is exactly EC,PC,NEUTRON_DECAY. The capture differential entrypoint rejects NEUTRON_DECAY; decay's conditional cone/Dalitz state has a different measure. No fake universal `(E,El,cos)` kernel is exposed across all three. `spectral_coefficients` returns coefficients stripped of the external neutrino occupation; it takes no Radiation argument. A separate local-source/reconstruction operation applies actual one-flavor radiation occupations/holes, without altering the coefficients or requesting a different hidden bath.

The first portable spectral candidate accepts ONLY the explicit bath T=2MeV,mu_n=919.8287578100662,mu_p=916.8449303084382,mu_e=.17446333200826397MeV and the fixed Stage A physical masses/current. These are the saved neutral nB1e−7fm^-3,Ye.3 state's chemical potentials, not an API that recalculates or assumes neutrality. No EOS root is run. Exact supplied-state identity is checked; a near-match, another T/n/Ye or a different current raises. The more general Stage A request domain does NOT silently become a spectral domain. No factory named `matched()` may solve a hidden EOS behind a nominal spectral request.

First numerical request guards: strictly finite non-Boolean E>0; capture E<=60MeV; decay E<=1.4MeV; capture differential El_total>ME and cosine[-1,1]; exact singular-coordinate/degenerate support receives a derived limit/alternate phase or explicit unresolved-coordinate exception. These energy bounds support the original reconstruction quadratures; they are **not** uniform accuracy certificates, physical emission endpoints or statements that the omitted tail vanishes. Disallow E=0 (DOS/division limit needs independent analysis). Controls use literal bounded integers and predeclared positive finite tail/budget/resource limits; exact numerical orders and conditioning margins must be LOCKED before future candidate calls, not inferred after failures.

Saved qualification points: capture E=2,5,10MeV; beta E=.2,.42,.5,.8MeV. Mandatory fresh independent review points are separately preregistered before calculations and cannot simply reuse these saved anchors. Primary cold/vacuum limits, q-axis versus longitudinal-shell/spinor points, timelike EC, spacelike PC, near-rest beta endpoint and above-rest beta support are distinct gates. First implementation should predeclare a few NEW differential and selected-E holdouts at this SAME bath; a later broader-bath numerical LOCK/review is a separate phase. Do not extrapolate source convenience guards into another state or soft-log relative-accuracy claim.

Returned spectral records name channel/flavor, physical current identity, bath, E, direct positive/log j and true_chi, log-safe relaxation, exact support evidence, actual maps/bounds/nodes, time budget/completeness and numerical qualification. Length conversions are explicitly named helpers/properties, never an ambiguous field `opacity`. No derivative, steady-state, net-root or model-error certificate.

## Oriented current transfer and phase normalization

Keep q_Gamma=pp−pn as the n->p current argument; reciprocal Gamma_pn(q_pn)=DiracAdjoint[Gamma_np(−q_pn)], q_pn=pn−pp=−q_Gamma. Form factors depend on invariant q_Gamma² but the F2/GP vertex terms are linear in the oriented transfer. Physical bare Mweak=(MN+MP)/2, not an effective mass. The exact source physics is the same [Guo Eqs26–29,47](https://arxiv.org/html/2006.12051v2) and [Strumia–Vissani convention check](https://arxiv.org/html/astro-ph/0302055) already pinned by Stage A; no duplicate literature model or new coupling choice is introduced.

For capture absorption geometry let Q_abs=k_nu−p_lepton=(omega,vector_Q), q=|vector_Q|, omega=E−El. EC absorption n->p has Q_abs=q_Gamma; PC absorption p->n has Q_abs=q_pn=−q_Gamma. EC emission uses the reciprocal baryon current; PC emission uses the direct n->p current. Pass physical neutron/proton vectors to the fixed scalar/literal provider: do not flip a scalar crossing sign a second time merely because its internal algebra is written in absorption ordering.

The full-current scalar A averages FOUR initial baryon/charged-lepton forward spins. The corresponding unaveraged sum is4A. After the exact baryon polar delta, baryon invariant phase is dEa dphi/(16pi²q), the outgoing charged-lepton phase is pe dEl dcos/(8pi²), and fixed incoming neutrino leg1/(2E). Therefore

    d(j or chi)/(dEl dcos) = pe/[32pi³ q E HBAR] * integral dEa <A>phi * material occupations.

No extra target-spin2, neutrino-helicity average, phi2pi or HC volume factor is inserted into these per-occupation coefficients. Current primitives already include GF²Vud² and their declared spin average. Differential units are s^-1MeV^-1 per cosine. Integrating actual El/cos yields j and chi in s^-1 per one active flavor occupation. No normalization fitted to event totals is permitted.

Direct material occupations, with physical on-shell species energies:

| channel | j product | true chi product | equilibrium flavor mu |
|---|---|---|---|
| EC | fp(1−fn)fe− | fn(1−fp)(1−fe−) | zeta=mu_p+mu_e−mu_n |
| PC | fn(1−fp)fe+ | fp(1−fn)(1−fe+) | −zeta |
| neutron decay | fn(1−fp)(1−fe−) | (1−fn)fp fe− | −zeta |

j and chi are accumulated separately in log space, BEFORE any external radiation occupation. Check log(j/chi)=−(E−mu_eq)/T only afterwards. This necessary equilibrium identity cannot establish a spin/prefactor/phase normalization or resolve a missing tail.

## Exact two-body support and numerical boundaries

For absorption initial baryon a, final b: Ea>=ma, Eb=Ea+omega>=mb, p_a=sqrt(Ea²−ma²). The exact shell requires

    cos(theta_aQ)=[ma²−mb²+omega²−q²+2omega Ea]/(2q p_a), abs(cos)<=1.

With c0=(ma²−mb²+omega²−q²)/2 and A0=omega²−q² this becomes

    A0 Ea² + 2omega c0 Ea + c0² + q²ma² <= 0,

followed by both positive-energy restrictions. Retain all intervals, spacelike/timelike/linear boundaries and degeneracies. The squared inequality alone is not a complete support proof. For q=0, division is a coordinate failure, not a physicalzero. A small discriminant cannot be arbitrarily clipped into a nonempty/empty branch; LOCK explicit conditioning/admission and test the exact linear limit or raise unresolved. Physical constraints and numerical representability must be distinguishable.

EC uses (ma,mb)=(MN,MP), PC uses (MP,MN), with total chemical potentials in matching absorption occupancy order. The physical free PC branch is strictly spacelike; EC also has qualified timelike support. The stationary free-proton antinu threshold1.8060662080187608MeV is a target-rest benchmark, not a finite-T E cut. The saved Eanti1.7,El=.5987748936352157,cos=−.1995027501115606 point is positive with exact moving-target support; preserve it as a future regression. Below-threshold tails must come from actual phase/FD weights, not an inserted width.

Outer El integration resolves El=ME+y², threshold and thermal/recoil/Fermi-edge panels. Inverse/forward material chemical centers both constrain truncation. Thermal/recoil scales guide numerical panels, not physical smearing. Distinct physical panel endpoints that collapse to the same floating y require a derived alternate coordinate or an explicit unresolved-panel exception; never log(0), silently remove a resolved band or call the result exactzero. Width representability, collapsed mapped endpoints and ordinary-output underflow ALONE NEVER justify skipping a panel. For example log(width)=−750 and log(integrand)=+100 give ordinary width0 but a represented positive weighted integral exp(−650); this is an abstract integration counterexample, not a new weak-rate evaluation.

Any optional panel omission requires a documented rigorous bound on the COMPLETE weighted contribution to BOTH j AND chi, including current amplitude, all FD factors, invariant measures/Jacobians/angular factors, HBAR/HC/DOS normalization as applicable, and the target coefficient/reconstruction units. Predeclare and independently validate the error policy before future physics calls. Retain log bounds even when ordinary bounds underflow. Control omitted Uj/Uchi against independently certified positive lower bounds Lj/Lchi on the retained contributions using the declared relative/log budgets (for example log1p(U/L)), and against separately declared absolute/gross source/energy budgets where cancellation matters. A bound on geometric width or one isolated factor is not a bound on that complete contribution. Do not substitute an upper bound for an actual coefficient. An omission must not erase known positive integrated support or finite-log semantics: if it is the only resolved positive contribution, or a valid full weighted/log error bound cannot be established, compute in the alternate coordinate or raise unresolved rather than returning zero. Record every allowed omission/bound and include it in the final error/completeness diagnostic. No implicit support cut, floor or numerical width is introduced. Test regular/timelike/near-collinear and boundary cases with a distinct original delta-shell/spinor reference and selected isolated energy/phi/tail controls.

## Conditional moving-neutron decay/inverse, not a capture adapter

For decay the FUTURE lepton-pair momentum Q_leptons=pe+kanti=pn−pp=−q_Gamma, with ME²<=Q_leptons²=q_Gamma²<=(MN−MP)². The rest antinu endpoint is E0=[MN²−(MP+ME)²]/(2MN)=.7820077025790402MeV. It is not a lab emission cutoff.

For parent gamma,beta and rest antinu angle cstar,

    E_lab=gamma E_star(1+beta cstar),
    E_star=E_lab/[gamma(1+beta cstar)],
    dE_star delta(E_lab−...) -> 1/[gamma(1+beta cstar)].

At E_lab>E0 exact minimum parent rapidity is log(E_lab/E0), with pmin=MN sinh(log(E_lab/E0)); rest cone cstar>=max[-1,(E_lab/(gamma E0)−1)/beta]. Below E0 the exact cone switch p=MN sinh(log(E0/E_lab)) must be a panel boundary. Parent p=0 needs its finite limit or a quadrature endpoint policy with a derived limit; division by beta cannot certify emptiness. A distinct independent reduction integrates the delta over cstar, yielding dE_star/[2gamma beta E_star], with rapidity/original-momentum quadrature. Agreement of both is a required selected forward gate.

Use the electron–proton pair S_ep=MN²−2MN E_star, different from the lepton-pair invariant used by the total rest map. Stable threshold distance deltaS=2MN(E0−E_star) gives p_rel²=deltaS*(deltaS+4MPME)/(4S_ep). The exact rest electron-energy Jacobian is |dEe_star/dcos_zeta|=E_star p_rel/sqrt(S_ep). The spin-avg2 rest measure is Aavg2 dE_star dEe_star/(64pi³MN HBAR). Multiply spin-two parent p²dp/(pi²HC³), proper-time MN/En, isotropic rest-neutrino dcosstar/2, the lab-energy delta Jacobian, actual electron angle and azimuth. Boost daughters before BOTH direct occupation products. No scalar stationary neutron, mean gap, decay-endpoint cut or histogram replaces the conditional chain.

The inverse is the same covariant final-neutron phase with swapped material occupations and actual reciprocal current, not an independently postulated inverse proper lifetime. Literal decay/inverse spin sums must match before spectral integration. The resulting dR/dE already includes the neutrino phase; divide ONCE by D(E)=E²/[2pi²HC³] to obtain per-occupation j/chi. Do not multiply a second spin or neutrino DOS.

For E>E0 the allowed-parent density integrates p>=pmin and is smaller than the full neutron density. This is not an EOS-root error. Saved beta E=.8 positive spectra and above-endpoint moment fractions are mandatory witnesses, not evidence of a nuclear collision width or interacting DU threshold broadening.

## Spectral source, one-helicity DOS, length units and ledgers

For actual radiation occupation f of the correct flavor,

    df/dt = j(1−f) − chi f,
    relaxation = j+chi,
    R_F=integral D(E) j(E)(1−f)dE, R_B=integral D(E) chi(E)f dE,
    Q_F/B adds one factor E, net is accumulated directly/gross-scale checked.

D(E) has units fm^-3MeV^-1 and already integrates the isotropic neutrino solid angle for one active helicity. No extra4pi or2 is inserted. j is not volume emissivity and chi is not j+chi. Fixed-bath occupation relaxation uses j+chi, while true absorption of a test neutrino uses chi. Lengths: chi/c in cm^-1 with c=2.99792458e10cm/s; km^-1=(chi/c)*1e5. Log conversions preserve finite logs/ordinary underflow. This neither defines angular transport opacity nor licenses the label `transport` on a total attenuation coefficient. Transport requires an angular redistribution/contraction and its own test.

A local FD reconstruction uses Stage A's explicit independent common-T nue/anti chemical potentials; LTE mu_nue=zeta,mu_antinue=−zeta. The coefficients remain unchanged between empty/LTE/changed-affinity scenarios. Empty incoming radiation has f=0,B=0 for the SOURCE; chi is still computed and positive. A falsezero chi cannot be justified by empty radiation. Conversely multiply hole1−f exactly once, not twice by computing j with external blocking already inside.

Per net spectral EC event material(n,p,e_net)=(+1,−1,−1),nue+1; PC/decay material(−1,+1,+1),anti+1, with PC positron destruction distinct from decay electron creation. Bin source number and energy ledgers apply the SAME positive quadrature weights and actual f/holes. Rest-inclusive material loss is E times net neutrino number per energy; no extraQ or duplicate chemical-energy heating. Species-energy moments, if exposed, require new differential ledgers; neutrino j/chi alone cannot invent electron/proton spectral energy distribution. Full energy/species comparison uses independently qualified Stage A event totals. MaterialLe plus nue−anti conserves; abundance conversion additionally requires an explicit material nB outside this spectral API.

FD entropy per energy is D(E)*(F−B)log(F/B) when F/B are the actual per-occupation source branches j(1−f),chi f. It is checked against the correct affinity only for commonT. Exactly empty radiation does not give a finite initial radiation entropy derivative. Bin-level E/Le identities formed from the same E, rate and stoichiometry are algebraic bookkeeping, not independent physical closure evidence; use actual distinct phase/current and species-energy tests separately. A finite-bin discretization or passing total E/Le arithmetic is not a continuum spectral/transport convergence certificate.

## Zero, underflow, support and complete-result admission

Keep the total-moment PositiveQuantity semantics unchanged. New SpectralQuantity has a strict discriminated support state: `positive_support` requires a finite log, with positive ordinary value or None plus precise underflow/overflow reason; `exact_kinematic_zero` requires ordinary float0, logNone and an independently auditable empty-support witness; `unresolved` is an exception/incomplete diagnostic, never a success coefficient. NaN, Boolean, zero with finite log, nonfinite logs, handwaved physicalzero and unsupported mass/current/bath all reject.

A differential phase point can have genuine exact empty support; that does not authorize an integrated zero. For the fixed finite-T particle baths and E>0, full integrated spectral channels are positive even when deep tails are unrepresentable ordinarily. A finite numerical truncation or unresolved support search may not certify exact empty. Complete positive j AND chi are required in integrated rows; preserve finite log with ordinaryNone. Exact zero of an inverse SOURCE from empty incoming radiation has a distinct absent-occupation reason, not a coefficient-support reason.

The saved extreme-soft PC log≈−13836 with ordinaryNone is representation evidence only, not a relative-accuracy qualification. Do not insert floors, physical widths or drop the soft band. No arbitrary `max(positive,tiny)` or fallback to j=chi*KMS. If model/source conditioning or quadrature cannot resolve a positive channel, raise and preserve the partial diagnostics. Hard wall/bounded memory checks cannot turn incomplete jobs into zeros.

Each reconstruction call names an explicit finite energy interval/node rule and returns FINITE-RANGE moments with that coverage, not an automatically complete infinite-domain channel. A whole selected finite-domain comparison may use preregistered disjoint intervals, each computed freshly within120s; the independent test harness verifies exact coverage/state/current/control identity before summing. An interrupted interval is not admitted. If bounded work cannot close the full-domain/tail gate, freeze selected spectra/range pilots with whole-moment reconstruction OPEN, not a fabricated complete result. First reconstruction should operate directly in memory without cache/resume. If future multi-process collection is required, consume the same bytes that are hashed, requiring exact bath/model/source/control/flavor/energy-node/radiation/units identity and complete disjoint node/weight coverage before summation. No cache keyed only by flavor/order, partial sum labelled complete, dynamic source registry or arbitrary client `qualified=True`. Physical per-node positivity is validated before cumulative cancellation; reject changed one/both j/chi branches even if totals are redistributed to match.

## Saved evidence, preserved FAILs and implementation/review gates

SAVED_EVIDENCE.json copies ten actual fine spectral rows, three full fine R/Q scenario groups, exact reconstruction controls and the PC moving-threshold witness, with source JSON pointers/hashes. No new EOS/rate/current calls are performed here. The accepted independent reviews, source notes, raw totals and repairs are pinned; stale author-note review-pending wording does not supersede the later reviews.

Mandatory future gates, LOCKED before first candidate calculation:

1. Exact accepted prerequisite Stage A source/base identity, physical masses/current/reciprocal signs, no campaign imports/generic aliases/import-time IO, cold and total API regression/strict zero semantics unchanged. Source/type/temperature/current/units mismatches reject before ratecalls.
2. Selected j/chi current/phase absolute agreement at saved and NEW independently preregistered same-bath points, direct branches before KMS. Capture longitudinal-shell/spinor versus q-axis polynomial; decay independent delta reduction/trace versus conditional rest-current. Include EC timelike and PC spacelike/moving-below-rest-threshold and beta E=.8>E0.
3. Conditional support polynomial/linear/near-degenerate/collinear branches, actual threshold/upper-tail/panel resolution and no unresolved-as-empty. Positive contractions and physical on-shell residuals precede differentiation or source claims.
4. Separate baryon/parent, lepton/cone/electron, azimuth, tail and incident-energy refinement. Initial selected resolved relative gate3e−6, literal-current/phase selected gate3e−10 where well conditioned, absolute gross-scale gates near cancellation; exact values/conditioning/admission/rules fixed before future data. No blind generic nearzero relative tolerance.
5. Reconstruct positive F/B R/Q for empty,LTE,and both affinity directions from actual node spectra and one-helicity DOS; compare independently integrated Stage A event totals at exactly the same chemical bath, radiation and current. Different channel scenario labels are not enough: exact mu_nue/mu_anti and T must match. In the SAVED data, PC `noneq` uses mu_anti=mu_eq−.8MeV while beta `emission_dominated` uses mu_anti=mu_eq−1.6MeV; these rows MUST NOT be summed as a common-radiation scenario merely because both favor emission. Keep F/B nonzero LTE branches, independently gross-normalized net/energy/Le/entropy. Direct reconstruction is a new physical calculation requiring its own budget, not a replay of saved rows.
6. Saved capture incident ranges reach60MeV; beta1.4MeV, with actual nonzero tails beyond each. Isolate tail extensions or independently bound the omitted effect at selected states. Coupled material/incident coarse-fine comparisons are labelled combined, not a separately proven bound. Beta above E0 contributes saved1.033801%ofnumber/1.748314%ofenergy at this bath: a rest-endpoint lab cut must fail. No continuum or uniform-domain certificate.
7. Original PC coarse4-node-per-panel reconstruction10FAIL (3e−6 gates), initial PC121PASS/1falsezero admissionFAIL and subsequent DIFFERENT568PASS repair remain distinct immutable records. Original cold capture failure/mapped-edge repair, strict-wrapper underflow exclusion, beta reviewer initial entropy/reporting/panel warnings and fit admission/model failures remain historical, not relabelled PASS. Passing shared KMS or moment arithmetic cannot erase them.
8. Preserve the Stage A nonzero-sign/None-log finding as a regression lesson: a helper for absent incoming occupation must never admit a claimed positive-support coefficient without a finite log. Initial1378PASS/2FAIL and separate417PASS repair review remain pinned, not rewritten as a single passing original record. New adversaries: missing/stale/incomplete/source-swapped/node-overlapping/control/radiation/bath mismatch, finite-log-underflow versus exactzero, falseempty both branches, changed aggregate preserving totals, nonfinite/Boolean/huge inputs, singular coordinate, wall timeout and incomplete reconstruction. Preserve original failed candidate/source before any explicitly authorized repair; recheck by a DIFFERENT agent.
9. Separate independent physics/numerical design review first, then explicit implementation authorization; final different numerical and packaging reviews and normal CI/minimum runtime. No derivative/AD, full interacting opacity, complete network, calibrated uncertainty or performance claim follows from this contract.

Optional positive EC/PC total-moment fits remain a different later layer with their own prequalified domain/source/heldout admission and actual detailed fallback. No spectral fit, signed net fit, interpolation-as-kernel or rate replacement is authorized here.
