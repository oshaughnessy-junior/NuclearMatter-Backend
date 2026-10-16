# Wider finite spectral bands versus fresh event totals

Frozen author research diagnostic, 2026-10-16. DIFFERENT review is required.
No infinite-energy, uniform inner-error, or complete-source qualification follows.
`whole_moment_qualified=False` throughout.

## Outcome

| Channel | Actual band nodes | Positive band/event R,Q maximum relative difference | Disposition |
|---|---:|---:|---|
| Electron capture (EC) | 160 complete | 4.41242e-8 | Selected finite-band numerical agreement at locked controls |
| Positron capture (PC) | 0 complete; 2 failed first nodes | unavailable | Incomplete; no band sum admitted |
| Neutron decay | 160 complete | 1.16231e-8 | Diagnostic only: upper-panel and inner-tail controls fail |

The complete evidence has **188 passing numerical/algebra checks, 12 failed
numerical checks, and 6 blocked assessments**. These counts are not independent
physical models or an error certificate. Electron-capture panel and band incident
12-to-20 changes are below 3.8e-13. Its selected inner-order/tail controls pass.
The event total is a separately evaluated event-phase integral sharing the same
fixed physical current, not an external code or empirical calibration.

PC fails the frozen positive-energy baryon-shell resolver at E=0.1 MeV for
fine, coupled-higher-order, and isolated-inner-tail controls. Its first actual
band nodes, E=0.06797838160944875 (order12) and 0.05669961579453249 MeV
(order20), fail with `ArithmeticError: Final positive-energy baryon shell unresolved`.
Both order paths stop immediately. All ten panel/order records identify their
missing nodes; no partial band, substituted zero, skipped support, or inferred
small weighted remainder is admitted. The 6 blocked assessments comprise three
unresolved controls and three radiation scenarios lacking complete PC bands.

Neutron decay has complete finite-band node data and close aggregate event
agreement, but its [1.1,1.4] MeV panel changes by up to 1.97779e-5 between
incident orders12 and20, failing ten positive F/B number/energy checks. At
E=1.2 MeV, the isolated parent-tail change70-to-90 at fixed fine orders changes
both j and chi by 2.23776e-4, failing two more checks. Coupled higher orders at
fixed tail change them by 9.93796e-7 and pass. Domain extension at fixed Gauss
order moves nodes, so this is numerical sensitivity, not a pure omitted-tail
estimate. Neither small absolute values nor aggregate cancellation waives a
failed gate. A full weighted remainder bound has not been established.

## Locked model and actual execution

LOCK.json and SOURCE_PIN.json were written before new physical outputs. The
source is the repaired accepted Stage B candidate dcfbafe85645caa4ac6f2ea1f4cd3a7cb55775d030eddc0c9c5a8c9211a8b216,
published PR55 head440094516498eb445ab5de2e6ebf89b361fea212. Imports use its
read-only real package mirror. All16 pinned source/review/provenance files were
verified before/after calls and again at freeze. No source or tolerance repair,
profile expansion, code mutation, or post-data gate adjustment occurred.

The explicit chemical bath is T=2 MeV, mu_n=919.8287578100662,
mu_p=916.8449303084382, mu_e=0.17446333200826397 MeV. This study does not solve
neutrality or an EOS. Nucleons are particle-only; e-minus and e-plus use opposite
chemical potentials; muons are absent. Fixed physical masses are
MN=939.56542052, MP=938.27208816, ME=0.51099895069 MeV,
hbar*c=197.3269804 MeV fm. Full source-defined Guo VA/F2/GP form factors and
bare weak mass (MN+MP)/2 are unchanged. For the n-to-p current qGamma=pp-pn;
future decay leptonic transfer is its negative. Reciprocal currents retain the
accepted adjoint and transfer conventions.

Capture panels are [.05,2,6,15,30,60] MeV; neutron-decay panels are
[.001,.2,.5,.8,1.1,1.4] MeV. Each uses incident Gauss orders12 and20 only.
Fixed fine spectral controls are lepton12/angle16/energy16/phi6, inner tail70,
outer tail40, parent12/cone16/electron16/decay-phi10, per-call budget110 seconds.
Predeclared spot checks at capture .1/30/55 and decay .01/1.2 MeV compare one
coupled higher-order profile; selected isolated tail changes are separately
labelled in LOCK.json. All21 spot calls were attempted:18 completed,3 unresolved.
Band calls attempted322 coefficient evaluations,320 complete and2 unresolved,
below the locked480 maximum. No E=0 or out-of-admission request was made.

Every node stores actual energy, positive quadrature weight, separate direct j
and chi finite logs, controls, bath, support/representation, and source identity.
Coefficients are independently evaluated before KMS checks; no inverse branch
is manufactured from detailed balance. Finite logs with ordinary valueNone
represent under/overflow, not exact zero. The own postprocessor validates the
actual node/rule metadata and assembles F=j(1-f), B=chi*f, using the one-active-
helicity DOS E^2/[2*pi^2*(hbar*c)^3]. Result units are fm^-3 s^-1 for number and
MeV fm^-3 s^-1 for neutrino energy. This saved-data harness is explicitly not a
public complete-source API record or an interpolation/cache replacement.

The same nodes are reused only for three declared radiation evaluations:
empty incoming radiation; common-T LTE with mu_nue=-2.809364169619812,
mu_antinue=+2.809364169619812; and common changed affinity with
mu_nue=-3.6093641696198118, mu_antinue=2.009364169619812 MeV. Both flavors are
shifted by the same -0.8 MeV; old unequal nonequilibrium labels are never mixed.
All9 corresponding Stage A fine event totals were freshly computed. They retain
actual forward/backward number, neutrino and species-energy moments and controls;
no saved fixed-bath moment replaced any source evaluation. Empty inverse source
is the structural absence of incoming neutrinos, not a zero opacity.

## Conservation and limits

Separate positive F/B R and Q are compared first at the prelocked3e-6 gates.
Signed differences use gross F+B scales; no cancelled net or nucleon rest-energy
denominator is used. Actual event species-energy and direct material-loss
ledgers agree with escaping neutrino energy. Charge, baryon, and electron-lepton
stoichiometry are explicit; rest masses enter once and no additional Q is added.
LTE/gross and finite-bin affinity/entropy checks pass where complete. These
arithmetic identities are not independent species-physics validation. Empty
radiation has no claimed radiation entropy derivative.

EC empty-band order20 R=3.3088602022466086e-7 and
Q=3.160384777048211e-6 versus fresh event R=3.308860348247349e-7,
Q=3.1603848367506778e-6 in the units above. This is selected finite-band
agreement only. Nonzero missing intervals E<.05 and E>60 MeV for captures,
and E<.001 and E>1.4 MeV for decay, remain OPEN. No rigorous bound covers all
omitted tails or uniform inner errors. Beta's failed local controls remain
unqualified even though its integrated band/event differences are small.

All scientific subprocesses were serial, thread-limited, at most4 band nodes
per job, and individually below120 seconds. Science finished before08:26 UTC;
no further physical calls are planned. The environment was Python3.12.12,
NumPy2.3.5, SciPy1.15.2; minimum-runtime and CI qualification are not claimed.
The NumPy environment is outside the separate validation pin<2.3. No model
uncertainty, EOS/AD, transport closure, opacity inventory, evolution, full Urca
inventory, or infinite-band source certificate follows. Frozen current/raw
failures are the next numerical-support planning evidence, not grounds to
silently repair geometry or loosen gates in this batch.

Two saved-only final-freeze harness errors (wrong completeness-key name and len of an integer count) are preserved with exact harness snapshots. They occurred after all scientific results; corrections only read existing evidence and regenerate final metadata. No physical call, failed gate, source, rule, or raw data was changed.
