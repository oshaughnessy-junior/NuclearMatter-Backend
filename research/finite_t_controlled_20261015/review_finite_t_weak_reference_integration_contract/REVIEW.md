# Independent weak-reference integration-design review

**ACCEPTED as a scoped staged design after the transfer-notation repair.** Initial review: **168 PASS / 1 required clarification**, preserved unchanged. Independent repair recheck: **134 PASS / 0 FAIL**. No physical rate, EOS, amplitude, native-library, install or production calculation was performed; these are source, saved-data, kinematic, integer-ledger and provenance checks.

Accepted repaired CONTRACT `60adb59adecb0d0bd827c25a10cde117e12a79a7bb77ec7aeedd6d77659c22c4`, MANIFEST `fae2e5f7cbb12f613569112549ddc3e6efce9b39c6514c0d067aae9cd3dae983`. Original CONTRACT `bb836077dbca372d0e231958d37788cc66b24266811522ee91751cae552fb370` and every original author/reviewer file remain pinned in the independent snapshot and INITIAL_CHECKPOINT. All **60 scientific/review/evidence source pins remain unchanged**. Initial SOURCE_INVENTORY, SAVED_REGRESSIONS, AUDIT_RESULTS and REPORT bytes are unchanged. The repair alters only current/phase-transfer labels and their explanation, plus repair provenance; no source equation, matrix element, rate, physical number, gate or proposed Stage A file scope changed.

## Resolved finding

The initial prose reused q both for the n→p current argument p_p−p_n and the future decay pair p_e+k_antinu=p_n−p_p. The frozen scientific beta implementation already used the correct sign. This was a design ambiguity before porting, **not a scientific source bug**.

The repaired text explicitly distinguishes

`q_Gamma=p_p−p_n`, `q_pn=−q_Gamma`,

`Gamma_pn(q_pn)=DiracAdjoint[Gamma_np(−q_pn)]`,

`Q_leptons=p_e−+k_antinue=p_n−p_p=−q_Gamma`.

Its six-reaction table correctly orients EC/PC emission and inverse absorption, and neutron decay/inverse. Capture geometry Q_abs and its positive spatial magnitude q are separately named. Decay support is future Q_leptons; the n→p current argument is past-directed there. Equal invariant squares cannot justify interchanging linear F2/GP arguments. Free physical/kinetic momenta coincide, while the later Hartree adapter must preserve orientation before distinguishing qphysical/qstar.

A standalone physical on-shell neutron-rest decay witness proves q_Gamma=−Q_leptons and equal squares, with opposite time components ±1.292831216681 MeV and q²=.730994726741 MeV². It evaluates no amplitude/rate. Independent diff inspection and a normalized vertex-block comparison verify that the repaired displayed vertex only renames its argument. The initial 168/1 finding remains failed in its original report; the new report accepts the repair separately.

## Stage A scope and physical accounting

The eleven-path first stage is a standalone **explicit-chemical** free electron charged-current total-moment reference. It must calculate separately positive forward-emission and inverse branches for EC, PC and neutron decay, with independently evaluated FD products and separately accumulated signed net number/energy moments. It is not an emission-only convenience wrapper. It does not imply spectral opacity, complete neutrino/Urca inventory, EOS/beta/neutrality factory, muon physics or medium interactions. A spectral request before Stage B must fail explicitly.

The fixed source model uses MN939.56542052, MP938.27208816, ME.51099895069 MeV, Mweak=(MN+MP)/2, hbarc197.3269804, GF1.1663785e-11, Vud.974 and the specified Guo full-current form factors. These differ slightly from existing free-utility NMB mass defaults and markedly from common939 GM1; neither substitution nor an extra Q term is admissible. Point-VA/stationary/allowed limits are named different models, not replacements for the default full physical phase calculation. Tree-level plane waves omit radiative/Coulomb/medium/pairing corrections; vacuum beta lifetime is a model/normalization anchor, not measured-lifetime calibration or a model-uncertainty certificate.

Per-spin baryon occupations are particle-only; electron and positron occupations have chemical potentials ±mu_e. No hidden antiparticle density or neutrality condition appears. A future matched factory must distinguish baryon particle density, electron NET charge, separate e± populations and explicitly absent muons. First explicit-chemical kernels must not depend on unmerged EOS drafts or campaign loader state.

Exactly empty incoming radiation differs from a finite common-T FD distribution. The two radiation species have separately specified occupations; LTE requires mu_nue=zeta=mu_p+mu_e−mu_n and mu_antinue=−zeta, not merely equal temperatures. Both F/B branches remain positive at LTE even if their net cancels. A transparent net-composition root is not chemical or thermal equilibrium. EC consumes one electron, PC consumes one positron, and decay creates one electron; their NET charge accounting cannot replace their distinct particle populations. The reaction table conserves baryon number, electric charge and electron-family Le, including the emitted neutrino sign.

Rest-inclusive n/p/e energy moments must reconstruct material loss against escaping or thermal/gross source scales, without a 939-MeV denominator hiding cancellation. For common-T FD radiation the per-event entropy ratio reduces to the constant affinity, but direct F/B integration remains required. Empty radiation does not have a finite initial entropy derivative. Material entropy may decrease; escaping moments alone cannot establish total radiation entropy.

## Normalization, units, support and saved evidence

The unaveraged covariant master measure has no extra helicity or initial-spin average. The existing EC/PC avg-four amplitude multiplies by four when used with that measure; decay's avg-two amplitude combines with the spin-two parent density. Independently combining fixed-neutrino leg, shell/lepton phase and azimuth gives the displayed avg-four prefactor pe/(32*pi³*q*Enu*hbar), with no missing 2, 2pi or target-spin factor.

Spectral one-helicity DOS is E²/(2*pi²*hbarc³). True chi and j+chi relaxation are distinct s^-1 coefficients; inverse length divides by c, cm^-1 converts to km^-1 by 1e5, and neither is automatically a transport opacity. Stage A total inverse moments depend on the actual radiation distribution; they are not an energy-independent opacity. Differential capture and decay conditional-energy APIs must remain distinct.

Exact unequal-mass support, moving targets/parents, final-state blocking, endpoint maps and separate log-positive accumulation precede refinement. Rest-target antineutrino capture threshold and neutron-rest antineutrino endpoint arithmetic match the pinned values, but neither is a global finite-T/lab cutoff. The PC/EC transfer ordering is explicitly different. q=0 is a coordinate singularity requiring a derived limit or unresolved-coordinate error, not a zero coefficient. Widths or sigmoid broadening are not quadrature controls.

Ten copied spectral rows and three full total F/B groups exactly match source hashes and JSON pointers. Saved KMS, relaxation, LTE nonzero balance and exactly empty inverse branches pass independent arithmetic checks. These saved source results are regression fixtures and previously qualified reference evidence, not new independent physical computations. In particular their whole groups retain empty, LTE and nonequilibrium forward/inverse moments, rather than only positive emission totals.

Representation must distinguish finite-log ordinary underflow, genuine empty support/occupation and unresolved integration. A completely positive finite-T branch cannot be assigned zero merely because ordinary nodes underflow. A signed net near LTE needs cancellation-aware accumulation and gross/absolute gates. Wall-time/partial grids must not become successful physical zeros. Log validity is not a blanket relative-accuracy guarantee for extreme tails.

## Dependency and approximation boundaries

The pinned inventory honestly records 29 modules' static dependencies and 60 source/evidence hashes; it is not an executed complete import closure. The port must remove generic module aliases, sys.path changes, mutable current/mass globals, file/report hash reachbacks, process dont_write_bytecode changes, driver execution and reviewer-directory runtime imports. A reviewer scalar polynomial should remain an independent test oracle, not become both runtime and its own test. Immutable explicit records must fingerprint actual model/state/radiation/controls, without a mutable JSON qualified flag. NumPy/SciPy APIs must match actual supported minimums.

Detailed quadrature remains default. Any later positive EC/PC correction fit needs its frozen basis/coefficients, exact source and complete passing holdout provenance; its selected positive-moment accuracy does not certify inverse FD closures, spectra, signed nets, roots, derivatives or a different EOS/current family. Near source cancellation the actual physical quadrature must run. Stationary allowed-model errors and old signed scattering-fit failures are model/approximation evidence, not numerical uncertainty in this port.

The Stage B informal five-path estimate should be replaced by a literal changed-path list before that later batch, counting public dispatch if distinct from tests/fixtures/docs. This is a nonblocking future scoping advisory; the Stage A eleven-path design is clear and unchanged by the repair.

## Gates still open

No implementation is accepted by this review. A future candidate still needs preregistered numerical request/domain/model guards, full-current literal/invariant and reciprocal-crossing anchors, fresh forward/reference density checks, separately resolved F/B number/energy and isolated tail controls, direct energy/Le/entropy ledgers, negative/underflow/timeout/incomplete tests, different physics/numerical review and separate package/runtime/CI review. Stage B independently reconstructs actual event totals. Later EOS/RMF binders require their own exact model, stable fields, physical/kinetic transfer and thermodynamic consistency gates.

Native external agreement, empirical uncertainty, interacting/paired/modified Urca, muon inventory, full transport/hydro, long-time evolution and AD remain open. The preserved original cold quadrature, strict-wrapper underflow, coarse grids, false-zero result admission, fit qualification and stationary-model failures must accompany future claims; none are erased by a design approval.

## Reproduction and preservation

`audit_initial.py` is intentionally tied to the immutable initial contract and retains 168/1. `recheck.py` separately verifies the repaired contract, all six orientations, old vertex after argument renaming, twelve original author snapshots, original reviewer artifacts, all sixty source pins and unchanged saved-data bytes; it reports134/0. Both run in less than a second using existing junior_tools Python with thread1, no scientific-module imports. Exact baseline quota/account/reset checked at05:10:01 and05:14:35UTC:27%. No installs, physical rate calls, production/git/PR/paper edits or source changes occurred in this independent review.
