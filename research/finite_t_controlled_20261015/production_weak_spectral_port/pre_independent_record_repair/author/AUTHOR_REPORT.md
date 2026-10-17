# Stage B free charged-current CPU spectra: frozen candidate

The candidate adds actual direct per-occupation j and TRUE absorption chi for
EC, PC and NEUTRON_DECAY to the reviewed Stage A package. The scope is one
EXACT chemical bath T2,mu_n919.8287578100662,mu_p916.8449303084382,
mu_e.17446333200826397MeV. It solves no EOS/neutrality/beta condition.
The material is free physical-mass particle n/p with physical e± and the
unchanged full Guo VA/F2/GP current. Separate branches integrate actual FD
products, not a KMS-manufactured inverse. This is a selected reference
candidate awaiting DIFFERENT numerical/physics and packaging reviews.

The real package copy is based on PR54 head
c17472b8a84655380ab64973ef883a6cd5fb2346. All11 accepted Stage A file hashes
match that base;71 original package source Gitblobs also match. The modified
paths are exactly the11 accepted Stage B paths in CANDIDATE_MANIFEST.json.
The four Stage A physical helpers, existing tests and fixture remain byte
identical. All existing public total classes/functions are AST-identical;
only the prior spectral stub and module discovery/doc text change.

The immutable numerical LOCK c051e3c23f290b657b4caad069a34d4ce7ee85a4e535368fb674bd5590ec188a
predates validation. Root explicitly approved SpectralControls opt-in:
controls=None preserves the legacy spectral-unavailable NotImplementedError;
explicit controls evaluate the new API. This compatibility choice differs
from the historical proposed default signature, without changing any total
constructor/API. The accepted design and initial/repaired design reviews
remain unchanged. Additional isolated-control/range protocols were locked
before those calls. Ninety-five original design source pins and the extended
source inventory remain unchanged; no mutable campaign loader enters runtime.

## Physical implementation

Capture directly reduces the original positive-energy baryon polar delta
in q-axis coordinates, retaining spacelike, timelike and exact-linear support.
Physical neutron/proton ordering is passed to the unchanged avg4 scalar
current. The coefficient measure is pe/(32pi³qE HBAR) integral dEa<A>phi.
Actual El=ME+y² lepton quadrature resolves recoil/thermal edges; both material
occupation products are separately log-summed. A forbidden differential
point has an analytic-support witness; integrated finite-T j AND chi always
require positive finite logs.

Conditional decay uses the electron–proton invariant pair at fixed lab
antinu energy, an actual parent distribution, exact boosted rest cone,
energy-delta Jacobian, proper-time factor and boosted proton/electron holes.
Its avg2 full current and spin2 parent measure reproduce the source
normalization. The conditional dR/dE is divided once by one-helicity
D(E)=E²/(2pi²HC³). The lab spectrum is not cut at neutron-rest E0.
q_Gamma=pp−pn throughout; future decay lepton-pair Q=−q_Gamma; reciprocal
orientation and physical q² are unchanged from Stage A.

SpectralQuantity is distinct from total PositiveQuantity/SignedQuantity.
A positive coefficient retains a finite log with ordinaryNone under/overflow.
Exact differential empty support has a separate discriminant. Unresolved
coordinates, collapsed weighted panels and timeout/incomplete work raise.
There is NO width-only, map-collapse or ordinary-underflow panel skipping.
Relaxation=j+chi is a log-safe sum; true absorption length uses chi/c in
cm^-1 and chi/c*1e5 in km^-1. Neither is angular transport opacity.

## Evidence and numerical limits

Unchanged Stage A40current/moment tests ran before spectral interpretation.
The final real-package test suite has86PASS:40Stage A,13cold/unit and33new
spectral/reconstruction tests, no skips, in20.841s. An additional selected
numerical/saved-node arithmetic assessment has250PASS. These overlapping
checks are not independent physical models or empirical error estimates.

Ten saved source-labelled spectra at EC/PC2,5,10 and beta.2,.42,.5,.8MeV
pass the locked3e-6 absolute-normalization and base/fine numerical gates.
Maximum selected fine-vs-saved relative difference1.912e-11; maximum saved
base/fine difference4.228e-9. Seven NEW predeclared incident energies
EC3.7/7.3, PC3.3/8.2, beta.31/.63/.83 pass the same profile gate, maximum
7.797e-8. Both direct branches are positive and their posterior KMS residuals
pass2e-9. Those checks do not establish uniform accuracy over all admitted
energy/control requests.

An independently assembled longitudinal-delta/transverse-momentum integral
with literal external spin sums, instead of the q-axis scalar reduction,
agrees at fresh EC(4.3,5.59,.27) and PC(1.73,.62,-.24) differential points
within1.075e-13. Its own refinement is5.33e-15. Literal current gates also
cover fresh timelike EC(4.3,5.59,.9999). The original moving-target PC1.7MeV
witness remains positive below the stationary1.806066208MeV proton threshold.
Beta.83MeV has j=.0005581834678717144s^-1 and
chi=.0002074739146003905s^-1, above E0=.782007703MeV. Isolated tail, outer
lepton, phi and individual-order variations at EC7.3 and beta.83 pass3e-6;
max isolated selected change2.835e-9. Independent DIFFERENT conditional-phase
review is still required; shared-source agreement is numerical evidence,
not model uncertainty.

Eighteen fresh finite-range pilots use2/4incident nodes for EC/PC2–6MeV and
beta.2–.8MeV, with empty/LTE/common changed-affinity radiation. All actual
nodes, state, controls, radiation and coverage are retained; the pilot
consumes no cached/external coefficients. Direct F/B R/Q, signed nets and
FD entropy reproduce separately assembled saved-node DOS/source arithmetic.
Empty radiation has zero inverse SOURCE but positive chi, and undefined
initial radiation entropy derivative. No extraQ is added to matter energy.

IMPORTANT: whole-moment reconstruction/tails remain OPEN. All returns say
whole_moment_qualified=False. The2→4incident-node forward R/Q sensitivities
reach0.34%EC,1.71%PC,5.29%beta in the finite ranges. These are pilots with
nonzero omitted ranges, not3e-6-converged total R/Q or a continuum certificate.
No Stage A whole-total agreement is claimed for them. Their bin E/Le/entropy
identities are algebraic bookkeeping, not independent species-energy closure.

## Preserved failures and narrow pre-handoff repairs

- Initial pytest without owned PYTHONPATH failed collection by finding an old
  editable checkout. Corrected execution explicitly selects the real owned
  package. The import failure is retained; no candidate physics ran then.
- Initial spectral tests32PASS/1FAIL: an empty-support test incorrectly assumed
  PC E5,El.6,c1 was forbidden. It is spacelike with a positive thermal tail.
  Original test/output are preserved. Only the fixture changed to analytically
  forbidden past-timelike PC E2,El3,c1; no kernel/tolerance changed.
- A malformed FD bath raised AttributeError before strict state validation:
  initial1PASS/1FAIL preserved with all11source files. Existing validation was
  moved above temperature access; valid physics outputs are unchanged.
- ReconstructionResult initially admitted missing nodes and a forged whole-
  moment flag:2FAIL preserved with all11source files. Added strict completed
  node/state/control/radiation/branch/scope/units guards. All18actual saved
  pilot payloads remain identical;108negative constructor checks pass.
- A reporting-only assessment glob also matched a contract-validation JSON
  list. Initial script/error are preserved; explicit18path enumeration fixes
  the report. No physical source or raw data changed.

Original campaign coarse/cold/guard/underflow and fit failures, and Stage A's
initial signed-log finding plus separate repair, remain immutable upstream
history. No gate/profile was relaxed and no model was rescaled.

## Readiness

No remote/Git/production-checkout/CI/dependency changes occurred. Import-time
runtime code has no file IO, campaign aliases, loaders, cache or scientific
integration. Selected real module origins and Python3.9 syntax are checked.
The executed environment is Python3.12.12/NumPy2.3.5/SciPy1.15.2; the NumPy
version lies outside the separate campaign validation pin<2.3. Actual minimum
runtime, full CI, clean install/wheel review and DIFFERENT numerical acceptance
remain OPEN. No EOS/RMF/AD, arbitrary-bath spectral domain, transport kernel,
full inventory, fit replacement, external calibration or empirical uncertainty
claim is made. Source is frozen for independent review; no further edits
without preserved findings and explicit narrow authorization.
