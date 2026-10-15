# Independent v2 joint lepton-scattering fit review

**Accept the selected fixed-state moment representation and actual cancellation fallback.** This review adds69 passing checks:64 selection/admission/fresh-physics controls and5 copied-fixture provenance guards. No numerical failed gate was found. It establishes neither uniform sign/error bounds nor a spectral redistribution kernel, derivatives, arbitrary bath-state fit, or physical-model uncertainty. V1 coefficients, two failed signed gates and the prior reviewer wrong-sign example remain unchanged.

The pinned v2 author manifest is `3d914218212a3e12ef62266f28b0a48c325d284379bb24e334b1de2ec544c01d`, API `edf38db39a3776965c97e36592fc0a95ab4e95110f35002e2bc45f2794edc658`, model `f881ee6f9733fef2120fe5c75cb38f7a8e44bd85f88bba4d3d80a829c690a6f0`. All131 pinned author/v1/reviewer/physical-source files match before and after this review. No author driver was imported or executed; new reviewer scripts write only here.

## Selection and contract audit

The independent audit re-read the21 declared development rows, reconstructed logR, logRtransport and signed g=DeltaQ/(2R), and recomputed all degree2–8 training/leave-one-out scores. The smallest passing choices are exactly4,5,6, giving18 coefficients; their coefficients reproduce within2e−13. These computations audit selection without refitting or modifying the locked model. The lock hashes match actual model/training/preregistration bytes, and the eight declared fresh author validation energies are disjoint from training. The artifact explicitly records locking before those validation calculations; this review confirms that consistent record and refusal logic, not independently timestamped third-party chronology. All v1 and prior reviewer points are development for v2.

The representation derives both energy moments from one R,g pair: `Qout=R(E+2g)`, `DeltaQ=2Rg`. There is no log-absolute fit across zero. Positive1%, signed absolute.005RrefT and relative2% only for |gref|>.25 are the unchanged preregistered gates. No2% relative claim is applied arbitrarily near zero.

The physical domain is incident nu_e energy1–30MeV, T2MeV, total mu_e.17446333200826397MeV, BOTH FD e−/e+ targets, final charged-lepton holes, unblocked final neutrino radiation. R and transport R are s^-1 per incident active-helicity neutrino; Qout and DeltaQ are MeV/s. Positive DeltaQ heats radiation and removes that energy from the fixed material bath. The pair populations and weak-current model remain the previously reviewed source, not a net-charge target substitution. [Guo etal Eq2 and the leptonic-current reduction](https://arxiv.org/html/2006.12051v2) use target FD and hole factors with the target spin convention; the shell audit retains that model, GF and electron-flavor coupling, with no Vud. Fit residuals do not estimate QED/correlation/current uncertainty.

Twelve fresh type/domain/state/closure checks reject Boolean or NumPy-scalar E, nonfinite E, extrapolation, changed T/mu, list-valued targets, wrong flavor/radiation closure and unsupported method. Both endpoints and separate interior points satisfy the joint arithmetic identity. This is a targeted review, not a replay of all624 author mathematical checks.

## Fresh physical heldouts near zero

Before evaluating new physics the reviewer selected8.1137 and8.1293MeV; neither appears in author/v1/reviewer physical input records searched here. Both default API calls execute actual fine CM event quadrature with controls32/40/40/48 and target tail80T. They load no saved reference row. The source loop runs separately for e− and e+ and reports actual elapsed time.

The independent implementation `energy_shell.py` evaluates the on-shell charged-lepton **energy** integral directly with adaptive quadrature and analytically averages target azimuth. This differs from the CM event reduction and from the author's transverse-momentum integration. The target lower support is `(-omega+q sqrt(1+4m²/a))/2`, where a=2EF(1−cos(theta)); pz=(2e omega−a)/(2q) and r²=e²−m²−pz². The phi average supplies the two invariant-square terms plus the electron-mass interference term. The prefactor remains F/(64pi³Eq hbar) times16GF², including the outgoing one-helicity phase space; no second outgoing DOS is added. Both target signs are integrated. This is numerical reduction independence with a shared weak model, not an independent physical uncertainty estimate.

Outer rules14/24 at Eout cutoff150MeV and tail70T are compared with20/34 at cutoff180MeV and tail90T, using different Eout panels from the author's shell anchor. All selected refinement gates pass. These simultaneous changes demonstrate bounded selected convergence; they are not separate universal tail and angular-resolution certificates.

| New Ein [MeV] | API actual DeltaQ [MeV/s] | Independent shell DeltaQ [MeV/s] | Raw fit DeltaQ [MeV/s] |
|---|---:|---:|---:|
|8.1137|+.00317688307885|+.00317688129785|+.00214304206886|
|8.1293|−.0676947019789|−.0676947037591|−.0686640045965|

Positive moments agree between shell and actual CM to8.44e−10 relative; signed transfer agrees to8.80e−11RrefT. The physical signs are resolved at both points, including the small positive heldout. The raw fit's near-zero relative DeltaQ error at8.1137 is about32.5%, yet only5.11e−5RrefT absolute: it passes the declared absolute gate while clearly motivating actual fallback. This is not a failed predeclared2% gate because that relative gate excludes the cancellation region. The API returns actual quadrature there, not this raw diagnostic approximation. A uniformly correct raw-fit zero/sign location is not certified.

## Fallback boundary and provenance

The locked fit's g=+.08 boundary is E7.756583763305099MeV. At E+1e−7 the default API invokes actual fresh fine CM quadrature; at E−1e−7 it stays on the fit path. The inside boundary reference gives DeltaQ1.54724287131MeV/s versus raw fit1.54485694708. Positive errors are at most.06348%, and signed error1.23479e−4RrefT, both within unchanged gates. The symmetric negative-g boundary is mathematically located at8.471627281739618MeV but was not given another physical integration here. The fallback condition is explicitly inclusive `abs(g)<=.08`; numerical tests bracket its positive boundary rather than claiming an exactly representable root.

The API does not infer qualification from mutable model metadata alone: its literal model hash and transitive physical file pins are checked before evaluation, and the detailed adapter has its own literal hash. In reviewer-owned copies, a stale model and stale fallback adapter each reject before physical invocation. Actual fresh-process module origins are verified as the pinned scatter.py, pair.py and events.py. These synthetic guard tests do not mutate the author's payload or physical sources. The copied `reference.py` in this review directory is deliberately marked stale and is a negative fixture, not a new physical reference.

Actual calls were made in a fresh subprocess with clean physical module names. The frozen source uses some ordinary cached module imports; this review does not certify arbitrary parent-process module-cache isolation. General integration should use the campaign's process-isolation contract or an independently validated loader. That is separate from the selected clean-process physical fallback checked here.

The main bounded experiment took30.17s; each new actual close-zero CM call took about8.8s, while independent coarse/fine shell calls took.59/1.37s. Pin guards required no new moment calculation. No full624-check replay, coefficient change, fit-point relabelling, dependency install, or production mutation was performed. Run `probe.py` and `check_pin_guards.py` with the existing base Python for replay. Raw results, partial checkpoints, commands/controls and hashes are retained. The accepted scope remains a fixed-state optional moment convenience with actual fallback, not a replacement for detailed transport physics.
