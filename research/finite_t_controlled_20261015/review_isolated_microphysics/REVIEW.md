# Independent review: process-isolated microphysics interface

**NOT ACCEPTED at the initial frozen source. Four concrete contract failures require repair before the stated interface qualifications can be accepted.** The selected physical source calls and parent-process isolation pass. No evidence here invalidates the already reviewed underlying EOS or beta/PC physics.

Reviewed worker SHA `98a60f1a106730fe76083cd46e73b3a86ea783bca4209a10e8ec70f5009b5c3b`, client SHA `cd8353ee7ca3049bb2f760c1b9681a04cc1d738af0a8efa0025e73a40fb6b0a1`, protocol SHA `f5125a9ca7e2d7a4bc0e71135ff330f69a8a470cc75a5f427019920ab192a3c6`. Initial independent results: **14 PASS,4 FAIL**,7.09seconds. All frozen source and raw files remain hash-identical before/after. Only the owned review directory and its disposable mirror were written.

## Reproduced blockers

1. **Mandatory transitive source coverage is not enforced.** `verify_pins` checks only names present in the supplied `sources` dictionary. Removing the actual PC reference source key still passes. The runtime identity then becomes a digest of the incomplete pin set, rather than rejecting the coverage error.
2. **The missing-PC coverage error reaches real computation.** In an owned exact source mirror, the review removes the `review_pc_moments/reference.py` pin, appends a harmless comment to that source, and recomputes request identity. The actual worker returns exit0/statusok and a physical answer. Its import audit cannot see the PC module because `module_from_spec`/`exec_module` never registers it in `sys.modules`. Thus the declared requirement that every physical source be checked before import/evaluation is false for this path. This is a provenance/admission failure; it does not depend on a malicious physics change or modifying any frozen source.
3. **A false exact-zero emission branch is admitted.** Starting from a real accepted response, set beta number and energy to the wire's exact-zero `empty_incident_radiation` representation and assign the previous combined positive moments to PC. All combined sources and ledgers stay consistent, and the parent accepts. An empty incoming antineutrino justifies zero inverse reaction, not zero neutron beta emission. The note requires represented positive channel totals in this selected admission box; `validate_positive` alone does not enforce that branch-specific contract.
4. **Boolean inverse zero passes numeric validation.** Replacing inverse value0 with JSON Booleanfalse still passes because the exact-zero case uses `value!=0`, without the numeric-type guard used for ordinary positives. This contradicts a strict typed numerical wire contract.

The raw initial response, altered-source mirror response, failed assertions and all immutable before/after hashes are in `initial_review_results.json`; probe source and preserved mirror are supplied. No author repair was attempted. The parent was notified immediately and further experiments stopped.

## Independently passing evidence

A uniquely loaded stdlib-only client was called with colliding parent modules named `study`, `protocol`, `hot_moments`, `kernel`, `beta`, `vacuum_cc`, `current`, `events`, `free_baryons` and `lepton_reference`. Every object and the parent sys.path remained identical, with no new modules after client initialization. No study driver was imported or executed by the review.

Three real fresh calls cover initialT2/Ye.3 base, a new heldout Heun endpointT1.9955690477112278/Ye.3218092326059971 fine, then repeated initial/fine. The heldout changed-state source agrees with the prior independently integrated actual forward source within2e-12 relative: R=1.0642163330393762e-6fm^-3s^-1,Q=1.1400114218759336e-5MeVfm^-3s^-1. These forward checks precede any derivative or interface claim. Underlying physical kernels are intentionally shared with separately qualified references; this is not a new matrix-element accuracy test.

Finite logarithms−55933 and+1000 retain ordinary-unrepresented flags with valueNone and no floor. Those are representation tests only; this selected physical interface demands represented positive totals and does not extend the physical kernels to arbitrary cold/underflow states.

The subprocess strategy addresses the actual accidental parent-module collision mechanism and computes fresh EOS/source results rather than reusing a source cache. The equations, one-body current conventions, restricted species inventory and rest-inclusive E/Le signs remain consistent with the separately accepted two-antinu-channel element. However, source identity and branch representation checks must be repaired and independently rerun before this adapter is admitted as the intended reviewable interface.

## Acceptance boundary after any repair

Require exact mandatory source coverage, actual loaded module path/hash coverage including spec-loaded modules, and branch-specific positive versus inverse-zero validation with strict numeric types. Preserve these initial failures. Recheck the actual altered-source mirror and real fresh calls after the new source freeze; passing an in-memory pin mutation alone is insufficient.

Even after repair this is trusted pinned research-code process isolation, not an OS security sandbox, hostile-interpreter defense, concurrent-filesystem race defense, empirical weak/EOS uncertainty estimate, certified entire admission rectangle, full flavor inventory or production API. No pair-annihilation, complete beta equilibrium, opacity or time evolution follows automatically.
