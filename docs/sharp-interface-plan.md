# Sharp-interface mode response: proposed contract and checks

17 November 2026; task notes only, no physics implementation or production run.
Source: Dong, Kuan & Yunes, ADS `2026arXiv260933002Z`, [arXiv:2609.33002v1](https://arxiv.org/html/2609.33002v1).
Paper interpretation and publication scope live in nmb-papers
`paper9_postmerger_modes/notes/SHARP-INTERFACE-PLAN-2026-11-17.md` on the
Paper8/9 development lineage (PR52, `aa88a3f`).

## Reuse and precise missing capability

Extend the [existing mode ladder](rotating-modes.md), not a second mode engine.
The current PR64 branch (`1d0890c`) supplies the quadratic kernel and branch
continuation; it does not assemble the cold full-GR sharp-interface polar
problem. `src/nmbackend/eos/hybrid.py`, `eos/base.py` and
`tov/phase_transition.py` contain equilibrium hybrid/tabular and static tidal
machinery. A static tidal jump is not an oscillatory conversion law. Preserve
those validated paths and [multi-branch inference design](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/e7e82527c57cbfc869db579130fbe65cf1aa6b93/docs/rift-multibranch-design.md)
when composing later work; do not collapse twins onto a single M-to-observable
map. The multi-branch design is on main `e7e8252`, separate from PR64's base.

No local AGENTS.md or .agents/skills is present in these inspected repository
lineages. Existing PR64 remains owned by its current author. Implementation
lead and independent interface reviewer are unassigned; Richard must designate
them. This follow-on documentation must be reconciled with PR64 and current
main before any code work, without resetting either owner's branch.

## Proposed sharp-interface contract (design, not an API already implemented)

- Versioned EOS/source hashes, units and density conventions; phase labels and
  inner/outer orientation. Store transition pressure Pt, one-sided energy and
  baryon densities and their signed jumps, chemical potentials and interface
  position. State Maxwell/no-thin-shell assumptions; finite surface tension or
  a stress-energy layer requires a different reviewed matching model.
- Retain one-sided dp/de, thermodynamic/composition susceptibilities where
  available and background radial pressure gradients. Never interpolate a
  smooth derivative across the forbidden density gap or treat the jump as a
  finite radial soft layer. A null-jump control must also match smooth-phase
  derivatives if it is to recover a single-phase reference.
- Separate background equilibrium closure from perturbative closure in each
  phase, with provenance. Record barotropic versus frozen composition and
  omitted buoyancy/rate terms explicitly. Preserve existing ModeBackground
  provenance checks; merely replacing sound speed does not add stratification.
- Store the conversion prescription independently: slow or rapid endpoint;
  algebraic alpha labeled sensitivity-only; future frequency-dependent law
  requires its own derivation/version. Record displacement, traction, metric,
  time/frequency sign, normalization and outgoing-boundary conventions.
- Carry central pressure/enthalpy, mass, radius, branch ID and stability
  classification separately from interface response. Share one immutable
  background between endpoint solves. Turning-point selection is not proof
  of conversion-dependent radial stability.

## Dependency-ordered work and acceptance evidence

| ID | Work | Required evidence |
| --- | --- | --- |
| I0 | Contract review and exact fixture intake | Serialized round trip retains phase/side labels, Pt, jumps, derivatives and closure provenance; rejects missing or inconsistent units/sides; frozen input hashes and reference targets |
| I1 | Spherical full-GR polar operator and endpoint matching, after the existing physical reference rung | Independently reviewed equations, regular center/free surface/outgoing exterior, reference no-transition spectrum; no Cowling result labeled full GR |
| I2 | Matching tests | Dimensionless residuals for metric matching, normal traction and endpoint-specific interface displacement; in source conventions continuity of H1, K, DeltaP plus xi_r (slow) or xi_r-DeltaP/Pprime (rapid), with Pprime evaluated on each side; residuals converge under refinement |
| I3 | Null and reproducibility tests | No-interface and zero-jump/equal-derivative fixtures recover smooth reference and endpoint coincidence within numerical error; independently rerunnable identical-background slow/rapid pairs recover selected published f/p1 values within combined reference/numerical error |
| I4 | Identity and convergence | Physical eigenfunction overlaps on a positive norm with phase/alignment control; subspaces near degeneracy and unresolved matches rejected; refine bulk/interface mesh, EOS table, center/surface cutoff and outgoing domain/matching radius; report complex-frequency and junction residual error separately |
| I5 | Nuisance comparison handoff | Versioned endpoint shifts and convergence bounds for Paper9 versus controlled EOS/stellar variations; branch and failed-case accounting retained; no inference or production job implied |

Set absolute and relative residual/frequency targets before implementation,
with scaling documented and numerical error below the smallest interpreted
contrast. I3 cannot be passed by copying source frequencies into tests; use
independent physical assembly. Existing polynomial left/right root residuals
are necessary but do not replace the physical junction/constraint tests.
A frequency-order sort or scalar toy overlap does not identify a stellar p1.

## Deferred work and current blockers

- Finite-rate interface law: blocked on a reviewed flux/affinity derivation,
  causal dissipative response and microscopic coefficients; test conservation,
  nonnegative dissipation and both endpoints before admission. The source's
  algebraic interpolation supplies none of those ingredients.
- Bulk composition/weak-rate response: separate closure/transport work; no
  bulk-viscosity coefficient can stand in for boundary conversion kinetics.
- Detectability: blocked on physical forcing/normalization, excitation, damping,
  signal transfer and nuisance-aware recovery; owned by Paper9's observable rung.
- Hot/remnant use: blocked on qualified finite-T/differential backgrounds,
  quasistationarity and dynamical transfer validation. Existing HOLD/STOP applies.
- Current backend main lacks PR64's physical-mode foundation and the foundation
  itself lacks full-GR assembly. These are implementation dependencies, not
  permission to launch the missing work. No new dependencies or tests were run
  by this planning change.
