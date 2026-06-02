# Dev log — 2026-06-02 (b) — validation harness + adaptive solver

**Who:** R. O'Shaughnessy (with AI pair).  Follow-on to `2026-06-02-initial-scaffold`.

## Trigger
R. flagged: TOVs are tricky — small phase transitions (cs²→0) need special
stepsizing; *validate against RePrimAnd/Zhu rather than hoping it's good enough*.

## What we did
- Built an **independent in-process reference**: `nmbackend.reference.scipy_tov`,
  a radius-domain SciPy RK45 solver with the (H,β) tidal system — deliberately a
  different variable, stepper, and tidal formulation from the production solver.
- Cross-checked it against the production fixed-step RK4-in-h solver.

## Key finding (the harness did its job immediately)
The fixed-step default (`n_steps=800`) was **badly under-resolved**: M off ~5%,
Λ off ~25%.  Convergence was only ~1st order (not RK4's 4th) → error localized at
the crude leading-order **central boundary start**, not the interior.  A
center-graded grid fixed M,R but broke Λ (surface under-resolved): the tidal sector
needs density at *both* ends.

## Decision
Switched the default to an **adaptive diffrax (Dopri5 + PID)** backend behind the
existing `solve_structure` seam.  Result: agrees with the SciPy reference to
**M<0.05%, R<0.02%, Λ<0.1%** in ~250 adaptive steps; forward-mode AD flows through
via `diffrax.ForwardMode` (dR/dL_sym matches FD to ~1e-4); `vmap` works.  The
fixed-step `scan` backend is retained (lightweight; reverse-mode for structure-only)
and shown to converge to the reference.

Accurate default numbers (were inflated by under-resolution): M_max 2.11→**2.08**,
Λ(1.4) 449→**443**.

## Delivered for external corroboration (user runs these)
- `reference/reprimand.py`: RePrimAnd adapter scaffold — mirrors RIFT's (old)
  `EOSManager` interface with version-sensitive calls marked `# VERIFY` (RePrimAnd's
  API has drifted; RIFT's is old).
- `scripts/validate_against_reprimand.py`: 3-way M/R/Λ comparison driver.
- `env/pixi.toml`, `env/environment.yml`, `docs/environment.md`: build a validation
  env (jax+diffrax+lalsuite + RePrimAnd build deps) on the workstation. Sandbox
  can't build RePrimAnd (C++/Boost/GSL, no conda, tight disk).

## Couldn't access
`junior_tools` is not visible from the sandbox (not under the mounted folders).
If pointed at it / given `pydoc pyreprimand`, I'll wire the reference adapter to the
exact installed API.

## Still open (do NOT trust yet)
Strong first-order **phase transitions / cs²→0**: needs the tidal junction
condition, a real SLy/BPS crust (stop clipping cs² / `cummax`-smearing pressure),
and the executed RePrimAnd + Zhu δ-TOV cross-checks.  See docs/validation-plan.md §8.

## Tests: 20 passing (added `test_validation_reference.py`).