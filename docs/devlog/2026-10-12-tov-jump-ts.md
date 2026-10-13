# Dev log — 2026-10-12 — TOV tangents: steps now stop at the EOS table nodes

**Who:** R. O'Shaughnessy (with AI pair).  Fix of the defect F1 found by the
adversarial review of Q2C PR #46.

## Defect
`tov.solver._solve_diffrax` integrated the TOV + tidal system in h with Dopri5 + PID
and no `jump_ts`.  The right-hand side is only C^0 in h at every table node:

- eps = dp/dh - p comes from a C^1 cubic Hermite of p(h), so eps' jumps;
- cs2 is interpolated linearly, so cs2' jumps.

So steps straddled the nodes, and about half of all steps were rejected there.
ForwardMode differentiates the discrete solve with the step factors held fixed.
The mesh still moves: t0 = h_c - dh0 and dt0 depend on h_c and h_min.  The kinks
also move, with the table parameters.  The tangent therefore picked up the
derivative of the kink position relative to the stage points.  That term is
not controlled by rtol.

The old advice "use rtol ≲ 1e-10 for gradient/Fisher work" (devlog 2026-06-02f,
roadmap, backlog) was a symptom of this defect.

## Fix (`tov.solver._adaptive_tov`, shared by `solve_structure`, `solve_profile`, `solve_hybrid`)
- `PIDController(..., jump_ts=stop_gradient(table.h[1:]))`.  Each step ends just
  before a node, and the next restarts just after it, so no step contains a kink.
  `made_jump` also forces a fresh first stage instead of the FSAL reuse.
  - `h[0]` is the surface, i.e. the end point, so it is left out.
  - Nodes outside [h_min, h_c] are inert; the diffrax ClipStepSizeController
    searches from t0 and clips at t1.  The array therefore keeps the table's
    static length under jit/vmap, and no masking is needed.
- The start time is frozen against differentiation, and dt0 too: t0s =
  stop_gradient(h_c - dh0).  Its h_c dependence moves into the start state through
  y0 + f Δ + ½ f' Δ², with Δ = t0s - t0 = 0 and f' = df/dt along the flow.  The
  value is unchanged, and the first tangent dy0 - f dt0 is exact.  The ½ f' Δ²
  term (added after the review of PR #43) keeps the f' dt0² term of the second
  tangent.  As a result, no mesh point before the last carries a tangent.
- t1 = h_min stays live.  The last step ends there and spans no node.
- Node positions are gradient-stopped.  Their motion enters the tangent through
  the vector field on each side of the node.  Because the RHS is continuous, the
  moving-kink boundary term vanishes.
- The analytic case needs no special handling: every EOS in the package reaches
  the solver as a tabulated `EOSTable`.
- The `scan` backend is unchanged and documented.  Its uniform grid moves with
  h_c, and its tangent is the exact derivative of that discrete map.
- `solve_structure`'s public signature is unchanged.  There is no opt-out.  The
  values that move are the old solve's errors being removed (see "Suite
  comparison"); old numbers are reproducible from the old pin.

### Which mechanism fixes which column (measured after the review of PR #43)
MetaModel star of the table below, AD/FD - 1 for (M, R, Λ):

| variant | rtol | h_c column | L_sym column |
|---|---|---|---|
| base (b94994a) | 1e-7 | -8.2e-4, +2.7e-2, +3.6e-2 | -1.0e-1, +6.8e-4, -8.6e-2 |
| frozen start, **no** jump_ts | 1e-7 | +1.1e-4, +1.3e-4, +1.0e-4 | -1.0e-1, +6.8e-4, -8.6e-2 |
| frozen start, no jump_ts | 1e-9 | +8.9e-7, -1.3e-6, +1.6e-6 | -1.9e-3, -7.1e-4, -1.0e-4 |
| jump_ts, **live** start | 1e-7 | +4.9e-1, +4.3, +3.2e-1 | -2.8e-6, +1.7e-8, +3.0e-5 |
| jump_ts, live start | 1e-8 | -4.1e2, -4.8e3, -2.1e2 | -3.6e-7, +1.5e-9, +6.7e-7 |
| jump_ts, live start | 1e-9 | -1.6e6, -1.7e7, -8.2e5 | -5.5e-8, +3.1e-10, -1.5e-7 |
| jump_ts + frozen start (this PR) | 1e-7 | +1.0e-4, +1.2e-4, +1.0e-4 | -2.8e-6, +1.7e-8, +3.0e-5 |
| jump_ts + frozen start | 1e-9 | +5.2e-7, +6.7e-7, +4.3e-7 | -5.5e-8, +3.1e-10, -1.5e-7 |
| + `centre_refine=True` | 1e-7 | +1.4e-6, +8.1e-6, +1.2e-6 | -2.3e-6, +8.8e-9, +3.0e-5 |
| + `centre_refine=True` | 1e-9 | +1.5e-8, +1.1e-7, -7.1e-8 | -5.2e-8, +2.6e-10, -1.5e-7 |

- `jump_ts` fixes the *parameter* column.  That column then converges at the level
  of the values.
- The frozen start fixes the *h_c* column.  It is **required** once `jump_ts` is on:
  with a live start the mesh moves with h_c while the gradient-stopped node clips
  do not, and the h_c error grows as rtol tightens.
- The h_c column converges like rtol (1.2e-4, 8e-6, 7e-7 at rtol 1e-7, 1e-8, 1e-9),
  but about 1e3x above the values.  It does not "converge like the values", as an
  earlier draft of this PR said.
- The h_c residual is a near-common factor on (M, R, Λ): +1.0e-4, +1.2e-4,
  +1.0e-4.  The fixed-mass (IFT) derivatives in `fisher.sensitivity_matrix` and
  `build_family`'s stability use ratios of h_c partials, so most of it cancels
  there.  This is why the default-rtol level is tolerable for Fisher work.
- `fisher.GRADIENT_RTOL = 1e-10` stays (h_c column ~5e-8 there), with a comment
  at the constant.

### Second derivatives in h_c: unsupported
- jacfwd(jacfwd) in h_c of the MetaModel structure-only M was checked against FD
  of the (accurate) first derivative at rtol 1e-12.  The frozen start makes d²M/dh_c²
  wrong by AD/FD - 1:
  - **+3.2e8 at every rtol** with the first-order start correction;
  - +3.7e4 (rtol 1e-7) and +7.9 (1e-10) with the ½ f' Δ² term;
  - +7.9 (1e-7) and +2.5e-3 (1e-10) with `centre_refine=True`.
- The base solver gave O(1) errors here.
- The rest is the frozen mesh under-resolving the (h_c - h)^(-3/2) second
  tangent near the centre.
- Mixed d²M/dh_c dL_sym is fine: -9.3e-4 at 1e-7, +2.4e-7 at 1e-10.
- First h_c derivatives only is now stated in the `solve_structure` and
  `_adaptive_tov` docstrings, the README and physics.rst.
- `test_tov_jump_ts.py::test_second_hc_derivative_is_not_supported` pins the error
  as large.
- Downstream: ~/nicer-hotspot-ad `LinearizedNmBackendProvider` takes jacfwd of a
  packet containing dM/dh.  Its h_c column is now garbage.  Today it feeds only
  `> 0` / `isfinite` checks, and it must not be used for likelihood gradients.

The review's snippet froze t1 as well.  Here t1 is left live instead, so
dh_min/dθ is carried if a table's surface enthalpy depends on θ.  With the
default SLy crust it does not.

## Reproduction (nmbackend only; `jvp` vs central FD at fixed h_c, ~1.25-1.41 Msun)
FD reference: rtol 1e-12 with the patched solver, relative step 1e-4.  Between
steps 1e-3 and 1e-4 it is stable to ≤ 8e-6.  On the old solver the FD itself was
stable only to 1e-3 for the MetaModel dM/dL_sym.
Script: `jvp` of `solve_structure` inside `jax.jit`; steps from `sol.stats`.

GM1 (RMFTEOS, 247 nodes, SLy crust; parameter C_omega), h_c = 0.19048, M = 1.4146:

| partial | rtol | before AD/FD-1 | after AD/FD-1 |
|---|---|---|---|
| dM/dh_c | 1e-7 / 1e-9 / 1e-11 | +1.0e-3 / -1.2e-4 / -1.2e-4 | +8.5e-5 / +5.0e-7 / +8.5e-9 |
| dR/dh_c | 1e-7 / 1e-9 / 1e-11 | -1.7e-2 / -3.4e-2 / -3.6e-3 | +6.6e-5 / +4.0e-7 / +5.3e-8 |
| dΛ/dh_c | 1e-7 / 1e-9 / 1e-11 | +1.8e-2 / -8.8e-4 / -1.3e-4 | +8.4e-5 / +4.3e-7 / -5.5e-8 |
| dM/dC_ω | 1e-7 / 1e-9 / 1e-11 | -6.8e-3 / +2.1e-3 / +7.2e-4 | -1.9e-7 / -5.9e-8 / -5.7e-8 |
| dR/dC_ω | 1e-7 / 1e-9 / 1e-11 | -1.6e-3 / -2.9e-3 / -3.4e-4 | +1.5e-8 / +1.8e-8 / +1.8e-8 |
| dΛ/dC_ω | 1e-7 / 1e-9 / 1e-11 | +3.6e-2 / -1.4e-2 / -2.0e-3 | +1.2e-5 / +2.1e-8 / +4.4e-8 |

MetaModel (566 nodes, SLy crust; parameter L_sym), h_c = 0.18725, M = 1.2485:

| partial | rtol | before AD/FD-1 | after AD/FD-1 |
|---|---|---|---|
| dM/dh_c | 1e-7 / 1e-9 / 1e-11 | -8.2e-4 / +8.7e-5 / -1.7e-4 | +1.0e-4 / +5.2e-7 / +7.3e-9 |
| dR/dh_c | 1e-7 / 1e-9 / 1e-11 | +2.7e-2 / +2.0e-2 / +6.1e-3 | +1.2e-4 / +6.7e-7 / +4.6e-8 |
| dΛ/dh_c | 1e-7 / 1e-9 / 1e-11 | +3.6e-2 / -1.5e-3 / +1.3e-4 | +1.0e-4 / +4.3e-7 / -7.8e-8 |
| dM/dL_sym | 1e-7 / 1e-9 / 1e-11 | -1.0e-1 / -1.9e-3 / +3.6e-2 | -2.8e-6 / -5.5e-8 / -2.8e-8 |
| dR/dL_sym | 1e-7 / 1e-9 / 1e-11 | +6.8e-4 / -7.1e-4 / -5.2e-4 | +1.7e-8 / +3.1e-10 / +2.0e-10 |
| dΛ/dL_sym | 1e-7 / 1e-9 / 1e-11 | -8.6e-2 / -1.0e-4 / +9.4e-4 | +3.0e-5 / -1.5e-7 / -2.4e-8 |

Steps (rejected) for one tidal solve:

| rtol | GM1 before | GM1 after | MetaModel before | MetaModel after |
|---|---|---|---|---|
| 1e-7 | 413 (177) | 327 (8) | 391 (165) | 539 (9) |
| 1e-9 | 1568 (868) | 505 (15) | 1577 (821) | 720 (19) |
| 1e-11 | 3137 (1610) | 1015 (25) | 5054 (3001) | 1183 (32) |

For these two stars, M, R and Λ change by at most 3e-5 (Λ) at rtol 1e-7,
2e-6 at 1e-9 and 3e-7 at 1e-11.

After the fix, the remaining default-tolerance error (~1e-4 on the h_c column)
converges like rtol, about 1e3x above the values: 8.5e-5, 6.5e-6 and 5e-7 on GM1
at rtol 1e-7, 1e-8 and 1e-9.  It is not set by dt0, which was scanned from 1e-4
to 1e-8.

### Residual: the h_c tangent near the centre
The cause of the residual on the h_c column: the start is now frozen, so
the near-centre mesh no longer co-moves with h_c.  There r ~ sqrt(h_c - h), and
the tangent dr/dh_c ~ (h_c - h)^(-1/2) is more singular than the primal that the
controller resolves.

On a kink-free table the old co-moving mesh was in fact better.  For the
`test_tov_profile` incompressible star at rtol 1e-9, dR/dh_c vs the exact
Schwarzschild interior was 7.8e-10 before and 4.9e-7 after.  The test bound was
tightened from 2e-5 to 2e-6 so it still catches a regression.

A shifted variable x = h_c - h would restore co-motion, but it was not adopted.
Stage points mapped back to h = h_c - x can round onto the wrong side of a node,
which costs O(kink * dt) in the tangent through ∂f/∂h.  The review measured it
worse: h_c column 1.2e-2, L_sym column 7.6e-2 at rtol 1e-7.

What does work is integrating the inner 5% of (h_c - h_min) at rtol × 1e-3 and the
rest at rtol.  This confirms the near-centre cause.  It is now an opt-in keyword,
`solve_structure(..., centre_refine=True)`, off by default:
- h_c column 1.2e-4 → 8e-6 at the default rtol;
- parameter column and values unchanged;
- cost +40% (MetaModel) to +75% (GM1) for values and +13% to +88% for jacfwd
  (table under Cost).

## Cost
Wall time for a vmapped 120-star family on the login node (jit, after compile;
values / jacfwd dM/dh_c).  Login-node timings vary by about ±25% between runs:

| table | rtol | before | after |
|---|---|---|---|
| GM1 (247 nodes) | 1e-7 | 42 / 75 ms | 38 / 56 ms |
| GM1 | 1e-9 | 126 / 233 ms | 60 / 83 ms |
| MetaModel (566 nodes) | 1e-7 | 37 / 66 ms | **108 / 169 ms** |
| MetaModel | 1e-9 | 164 / 227 ms | 134 / 203 ms |

Every in-range node now costs at least one step.  On the dense MetaModel table
(SLy crust plus 240 low-density core nodes), a values-only family at the default
rtol is therefore 2.9x slower.  For gradient work, the comparison that matters is
after-at-1e-7 (1e-4 accurate) against before-at-1e-11 (still 0.1-4% biased, and
slower).  A table with thousands of rows, or one refined with `refine_table` to
a large `n_out`, pays per node.

The two keyword options, measured interleaved (min of 7) at the default rtol,
values / jacfwd:

| table | default | `jump_nodes=False` | `centre_refine=True` |
|---|---|---|---|
| MetaModel | 112 / 175 ms | 35 / 52 ms | 157 / 197 ms |
| GM1 | 36 / 50 ms | 33 / 52 ms | 63 / 94 ms |

Family value error vs rtol 1e-12, stars with M > 0.5:

| table | variant | M | R | Λ |
|---|---|---|---|---|
| MetaModel | default | 4.5e-8 | 6.9e-9 | 2.2e-7 |
| MetaModel | `jump_nodes=False` | 8.4e-6 | 7.3e-6 | 7.7e-4 |
| MetaModel | `centre_refine=True` | 3.4e-8 | 6.2e-9 | 1.9e-7 |
| GM1 | default | 5.9e-8 | 6.9e-9 | 2.4e-7 |
| GM1 | `jump_nodes=False` | 1.8e-5 | 1.8e-5 | 1.8e-4 |

`jump_nodes=False` is the documented opt-out for bulk value-only scans of very
dense tables.  It keeps the frozen start, so its h_c column is fine (1.1e-4).  Its
parameter column is the base's (up to 10%).  The review measured that thinning
the jump set does not help either: every unmarked kink is straddled.

Adding nodes to an interval where the
interpolant is already smooth only adds steps.

## Tests
- New `tests/test_tov_jump_ts.py` (~60 s):
  - AD vs FD of all six partials (M, R, Λ by L_sym and h_c) is ≤ 3e-4 at the
    default rtol; it was up to 10%.
  - The same is ≤ 1e-5 at rtol 1e-9; it was up to 1%.
  - Rejected steps are below 10% of accepted steps; the old solver had 162 of 225.
  - Batched (vmap over EOS parameters) and unbatched solves agree to 1e-10.  The
    old solver differed by 2e-5, because roundoff flipped accept/reject decisions
    at the kinks.
  - All four checks fail on the old solver.
  - Added after the review: `centre_refine=True` brings the h_c column to
    ≤ 3e-5 (measured 8.1e-6).  `test_second_hc_derivative_is_not_supported`
    pins the second h_c derivative as wrong by more than 10x (measured 3.7e4).
- `test_tov_profile` jacfwd bound tightened from 2e-5 to 2e-6 (measured 4.9e-7).
- `test_ad_gradients.py::test_fwd_grad_lambda_converges_with_tolerance` asserted
  the symptom, "the tight rtol is closer to FD than the loose one".  It failed once
  both were accurate: AD at 1e-8 and at 1e-12 agree to 7e-7, and the test's ±0.2 FD
  step had a 2.4e-5 truncation error.  It is replaced by
  `test_fwd_grad_lambda_matches_fd`, with a ±0.01 step and bounds of 1e-4 at the
  default rtol and 1e-5 at 1e-10.
- Existing-suite value changes: see "Suite comparison" below.

## Suite comparison vs origin/main (b94994a)
The full suite ran on SLURM debug (8 CPUs), once on b94994a and once on this
branch:

| | b94994a | this branch |
|---|---|---|
| passed / skipped / failed | 298 / 94 / 0 | 302 / 94 / 0 (304 / 94 / 0 after the review responses) |
| wall time | 819 s | 919 s (953 s) |

- The 4 extra passes are the new file.
- The 94 skips are the NSCool-dependent cooling tests (NSCool is not built
  here), the same on both sides.
- About 30 s of the extra wall time is the new file.

A pytest plugin logged every star (h_c, M, R, Λ) that each test solved, through
`jax.debug.callback` in `_finish`.  Stars were matched by exact h_c.  Over stars
with M > 0.5 Msun and R < 30 km, in every test the largest changes were:

- M and R by ≤ 2.5e-5 in tidal solves.
- M by up to 7e-4 in the structure-only probes of `default_hc_grid` (rtol 1e-6),
  on low-mass stars of crust-join tables.
- Λ by up to 3e-3, near M_max on soft tables, where Λ ~ 8-25.

Spot-checked against rtol 1e-12, each of these changes is the *old* solve's error
being removed:

- Probe masses of the (L, K_sym) = (80, 50) fallback-join table at rtol 1e-6:
  the old error was 9e-5 to 5e-4, the new one 2e-7.
- Λ over the stable branch of the eight L = 20 near-rejection tables of
  `test_lambda_smooth_to_rejection_boundary` at the default rtol: the old error
  was 2.5e-3 to 6.3e-3 per table.  The new error is ≤ 1.3e-5, except 3.2e-4 on
  one table (d = 10).
- M on the same branches: the old error was 7e-6 to 1.7e-5, the new one 9e-8.

Two residuals in the spot-checks, explained after the review:

- **"3.2e-4 on one table (d = 10)".**  This is star 0 of that family: M = 0.079
  Msun, Λ = 2.7e12, an extended low-mass configuration that the family labels
  stable.  The rtol 1e-12 reference is itself unconverged there: rtol 1e-11 and
  1e-12 differ by 7.7e-4, because k2 → 0 as C^5 and Λ = (2/3)k2/C^5 cancels.  So
  it is not a solver residual.  On the stars that matter the post-fix error is
  ≤ 3.8e-7, which agrees with the review's 3e-7 near M_max.
- **The Λ "floor" of 3e-7 at rtol 1e-9** (default MetaModel family) is the same
  effect.  It is the lowest star (M = 0.108, Λ = 4e8), where rtol 1e-11 and 1e-12
  also differ by 3.4e-7.  For M > 0.5 the family Λ error at rtol 1e-9 is 1.7e-9.
  This item is closed as a reference-conditioning artefact.  Open sub-item: Λ
  for M ≲ 0.1 Msun is not reliable to better than ~1e-3 at any rtol.

So the old solver's *values* were also degraded by the node-straddling, not only
its tangents.  No assertion margin tightened materially.  For example, the
`test_lambda_smooth_to_rejection_boundary` residual went from 3.4e-4 to 3.3e-4,
against a bound of 1e-3.

Gradient values asserted by existing tests (AD/FD - 1):

| test | b94994a | this branch |
|---|---|---|
| ad_gradients mmax (dM_max/dK_sym, family) | -1.5e-4 | +1.2e-5 |
| ad_gradients radius (dR/dL_sym, rtol 1e-9) | -1.3e-3 | +1.9e-7 |
| ejecta dM_ej/dL_sym (family, rtol 1e-9) | -2.3e-3 | -9.5e-4 |
| fisher J, Λ1.4 by (L_sym, cs2_tail) | +7.9e-4, -2.8e-4 | -3.2e-4, -6.1e-4 |
| fisher J, R1.4 by (L_sym, cs2_tail) | -6.2e-4, -1.3e-3 | -3.3e-5, -5.4e-4 |
| fisher J, M_max by (L_sym, cs2_tail) | -5.1e-3, -8.6e-3 | -5.0e-4, -3.7e-4 |
| tov_profile dR/dh_c vs exact (rtol 1e-9) | -7.8e-10 | +4.9e-7 |

- The Fisher FD reference uses steps 0.4 and 0.01, and family-located h_c.  Its
  own truncation error is probably what limits the Λ row.
- The profile row is the one that got worse.  The star is kink-free, and the
  mechanism is explained above.  The test bound is 2e-5.

## Downstream
- Q2C pins nmbackend via `NMB_PIN`.
- `~/nicer-hotspot-ad` imports nmbackend editably from 40 files.
- Every quoted gradient/Fisher number that went through the adaptive TOV tangent
  moves by up to a few 0.1%.  Examples are Q2C's dΛ1.4/dθ and Paper II's Fisher
  NS rows.  Values do not move beyond tolerance.  Downstream reruns are needed
  after the bump.
- `fisher.GRADIENT_RTOL = 1e-10` is left alone.  It is still valid, though 1e-9
  now suffices.

## Review of PR #43 (2026-10-12): responses

| # | finding | disposition |
|---|---|---|
| 1 major | d²/dh_c² wrong by ~1e8 | ½f'Δ² start term added (3.2e8 → 3.7e4 at 1e-7, 7.9 at 1e-10).  "First h_c derivatives only" is in the docstrings, README and physics.rst.  `test_second_hc_derivative_is_not_supported` pins it.  The nicer-hotspot-ad consumer is flagged in the PR.  The near-centre refinement is implemented as opt-in `centre_refine` (default off). |
| 2 major | attribution and "converge like the values" | Corrected in the module docstring, README, physics.rst, architecture.md, roadmap, backlog and here.  The freeze is marked REQUIRED in `_adaptive_tov`'s docstring and code. |
| 3 minor | h_c residual is a common factor | Added above and in the README and test comment. |
| 4 minor | near-centre refinement | `centre_refine=True`, measured above. |
| 5 minor | test_tov_profile bound | 2e-5 → 2e-6. |
| 6 minor | GRADIENT_RTOL comment | Added at the constant. |
| 7 minor | Λ floor 3e-7 | Explained above (low-mass reference conditioning); open sub-item recorded. |
| — | d = 10 residual 3.2e-4 | Explained above (star 0, M = 0.079, Λ = 2.7e12). |
| — | optional `jump_ts=False` | Implemented as `jump_nodes=False` (default True), with its accuracy loss stated. |
