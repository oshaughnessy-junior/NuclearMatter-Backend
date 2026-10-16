# Dev log — 2026-10-14 — the fixed-step `scan` TOV backend was wrong at every h_c

**Who:** R. O'Shaughnessy (with AI pair).  Found by the adversarial review of NMB
PR #44.  On MetaModel at h_c = 0.3 it gave M = 3.44 and Λ = NaN at n_steps = 2000,
and Λ 29.5% low at 8000.

## Defect
`tov.solver._solve_scan` stepped RK4 uniformly in h.  It ran from the series
start h_c − dh0 to h_min, with dh0 = 1e-8 (h_c − h_min), the same start as the
diffrax path.

- Near the centre r ∝ (h_c − h)^½, so dr/dh ∝ (h_c − h)^−½ is singular.
- The first step is ~1e4 dh0 long.  Its k1 slope is taken at r0 ∝ dh0^½, so the
  step overshoots r by orders of magnitude.
- That leaves a core with r far too large for its m: a non-regular solution that
  nothing downstream recovers.

The bug was not confined to high h_c.  It has been there since the first commit,
at every h_c and on every table.  Relative errors on main 25e6818, vs diffrax at
rtol 1e-10, over h_c ∈ {0.05, 0.1, 0.15, 0.2, 0.25, 0.29, 0.3, 0.35, 0.4, 0.5}:

| table | n_steps = 2000 | n_steps = 8000 |
|---|---|---|
| MetaModel | M +63..+210%, R ≤ +28%, Λ NaN at 9/10 | M +2.7..+11%, R ≤ 1.8%, Λ −24..−36% |
| GM1 | M +68..+210%, R ≤ +28%, Λ NaN 9/10 | M +3.5..+10%, R ≤ 1.3%, Λ −25..−34% |
| SLy4 | M +69..+210%, R ≤ +27%, Λ NaN 9/10 | M +3.7..+10%, R ≤ 2.1%, Λ −24..−35% |
| bridge1 (draw 1) | M +67..+210%, R ≤ +25%, Λ NaN 9/10 | M +1.9..+9.6%, R ≤ 2.5%, Λ −24..−35% |

Reverse-mode gradients, MetaModel, d(M, R, Λ)/d(L_sym, h_c):
- n_steps = 2000: NaN with the tidal sector at h_c = 0.2 and 0.3.  dM/dL_sym was
  off by a factor up to 37 (+3.6e3%).
- n_steps = 8000: dM/dL_sym +150%, dΛ/dL_sym up to −330%.

Two existing tests did not catch it:
- `test_scan_converges_to_reference` only checked that the error *decreases*
  from 2000 to 8000 steps.  A 100% → 7% error does.
- `test_reverse_grad_structure_only` only checked finiteness, at a fixed table.

## Fix (`tov.solver._scan_mesh`, `_eos_points`, `_solve_scan`, `_cell_eos`)
1. **Integrate in σ = √((h_c − h)/H)**, H = h_c − h_min, from σ0 = √(dh0/H) to 1.
   r(σ) is regular at the centre, and dX/dσ = f(h_c − Hσ², X) · (−2Hσ).
2. **Mesh, frozen against differentiation** (as the diffrax `jump_ts` are).
   Static length n_steps + n_steps//4 + len(h); points outside the star clip to
   an end and give zero-length steps.
   - `n_steps` base points at σ0 + (1 − σ0) sin(πs/2), s uniform.  The sin map
     packs steps toward the surface.
   - Every table node.  ε′ and the tidal term dε/dh jump there, as do the jump
     cells of `_degenerate_cells`.  Each step evaluates the EOS on **its own**
     cell (`_cell_eos`; cell chosen at the step's primal midpoint).
   - `n_steps // 4` points equidistributed in the per-cell EOS variation
     (`_eos_points`):
     - a regular cell weighs |Δln c_s²| + |Δln ε|;
     - a jump cell weighs nothing;
     - the cell carrying a jump's junction bubble weighs ln(1 + Δε/ε).  Its
       points sit at t = u^{1/(q+1)}, which equidistributes the bubble
       (q+1) t^q Δε/dh.
3. **r and m on the frozen mesh; y in a node-anchored picture**, as in the
   diffrax path (`_derivs_x`).
   - In step cell i, y holds y(σ̃(σ)).  σ̃ is linear in the cell and maps the
     cell's frozen ends to their live positions.  It is zero-valued: σ̃ == σ in
     value.
   - The jump of the tidal term at a node that moves with h_c or θ then
     contributes its boundary term.
   - r and m are taken at σ̃ to first order.
   - σ̃ = σ at both ends of the star (σ0, and the surface node at σ = 1), so no
     start or end correction is needed.

Cost: n_steps + n_steps//4 + len(h) − 1 RK4 steps.  For MetaModel at the default
2000 (566 nodes) that is 3065, against 2000 before.  The tidal case costs twice
the EOS evaluations per stage (structure at σ, y at σ̃).  Timings are in the PR.

## Results (vs diffrax at rtol 1e-10; main 25e6818 + this change)
Values, worst over the same h_c grid.  The worst point is h_c = 0.05 (0.26 M☉,
Λ ~ 2e6).  For h_c ≥ 0.2 it is ≤ 8e-7 (M) and ≤ 5e-6 (Λ), except bridge1's Λ
(5.2e-5 and 7.8e-5 at h_c = 0.3 and 0.4; 7.5e-6 at 0.29):

| | MetaModel | GM1 | SLy4 | bridge1 |
|---|---|---|---|---|
| n=500 | M 8.9e-4, R 3.1e-4, Λ 4.0e-3 | 8.4e-4, 2.1e-4, 3.6e-3 | 9.4e-4, 3.8e-4, 4.3e-3 | 8.9e-4, 3.1e-4, 4.0e-3 |
| n=2000 | M 1.0e-5, R 3.7e-6, Λ 4.7e-5 | 1.0e-5, 2.7e-6, 4.5e-5 | 1.1e-5, 4.4e-6, 4.9e-5 | 1.0e-5, 3.7e-6, 7.8e-5 |
| n=8000 | M 4.1e-8, R 1.5e-8, Λ 1.9e-7 | 4.2e-8, 1.1e-8, 1.8e-7 | 4.2e-8, 1.7e-8, 2.0e-7 | 4.1e-8, 1.5e-8, 6.5e-7 |

Reverse-mode `jax.jacrev` of the scan vs `jax.jacfwd` of diffrax.  Diffrax here is
at rtol 1e-10 and matches FD at rtol 1e-12 to ≤ 1.5e-7.  MetaModel,
θ = (L_sym, h_c), worst over h_c ∈ {0.1, 0.2, 0.3} and both columns:

| | M | R | Λ |
|---|---|---|---|
| old n=2000 | NaN / ×37 | NaN / ×4.8 | NaN |
| old n=8000 | +150% | +50% | −330% |
| new n=500 | 1.9e-3 | 4.0e-4 | 1.0e-3 |
| **new n=2000** | 6.1e-5 | 8.0e-6 | 2.3e-5 |
| new n=8000 | 4.3e-7 | 7.9e-8 | 1.7e-7 |

`tidal=False` gives the same M, R rows.

Steep-joint table of `tests/test_tidal_cs2_consistency.py` (p ×2.7 across one
cell; the nodes above it move with `scale`), M = 1.4.  Scan `jacrev` vs FD of
diffrax at rtol 1e-11:

| | dM | dR | dΛ |
|---|---|---|---|
| n=2000 (scale, h_c) | 4.4e-7, 1.8e-6 | 2.0e-8, 7.0e-7 | 7.1e-6, 6.2e-6 |
| n=8000 | 4.9e-9, 1.2e-8 | 2.1e-9, 1.8e-9 | 7.4e-8, 8.1e-8 |
| n=2000, y picture off | (same) | (same) | **−1.5, −0.36** at every n |

## What did not work (measured)
The first four rows were measured on a98be39, before PR #44, when the tidal term
was the C⁰ node c_s² interpolated in h.

| variant | result |
|---|---|
| steps at the table nodes, still in h | identical to the old scan: the centre is the defect |
| σ uniform + nodes | Λ error stalls near 4e-6 as n grows (surface under-resolved) |
| mesh points that follow their nodes (live) | base points cross nodes; Λ(L_sym) ripples at 1e-6 on a 0.01 MeV scale; AD exact for that map but 3.6e-3 off the true dΛ/dL_sym |
| frozen σ mesh, no EOS points | dΛ/dL_sym up to 9e-3 off at n=2000, 2nd order (values 1e-6).  Localised by refining one σ band at a time: only σ ∈ (0.94, 0.95) at h_c = 0.2 mattered, i.e. the crust–core joint, where c_s² triples over two node intervals and d c_s²/dL_sym ≈ 18%/MeV.  Where the table is denser than the base grid a step is one node interval however large n is.  Splitting every interval ×2/×3/×4 gave 1.4e-3/5.7e-4/1.8e-4; +300/+600/+1200 EOS-placed points gave ~2e-5, hence n_steps//4 |
| (on main) every mesh point anchored to its cell, live | dM/dL_sym 1.2e-2 off at n=2000.  The top cell runs from the node below h_c (moves with θ) to the start (does not), so the near-centre steps rescale by tens of percent per MeV; #44 saw the same in diffrax (moving r, m: dM/dL_sym 1e-6 → 6e-4) |
| (on main) frozen mesh, no y picture | the steep-joint row above: dΛ/dscale −150% |

## Not changed
- The EOS interpolation and the central series start.
- The diffrax backend's values and tangents.  Its `_derivs_x` now gets the cell
  EOS from `_cell_eos`, which the scan shares, with the same operations in the
  same order.
- `_derivs` (the old h-RHS of the scan, through `EOSTable.at_h_tidal`) had no
  other caller and is removed.

## Review follow-ups (2026-10-16)
Measured on the #46 head after the review, MetaModel unless stated.

Second derivatives through the scan are O(1) wrong at any n_steps.  The frozen
mesh and the first-order y picture keep only first-order terms.  The Hessian also
disagrees with FD of the scan's own `jacrev` by the same amounts, so the cause is
not discretisation.  Now documented in `solve_structure` and pinned by a test.

| Hessian diagonal / FD(diffrax jacfwd at rtol 1e-12) − 1 | n=2000 | n=8000 |
|---|---|---|
| d²(M, R, Λ)/dh_c², h_c = 0.2 | −0.49, +0.93, −0.096 | same |
| d²(M, R, Λ)/dh_c², h_c = 0.3 | −0.46, −0.63, −0.090 | same |
| steep joint d²(M, R, Λ)/dscale² | −0.087, −0.087, +7.2 | same |

Second-order AD through `MetaModelEOS.eos_table` fails outright, inside its
equinox loop, for either backend.

AD of the scan against central FD of the scan's own values, n_steps = 2000 and
8000, h_c ∈ {0.1, 0.2, 0.3}: ≤ 2.2e-7 at the smallest steps (ΔL_sym = 0.002, Δh_c = 1e-6).
Larger steps differ by the FD truncation error.  An earlier review note of 3e-6
came from the abandoned first version.

Worst first-derivative error over h_c ∈ {0.1, 0.2, 0.3}, both columns:

| n_steps | 500 | 1000 | 2000 | 4000 | 8000 | 16000 |
|---|---|---|---|---|---|---|
| dM | 1.9e-3 | 3.9e-4 | 6.1e-5 | 6.1e-6 | 4.3e-7 | 4.4e-8 |
| dR | 4.0e-4 | 7.1e-5 | 8.0e-6 | 6.5e-7 | 7.9e-8 | 4.9e-8 |
| dΛ | 1.0e-3 | 1.9e-4 | 2.3e-5 | 1.9e-6 | 1.7e-7 | 5.3e-8 |

That is about n^-2.5 below 2000 steps and n^-3.5 above.  The 16000 column is near
the diffrax reference's own accuracy (~5e-8).

EOS-point weights.  The review asked whether node c_s² (a diagnostic since #44)
should be replaced by the variation of the integrated tidal term dε/dh within
each cell.  Worst Λ error (values for the tables, first derivatives for MetaModel
and the steep joint) at n_steps = 2000:

| weights | MetaModel | steep | GM1 | SLy4 | bridge1 | bridge2 |
|---|---|---|---|---|---|---|
| node c_s² (kept) | 2.3e-5 | 7.1e-6 | 4.5e-5 | 4.9e-5 | 7.8e-5 | 1.0e-4 |
| dε/dh within the cell | 2.3e-5 | 7.1e-6 | 4.5e-5 | 4.9e-5 | 5.5e-4 | 1.0e-3 |
| both | 2.3e-5 | 7.1e-6 | 4.5e-5 | 4.9e-5 | 9.3e-5 | 2.8e-4 |

The node c_s² weights are never worse, so they stay.

The scan at n_steps = 2000 is 5.7e-5 off the SciPy radius-domain reference in M
at 0.2 h_max, the same as diffrax (5.6e-5).  `test_scan_converges_to_reference`
now bounds that at 3e-4; it used to check only that the error decreases.

## Callers of `backend="scan"` (2026-10-14)
Only NMB's own tests use it:
- `tests/test_validation_reference.py::test_scan_converges_to_reference`
- `tests/test_ad_gradients.py::test_reverse_grad_structure_only` (jax.grad, tidal=False)
- `tests/test_ad_gradients.py::test_jacfwd_matches_jvp_with_tidal` (jacfwd/jvp)

There are no callers in `~/Genesis_Quarks2Cosmos` (all its worktrees included) or
`~/nicer-hotspot-ad` (its vendored NMB copy has the old dispatcher but no
callers).  No CLI, YAML or env option selects it.  It is reachable only through
the public `backend=` argument of `solve_structure` / `build_family` and through
`durca_mass(**tov_kw)`.  So no downstream result was computed with the broken
scan.

## Tests (`tests/test_tov_scan.py`)
- Values vs diffrax on MetaModel, GM1, SLy4 and bridge1 at n = 2000 and 8000.
- `jacrev` of the scan vs `jacfwd` of diffrax on MetaModel (L_sym, h_c), tidal on
  and off.
- `jacrev` on the steep-joint table vs FD.  This one fails without the y picture.
- Second derivatives through the scan are pinned as unsupported.
- `n_steps < 1` raises.

The first three fail on the old scan.  `test_scan_converges_to_reference` in
`tests/test_validation_reference.py` now has an absolute bound.
