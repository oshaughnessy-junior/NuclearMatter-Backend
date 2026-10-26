# 2026-10-25: CompOSE eos.nb header and the 1e-12 pressure floor

Two defects in `eos/compose.py` at 25e6818, both found by Q2C (Q2C notes/41
and notes/43a).

## 1. `read_compose` kept the eos.nb header as densities

A CompOSE `eos.nb` starts with two lines: the first and the last point index
(`1`, `301` for eos/116; `1`, `1191` for eos/188). The reader expected one
line holding the point count. On a real file that test failed, so it kept both
header numbers as densities: rows 1 and 2 got n = 1 and n = 301 fm⁻³ (eos/116)
or n = 1 and n = 1191 fm⁻³ (eos/188), and every row `i_nb` ≥ 3 got
n_B[i_nb − 2]. The round-trip tests passed because `write_compose` wrote the
one-line form.

Fix: `read_compose_nb` reads the two-line form and maps row `i_nb` to
`values[i_nb − first]`. The one-line form is still read, for files written by
`write_compose` before this change. Any other header raises `ValueError`, as do
a repeated or out-of-range `i_nb` and a table with several T or Y_q indices.
`write_compose` now writes the two-line header. Fortran `D` exponents are
accepted.

| table | old reader | new reader |
|---|---|---|
| eos/116: max n_B ratio | 99.3 | 1 |
| eos/116: M_max, R_1.4, Λ_1.4 | 2.1195, 13.996 km, 1236.7 | 2.0691, 13.003 km, 896.2 |
| eos/188: max n_B ratio | 393.1 | 1 |
| eos/188: M_max, R_1.4, Λ_1.4 | 2.1194, 14.462 km, 1211.2 | 2.0687, 13.543 km, 909.0 |

## 2. The pressure floor made Λ NaN

`eos_table_from_arrays` floored p at 1e-12 MeV/fm³ and then took a running
maximum. A table with k ≥ 2 rows below 1e-12 got k − 1 cells of equal p at the
bottom. Their h-steps are exactly 0, so h[0] = h[1] = … = h[k−1]. The TOV
solve ends at h[0]. There `searchsorted` returns 0, clipped to cell 1, which
has zero width, so in `_derivs_x`:

- the y-picture map dh/dx = (H[1] − H[0]) / (U[1] − U[0]) is 0/0 = NaN, so
  dy/dx and y, hence Λ, are NaN;
- the cell coordinate t is also 0/0, so eps and p are NaN too, but
  `_floor_eps_p` maps NaN to 1e-300 (`where(x > 1e-300, ...)` is False for
  NaN). That is why M and R stayed finite.

Measured on the default RMFT table with the packaged raw SLy rows below
3e-10 fm⁻³ prepended (four rows with p < 1e-12): three zero-width cells, dh/dx
NaN at x = h[0], Λ NaN, M and R finite. Q2C's SLy-crust table B (notes/43a)
has nine such rows.

Fix: positive pressures are no longer floored. The table starts at the first
row with p > 0. Of a run of equal pressures at the bottom, only the first row
is kept. That run is a layer of zero thickness; its Δε could only enter a
junction term below the surface, which the solver does not have. The first
row is kept, not the last, because when the run comes from the running
maximum of a falling p, the first row is the only one whose p is its own; the
others carry p[0] with their own, larger ε. Fewer than two distinct positive
pressures raise `ValueError`.

The solver still returns NaN Λ for any `EOSTable` whose first cell has zero
width, however it was built. It is not guarded in `tov/solver.py` here
because open PR #46 rewrites that part of the file. PR #46 (at b74f84b) keeps
`_u_cell` as it is, so its adaptive path has the same 0/0 at h[0].

Draft follow-up issue (not filed):

> **Title:** TOV adaptive path gives NaN Λ on an EOSTable with a zero-width
> bottom cell
>
> `_u_cell` takes `i = clip(searchsorted(U, x), 1, N − 1)`. At the end point
> x = h[0] with h[0] = h[1], cell 1 has zero width, and
> dh/dx = (H[1] − H[0]) / (U[1] − U[0]) is 0/0. y and Λ are NaN; M and R stay
> finite because `_floor_eps_p` maps NaN eps and p to 1e-300.
> `eos_table_from_arrays` no longer builds such tables (PR #60), but
> `build_h_and_cs2` and `refine_table` accept them. Proposed: start the
> search at the first node with U > U[0] (identical for every table without
> such a cell), or reject the table in `build_h_and_cs2` when eager. Apply it
> after PR #46 lands, since that PR rewrites `_derivs_x`.

## Behaviour changes

- Tables whose first pressure is above 1e-12 and rises at the first cell are
  bit-identical (`2026-10-25-compose-header-floor/compare.py`, outputs and log
  beside it). Checked on the inputs of tests/test_compose.py,
  tests/test_model_diagnostic.py, the default MetaModel table, two
  `draw_bridge_eos` draws and eos/116 and eos/188: identical n, eps, p, h,
  cs2, so M, R and Λ do not move.
- The RMFT table with the raw SLy surface rows: Λ_1.4 NaN → 905.623, R_1.4
  13.99035 → 13.99152 km, M_max +3.7e-7 Msun.
- A leading row with p ≤ 0 is dropped; before, it was raised to 1e-12.
- An `eos.nb` without any header line now raises; before, all its numbers
  were read as densities.
- Any CompOSE download read with `load_compose` before this change, including
  through `scripts/build_eos_sequence.py`, gave a wrong table.
