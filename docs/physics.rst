Physics core
============

TOV solver in pseudo-enthalpy
-----------------------------

The stellar-structure equations are integrated in the pseudo-enthalpy variable
:math:`h(p) = \int_0^p dp'/(\varepsilon + p')` (Lindblom 1992), which removes the
surface coordinate singularity. Alongside :math:`(r, m)` the tidal variable
:math:`y` (Postnikov, Prakash & Lattimer 2010) is carried in-loop, giving the
:math:`\ell=2` Love number :math:`k_2` and the dimensionless tidal deformability
:math:`\Lambda`.

Two backends sit behind :func:`nmbackend.tov.solver.solve_structure`:

* ``diffrax`` (default): adaptive Dopri5 + PID controller. Accurate, recommended;
  forward-mode AD flows through it (``diffrax.ForwardMode``), reproducing the
  perturbative :math:`\delta`-TOV sensitivity equations. Steps stop at every EOS
  table node (``jump_ts``), where the right-hand side has slope kinks; this makes
  the EOS-parameter tangents converge with ``rtol`` at the level of the values.
  The start of the integration is frozen against differentiation; this makes the
  :math:`h_c` tangent converge like ``rtol``, about 1e3 times above the values
  (~1e-4 at the default ``rtol``). Only *first* derivatives in :math:`h_c` are
  supported: :math:`\partial^2/\partial h_c^2` is wrong by orders of magnitude,
  while mixed :math:`\partial^2/\partial h_c\,\partial\theta` is fine.
* ``scan``: fixed-step RK4 ``lax.scan``; lightweight, reverse-mode-capable for
  structure-only solves.

Thermodynamically consistent EOS evaluation
-------------------------------------------

EOS tables are evaluated at arbitrary :math:`h` through a cubic-Hermite
interpolation of :math:`p(h)` whose node slopes are set to the exact thermodynamic
value :math:`dp/dh = \varepsilon + p`, so :math:`\varepsilon(h) = dp/dh - p` is
*derived*, not independently interpolated. See
:func:`nmbackend.eos.base.hermite_eos_at_h`.

The node pseudo-enthalpies are :math:`\int dp/(\varepsilon + p)` along the
log-linear (piecewise-polytrope) :math:`\varepsilon(p)` of each cell, by
12-point Gauss–Legendre (``build_h_and_cs2(..., h_rule="loglinear")``, the default
since 2026-10-14). The trapezoid rule used before is inconsistent with the
Hermite's node slopes by :math:`O((\Delta\ln\varepsilon)^2)` per cell; the cubic
then overshoots between coarse crust nodes, which put :math:`R_{1.4}` 4–10 m high
on the MetaModel and GM1 tables (41 m on a 250-node SLy table) at any core
resolution. ``h_rule="trapezoid"`` reproduces the old node :math:`h` only, for
table comparison. It does not reproduce the old :math:`\Lambda`, and it is not
valid with the new tidal term: on the packaged SLy crust its Hermite
:math:`\varepsilon(h)` is non-monotone (:math:`d\varepsilon/dh` down to
:math:`-1.4\times10^5(\varepsilon+p)`); use the default rule with the tidal term.

Tables that stop near neutron drip are continued to the surface with a consistent
polytropic tail (:func:`nmbackend.eos.base.extend_to_surface`); this surface
*convention* is worth :math:`\sim 100` m in radius and was the key to closing the
gap to the RePrimAnd reference solver (final agreement: few metres in :math:`R`).

Tidal term
----------

The Love-number equation for :math:`y` contains
:math:`(\varepsilon + p)/c_s^2 = (\varepsilon + p)\,d\varepsilon/dp`. With
:math:`dp/dh = \varepsilon + p` this is exactly :math:`d\varepsilon/dh`, and the
solver takes it as the analytic derivative of the *same* Hermite cell that gives
:math:`\varepsilon(h)` (:func:`nmbackend.eos.base.hermite_eos_tidal_at_h`). Across
every table cell the tidal source therefore integrates to exactly the
:math:`\Delta\varepsilon` that the structure equations see, however steep the
cell. A steep but finite layer is the smooth limit of the junction condition
:math:`y_{\rm out} = y_{\rm in} - 4\pi r^3 \Delta\varepsilon/(m + 4\pi r^3 p)`, with
:math:`\Delta\varepsilon = \varepsilon_{\rm in} - \varepsilon_{\rm out} > 0` the drop
outward, so :math:`y` *falls* across it. Its Newtonian limit is the self-bound
surface term :math:`\Delta y = -3`, and it is counted correctly without a junction
(Takátsy & Kovács 2020).

Two guards act on the tidal term.

* A **jump cell** is a density jump at (nearly) fixed pressure: width at most
  :math:`10^3` ulp in :math:`h`, or chord :math:`\Delta p/\Delta\varepsilon \le 10^{-8}`
  at any width.

  - Examples: the low-table/bridge join of every ``draw_bridge_eos`` table,
    exact plateaus, and plateaus nudged to strict increase.
  - The chord criterion leaves no window between this test and the clip below.
    A width-only test let cells 1e3 ulp to ~3e-11 wide lose most of their
    :math:`\Delta\varepsilon` to the clip, a 1.2% drop in :math:`\Lambda`.
  - The tidal term is 0 inside a jump cell. Its :math:`\Delta\varepsilon` is
    carried instead by the bubble :math:`(q+1)t^q\,\Delta\varepsilon/\Delta h`,
    :math:`q = 32`, in the regular cell just below. That is the junction,
    smeared over the top :math:`\sim 1/(q+2)` of that cell.
  - The smearing error scales with the width of that cell: +1.4e-5 to
    +2.8e-5 in :math:`\Lambda` on hybrid tables, 1.8e-6 on bridge draws.
  - The classification is a hard mask. Where a parameter change moves a cell's
    chord through 1e-8, :math:`\Lambda` steps by that smearing error, and its
    parameter derivative carries a delta function there. No shipped generator
    moves a cell through that threshold.
  - Gradients through a jump cell: the concentrated bubble needs a tighter
    tolerance. On a bridge draw, AD vs FD of :math:`d\Lambda/d\theta` (for a
    parameter that moves the jump) is 3.2e-6 at the default rtol 1e-7 and
    1.3e-7 at rtol 1e-9; use rtol :math:`\le 10^{-9}` there. The smearing also
    biases that derivative against the exact junction, by 1.2e-4 on the bridge
    draw (the value bias is only 1.8e-6). A Fisher row through such a join
    carries this bias.

* Elsewhere the term is capped at :math:`10^{10}(\varepsilon+p)`; the chord test
  keeps every cell that could reach the cap a jump cell. It is **not** floored at
  0, so a regular cell whose cubic :math:`\varepsilon(h)` dips (a soft cell over a
  large :math:`\varepsilon` ratio) still integrates to exactly its
  :math:`\Delta\varepsilon`. A zero floor over-counted such cells, by up to 7.6%
  at :math:`\Delta\varepsilon = 3000` MeV/fm³ (only unstable-branch stars were
  affected). With the default node :math:`h`, the cap acts on none of the
  tables of the devlog.

Until 2026-10-14 the tidal term used the node values ``EOSTable.cs2``
(finite differences :math:`\Delta p/\Delta\varepsilon`) interpolated linearly in
:math:`h`. That is not the derivative of the Hermite :math:`\varepsilon(h)`.
Across a stiff joint (pressure :math:`\times 2.7` over an 11% rise in
:math:`\varepsilon`) the stiff node value leaked into the soft cell, the tidal
source missed part of the energy jump (a "phantom" :math:`\Delta\varepsilon`), and
:math:`\Lambda_{1.4}` came out 2–10% low (6.2% on Q2C's FSUGold2 crust–core join).
The consistent term alone is not enough on coarse smooth tables: it exposes the
trapezoid-:math:`h` overshoot above (a 250-node SLy table went from +1.1% to
+2.2%), which the old node :math:`c_s^2` partly cancelled. With both changes,
:math:`\Lambda_{1.4}` agrees with an independent radius-domain reference on the
same nodes (log-linear :math:`\varepsilon(p)`, consistent :math:`c_s^2`) to
:math:`\le 6\times10^{-6}` on 18 test tables, smooth and steep
(``docs/devlog/2026-10-14-tidal-cs2.md``).
``EOSTable.cs2``, and the :math:`c_s^2` returned by ``at_h``, are kept as
diagnostics (e.g. the central sound speed); they are not the derivative of
:math:`\varepsilon(h)`.

The tidal term jumps at the nodes (the Hermite is :math:`C^1`). The adaptive solve
stops at the nodes and carries :math:`y` in a node-anchored picture (equal to
:math:`h` in value) in which the nodes do not move with the EOS parameters, so
forward-mode tangents keep the boundary terms of those jumps; :math:`r, m` stay
in :math:`h`. First derivatives only: second derivatives in EOS parameters were
never accurate and are not supported. ``jump_nodes=False`` is no longer cheaper
for tidal solves and is not recommended there.

A density jump at *fixed* pressure inside one table (the bridge join, or the
pressure plateaus of ``allow_unstable_core=True`` tables) has zero width in
:math:`h` (or nearly), and gets the smeared junction above. A genuine first-order transition
between two phases should be built as a ``HybridEOS``, whose ``tov.solve_hybrid``
applies the junction condition exactly at the transition.

EOS models
----------

* **Metamodel** (:class:`nmbackend.eos.metamodel.MetaModelEOS`) — Taylor expansion
  of the energy per nucleon in the nuclear empirical parameters
  (:math:`E_{\rm sat}, K_{\rm sat}, Q_{\rm sat}; E_{\rm sym}, L_{\rm sym},
  K_{\rm sym}, Q_{\rm sym}`), :math:`\beta`-equilibrium with relativistic
  electrons, and a constant-:math:`c_s^2` causal high-density extension
  (``cs2_tail``) above ``n_match``.
* **RMFT** (:class:`nmbackend.eos.rmft.RMFTEOS`) — self-consistent
  :math:`\sigma\!-\!\omega\!-\!\rho` mean-field (GM1 default), robust first-root
  bracketed-bisection scalar gap solve, electrons *and muons* in
  :math:`\beta`-equilibrium. Validated against all six published GM1 saturation
  properties and :math:`M_{\max}`. Adding further parameter sets requires
  verifying the coupling normalization against the published saturation point
  (see the backlog note).
* **Hybrid / phase transitions** (:class:`nmbackend.eos.hybrid.HybridEOS`) —
  Maxwell construction with the tidal junction condition at the density
  discontinuity.
* **Tabular ingestion** (:mod:`nmbackend.eos.compose`) — CompOSE directories,
  simple three-column tables, and MUSES-CMF CSV output.

Validation
----------

``tests/test_validation_reference.py`` compares against an independent
radius-domain solver; ``scripts/validate_against_reprimand.py`` drives the
RePrimAnd cross-check (matched surface convention, stable/unstable-branch aware).
``tests/test_ad_gradients.py`` and ``tests/test_tov_jump_ts.py`` validate
forward-mode tangents against finite differences (parameter column ~3e-5 and
:math:`h_c` column ~1.2e-4 at the default ``rtol=1e-7``; below 1e-6 at
``rtol=1e-9``), and pin the unsupported second :math:`h_c` derivative.
