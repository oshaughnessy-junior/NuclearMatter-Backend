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
resolution. ``h_rule="trapezoid"`` reproduces the old tables.

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
:math:`\Delta y = -4\pi r^3 \Delta\varepsilon/(m + 4\pi r^3 p)` (Takátsy & Kovács
2020) and is counted correctly without one. No :math:`c_s^2` clipping is involved.

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

Not covered: a density jump at *fixed* pressure (a first-order transition, or the
pressure plateaus of ``allow_unstable_core=True`` tables) has zero width in
:math:`h` and needs the junction condition (``tov.solve_hybrid``).

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
