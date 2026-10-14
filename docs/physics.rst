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
*derived*, not independently interpolated. :math:`c_s^2` is interpolated from
accurate node values and is allowed to be non-monotone (phase transitions); it is
never monotonized. See :func:`nmbackend.eos.base.hermite_eos_at_h`.

Tables that stop near neutron drip are continued to the surface with a consistent
polytropic tail (:func:`nmbackend.eos.base.extend_to_surface`); this surface
*convention* is worth :math:`\sim 100` m in radius and was the key to closing the
gap to the RePrimAnd reference solver (final agreement: few metres in :math:`R`).

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
