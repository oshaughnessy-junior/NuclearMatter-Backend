Inference utilities
===================

Fisher forecasting from exact sensitivities
-------------------------------------------

:mod:`nmbackend.inference.fisher` turns measurement uncertainties on NS observables
into forecast constraints on EOS parameters, using forward-mode sensitivities:

.. code-block:: python

   from nmbackend.eos import MetaModelEOS
   from nmbackend.inference import sensitivity_matrix, fisher_covariance

   mm = MetaModelEOS()
   params = {"L_sym": 60.0, "K_sym": -100.0, "cs2_tail": 0.5}
   J, obs_names, pnames = sensitivity_matrix(
       mm, params, ["L_sym", "K_sym", "cs2_tail"],
       masses_lambda=(1.4,), masses_radius=(1.4,), include_mmax=True)
   cov = fisher_covariance(J, sigma=[140.0, 0.6, 0.11])

How the sensitivities are computed (and why)
--------------------------------------------

Differentiating an interpolated :math:`\Lambda(m)` over a TOV family is **wrong**:
the dominant term at fixed mass is the reparameterization of the central enthalpy
as the mass target shifts, and the coarse interpolation can even flip the sign of
the gradient. Instead, fixed-mass rows use the **implicit-function theorem** on
accurate single-star tangents,

.. math::

   \frac{d\Lambda}{d\theta}\Big|_m
   = \frac{\partial\Lambda}{\partial\theta}\Big|_{h_c}
   - \frac{\partial\Lambda}{\partial h_c}
     \frac{\partial M/\partial\theta|_{h_c}}{\partial M/\partial h_c},

and the maximum mass uses the **envelope theorem**
(:math:`dM_{\max}/d\theta = \partial M/\partial\theta` at the peak, where
:math:`\partial M/\partial h_c = 0`).

Validated against a *re-rooted* fixed-mass finite difference (the central enthalpy
Newton-polished onto the target mass for every perturbed EOS):
:math:`d\Lambda/dc_{s,\rm tail}^2` to 0.0%, :math:`d\Lambda/dL_{\rm sym}` to 2.5%,
:math:`dR` to 0.3%, :math:`dM_{\max}` to 0.1% (``tests/test_fisher.py``).

These are the same gradients an HMC/NUTS sampler over EOS parameters consumes.

Worked example
--------------

``examples/fisher_forecast/forecast.py`` forecasts
(:math:`L_{\rm sym}, K_{\rm sym}, c_{s,\rm tail}^2`) constraints from a
GW-tidal + NICER-radius + massive-pulsar measurement set, exposing the
near-perfect :math:`L_{\rm sym}`–:math:`K_{\rm sym}` degeneracy of single-mass
observables (and how multi-mass observables break it).
