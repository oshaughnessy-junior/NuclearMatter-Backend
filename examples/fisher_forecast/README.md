# Fisher forecast of EOS-parameter constraints

`forecast.py` turns measurement uncertainties on neutron-star observables into forecast
constraints on nuclear EOS parameters, using the differentiable backend's forward-mode
sensitivities (`nmbackend.inference.fisher`).

```
PYTHONPATH=src python examples/fisher_forecast/forecast.py
```

## What it does

For a fiducial metamodel EOS and a measurement set — Λ(1.4) (GW-tidal),
R(1.4) (NICER-class), M_max (massive pulsar) — it computes the sensitivity matrix
`dObs/dθ` (via the implicit-function theorem for fixed-mass observables, the envelope
theorem for M_max), forms the Fisher matrix `F = Jᵀ C⁻¹ J`, and reports the forecast
1σ parameter constraints and their correlations.

## Reading the example output

With this particular three-observable set the forecast shows `cs2_tail` (high-density
stiffness) well constrained, but `L_sym` and `K_sym` almost perfectly anti-correlated
(corr ≈ −1) and individually poorly constrained. That degeneracy is physical: a single
Λ(1.4)+R(1.4) pair constrains a combination of the isovector parameters, not each
separately — so the marginalized errors are large while the well-measured *direction*
in parameter space is tight. Adding observables at other masses (e.g. Λ at 1.2 and
1.6 M⊙) or a second radius breaks the degeneracy; edit `obs_kw` / `param_names` to
explore.

## Caveats

This is a linearized Gaussian *forecast*, not a posterior — it is exact in the
sensitivities (validated vs finite differences in `tests/test_fisher.py`: dM_max ~0.1%,
dR ~0.3%, dΛ ~4%) but assumes locally-linear observables and Gaussian errors. For the
full posterior, feed the EOS sequence through the RIFT tabular pipeline
(`examples/gw170817_tabular/`). The same `sensitivity_matrix` gradients are what an
HMC/NUTS sampler would consume.
