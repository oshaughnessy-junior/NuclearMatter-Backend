NuclearMatter-Backend
=====================

Research software for neutron-star equations of state, stellar structure, tidal
deformability and EOS inference. The JAX core supports selected differentiable
paths; RIFT adapters and sequence I/O are implemented, while gwkokab is planned.

Highlights
----------

* **EOS models:** empirical-parameter metamodel, nonlinear and density-dependent
  RMF, standard/extended Skyrme, hybrid transitions and tabular readers.
* **Structure and tides:** adaptive enthalpy-domain TOV, tidal junctions,
  M–R–Lambda sequences and independent SciPy reference checks.
* **Priors and inference:** Brownian-bridge EOS draws, PCA compression/coefficient
  priors, fixed-mass sensitivities, Fisher forecasts and laboratory likelihoods.
* **Interfaces:** RIFT tabular workflows and separate external RNS/NSCool bridges,
  with conditional JAX thermal transport on frozen backgrounds.

Start with :doc:`capabilities` and :doc:`validation`. Numerical checks have scoped
coverage: the historical RePrimAnd radius mismatch was reduced to a reported
3–10 m in selected comparisons, while branch coverage, broader external checks
and fully coupled rotating cooling remain open. See each guide for its limits.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   capabilities
   validation
   physics
   priors
   representations
   rift
   inference
   rotating
   rotating-tabulated
   cooling/native_core
   cooling/core_transport_validation

.. toctree::
   :maxdepth: 1
   :caption: API reference

   api

.. toctree::
   :maxdepth: 1
   :caption: Design notes

   rift-sequence-audit
   rift-tabular-pipeline
   muses-integration
   backlog

.. toctree::
   :maxdepth: 1
   :caption: Development log
   :glob:

   architecture
   roadmap
   validation-plan
   environment
   overnight-status
   devlog/*
