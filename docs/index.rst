NuclearMatter-Backend
=====================

A differentiable (JAX) neutron-star equation-of-state and TOV toolkit for EOS
inference, interfacing the RIFT parameter-estimation pipeline (per-EOS evidence and
the tabular EOS hyperpipeline) and population-level inference.

Highlights
----------

* **Differentiable physics core** — pseudo-enthalpy TOV solver (adaptive
  diffrax/Dopri5) with in-loop tidal Love number; forward-mode AD reproduces the
  perturbative :math:`\delta`-TOV sensitivities. Validated against RePrimAnd to
  :math:`\sim` few-metre radius agreement.
* **Physical EOS models** — nuclear-empirical-parameter metamodel with causal
  high-density extension; self-consistent RMFT (:math:`\sigma\!-\!\omega\!-\!\rho`,
  GM1, muons in :math:`\beta`-equilibrium); Maxwell hybrid (phase-transition) EOS
  with tidal junction condition; CompOSE / MUSES-CMF tabular ingestion.
* **Modern nonparametric prior** — the constrained Brownian-bridge prior
  (Gorda, Komoltsev, Kurkela & Sunde, arXiv:2512.18044): stable, causal,
  thermodynamically consistent draws by construction, bridging the nuclear band to
  perturbative QCD.
* **Pluggable sequence representations** — tabular draws, compressed PCA
  (functional-space) storage, and the legacy ``EOSSequenceLandry`` format behind a
  single accessor protocol; RIFT auto-detects all three.
* **Inference utilities** — Fisher forecasting from implicit-function-theorem
  sensitivities (validated against re-rooted finite differences).

.. toctree::
   :maxdepth: 2
   :caption: User guide

   installation
   quickstart
   physics
   priors
   representations
   rift
   inference

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
