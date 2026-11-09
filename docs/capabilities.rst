Capabilities and boundaries
===========================

NuclearMatter-Backend combines EOS construction, stellar observables and inference
interfaces. The inventory below describes implemented code, not uniform scientific
validation or automatic-differentiation coverage. See :doc:`validation` for evidence
and :doc:`physics` for numerical and differentiation limits.

EOS construction
----------------

* ``MetaModelEOS``: empirical nuclear parameters, beta equilibrium and a causal
  constant-sound-speed high-density tail.
* ``RMFTEOS``: nonlinear sigma–omega–rho mean field (GM1 default).
* ``DDRMFEOS``: density-dependent RMF with DD2/DDME2 sets and rearrangement terms.
* ``SkyrmeEOS``: standard and extended Skyrme forms, including SLy4/BSkG3 sets.
  High-density acausality and negative effective masses are not automatically
  corrected; inspect diagnostics and choose a physically appropriate domain.
* ``HybridEOS``: Maxwell first-order transition plus a constant-sound-speed quark
  branch; ``PolytropeEOS`` supplies an analytic EOS for solver checks.
* ``TabularEOS`` and readers: CompOSE, simple tables and MUSES-CMF CSV output.
  MUSES is an out-of-band provider/reference, not an integrated hot-path service.

Crust matching, leptons and composition utilities accompany these models.
Crust choices and matching remain physical systematics, even when numerical
solvers agree. The primary implementations are in ``src/nmbackend/eos/``.

Structure, sequences and inference
----------------------------------

The adaptive diffrax TOV solver computes mass, radius and tidal deformability;
the hybrid solver applies a tidal junction at the phase boundary. Families expose
central-enthalpy sequences and derivatives. Complete twin-star branch handling in
``build_family`` and the single-EOS adapter is still incomplete.

:doc:`priors` covers constrained Brownian-bridge draws and Gaussian PCA coefficient
priors. :doc:`representations` describes native tabular, PCA and legacy Landry
storage. :doc:`rift` describes the implemented adapter and RIFT pipeline support,
including the restricted smoke-test evidence. Scalar sequence accessors retain the
primary stable rising branch and expose branch counts; they do not integrate over
all disconnected branches. **gwkokab remains planned**; an optional dependency does
not supply an implemented adapter.

:doc:`inference` covers fixed-mass implicit-function sensitivities and maximum-mass
envelope-theorem derivatives, Fisher forecasts, laboratory likelihoods and ejecta
fitting. These tools have method-specific domains and do not establish a validated
observational posterior.

External rotation and cooling paths
-----------------------------------

* :doc:`rotating` and :doc:`rotating-tabulated` describe a pinned external RNS
  runner, EOS conversion, geometry exports and recorded forward comparisons.
  The tabulated GM1 result is not a convergence/precision claim, conserved-baryon
  sequence or rotating-stability test. RNS is not vendored or relicensed here.
* The NSCool driver in ``src/nmbackend/cooling/`` wraps a separately obtained
  external code. Its README records source/build details, model-curve checks and
  time-step/physics limitations; the referenced upstream tarball has no stated
  license. The wrapper does not grant redistribution rights to that code.
* :doc:`cooling/native_core` and :doc:`cooling/core_transport_validation` describe
  JAX thermal transport with scalar-reference checks on fixed backgrounds.
  Conditional transport does not include EOS-to-equilibrium response or establish
  full rotating-star cooling evolution.

Credit
------

The repository's ``ATTRIBUTION.md`` records authors, primary scientific references,
reused data and external licensing boundaries. In particular, independently
implemented equations and interface compatibility do not imply copied upstream
code or endorsement by the referenced projects.
