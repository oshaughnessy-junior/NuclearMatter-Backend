Validation evidence and limits
==============================

This page separates repository-reported numerical checks from outstanding
validation work. It does not certify the full EOS space or reproduce those
scientific runs as part of a documentation build.

Internal reference comparisons
------------------------------

``tests/test_validation_reference.py`` compares the adaptive enthalpy-domain
solver against an independently implemented radius-domain SciPy RK45 reference
with an (H, beta) tidal formulation. The documented smooth-EOS comparisons report
sub-0.1% mass, radius and tidal-deformability differences. This comparison exposed
the original under-resolved fixed-step default and motivated the adaptive backend.

Hybrid comparisons also report sub-0.1% agreement, but both solvers use the same
cited tidal junction formula: agreement checks integration, not independent
confirmation of that formula. The junction-sign check against a smeared transition
is recorded in :doc:`devlog/2026-10-14-tidal-cs2`. That note also reports consistent
interpolation/tidal-source results on 18 selected tables; those bounds apply to
those tests and conventions, not every imported EOS.

External RePrimAnd chronology
-----------------------------

1. :doc:`devlog/2026-06-02d-forward-roadmap` records an initial roughly 200 m
   radius discrepancy, despite much tighter internal solver agreement.
2. :doc:`devlog/2026-06-02e-surface-convention` identifies and quantifies the
   low-density surface-extension effect.
3. :doc:`devlog/2026-06-02f-consistent-interpolation` reports roughly 3–10 m radius
   differences and below 0.1% tidal-deformability differences against RePrimAnd
   1.7 after the surface/interpolation changes.

The initial 200 m mismatch is therefore historical, not an unresolved current
headline. The later dated report is evidence for the compared configurations; it
is not a fresh external verification of all subsequent solver changes or all EOS
families. RePrimAnd remains optional. Zhu delta-TOV, lalsim and MUSES qlimr entries
in plans should not be read as completed external validation.

Gradients and topology
----------------------

Forward-mode derivatives have finite-difference comparisons; see :doc:`physics`,
:doc:`inference` and :doc:`devlog/2026-10-12-tov-jump-ts` for tolerances and domains.
Second derivatives in central enthalpy and EOS parameters are unsupported;
reverse-mode tidal solves are not recommended. Jump handling and crust/core joins
require particular care. Numerical differentiation success does not establish
physical adequacy of a prior or EOS.

Complete disconnected-branch support remains unfinished in ``build_family`` and
single-EOS adapters. RIFT sequence scalar accessors/export retain only the primary
stable rising branch, with branch counts available to detect discarded topology.

Integration evidence
--------------------

:doc:`rift` records a real RIFT CIP smoke run using a 12-EOS sequence remapped
against an interpolant indexed for 10,000 EOS. It demonstrates interface plumbing,
not science-grade per-EOS evidence. Matching the interpolant's EOS indexing is
required for a scientific inference run.

Rotation and cooling checks have separate domains: :doc:`rotating-tabulated`
records a forward RNS comparison, while :doc:`cooling/core_transport_validation`
records conditional fixed-background transport checks. Neither establishes a
fully coupled rotating-star evolution model.

Outstanding work
----------------

:doc:`validation-plan` retains the original acceptance ladder and dated results;
:doc:`roadmap` retains sequencing and follow-up work. Broader external-reference
coverage, tolerance/table-resolution convergence, crust systematics and
phase-transition/twin-branch coverage remain necessary before a precision
inference claim.
