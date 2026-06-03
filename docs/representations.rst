Sequence representations
========================

An EOS *sequence* (an indexed family of EOS with their stellar structure) is stored
behind the :class:`nmbackend.rift.SequenceRepresentation` accessor protocol::

    lambda_of_m(m, e)   radius_of_m(m, e)   m_max(e)   ordering_values(m_ref, kind)

Consumers — the RIFT tabular pipeline, the generator CLIs, plotting — depend only on
this protocol. The on-disk representation is a pluggable detail, dispatched by the
HDF5 ``representation`` attribute via :func:`nmbackend.rift.open_sequence`:

``tabular_hc/1`` — :class:`nmbackend.rift.NSSequence`
-----------------------------------------------------

The native format: every quantity stored versus **central pseudo-enthalpy**
:math:`h_c` (monotone along the sequence), so :math:`M(h_c), R(h_c), \Lambda(h_c)`
are single-valued and branch structure (twin stars) is explicit via a ``stable``
flag. Lookups are branch-aware. Written by :func:`nmbackend.rift.write_nss`.

``pca_hc/1`` — :class:`nmbackend.rift.PCASequence`
--------------------------------------------------

Compressed, *functional-space* storage: a per-channel PCA of the
:math:`M(u), R(u), \log\Lambda(u)` curves (mean + basis + per-EOS coefficients).
At 5–6 components the reconstruction matches the tabular file to
:math:`\sim 0.1\%` in :math:`\Lambda(1.4)`, metres in :math:`R`, and the
compression grows with the number of draws (:math:`\sim 12\times` at
:math:`10^4` EOS). Because any coefficient vector reconstructs a smooth sequence,
this is also the seam for Gaussian-process or empirical-interpolant priors over the
sequence space — modelling functions, not replaying draws. Produce with
:func:`nmbackend.rift.write_pca_from_nss` or ``scripts/compress_sequence.py``.

Legacy ``EOSSequenceLandry``
----------------------------

The format consumed by released RIFT (and the published LCEHL files).
:func:`nmbackend.rift.write_landry_legacy` emits it exactly (stable branch only, so
RIFT's single-valued :math:`M \to \Lambda` interpolation is correct for
phase-transition EOS), and ``scripts/convert_landry_to_nss.py`` converts existing
files to the native format.

RIFT-side auto-detection
------------------------

``RIFT/physics/EOSManager.py`` gained ``EOSSequenceFromFile``, which inspects the
``representation`` / ``schema_version`` attributes and returns ``EOSSequencePCA``,
``EOSSequenceNMB``, or ``EOSSequenceLandry`` — all exposing the identical consumer
API, so any of the three file types drops into the pipeline unchanged.
