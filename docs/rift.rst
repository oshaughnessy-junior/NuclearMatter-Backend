RIFT integration
================

Tabular per-EOS evidence pipeline
---------------------------------

The driver entry point is::

    util_RIFT_pseudo_pipe.py --assume-matter --internal-tabular-eos-file SEQ.h5 ...

which flows down to CIP (``util_ConstructIntrinsicPosterior_GenericCoordinates.py``)
as ``--tabular-eos-file SEQ.h5 --input-eos-index N``. CIP reduces each EOS to its
stable :math:`M \to \Lambda` map, evaluates :math:`\Lambda(m_1), \Lambda(m_2)`, and
marginalizes the trained likelihood interpolant (``all.net``) over the binary
masses — the per-EOS evidence. The EOS index is the sampled ordering coordinate, so
one run sweeps the whole sequence. The full call chain is documented in
:doc:`rift-tabular-pipeline`.

Any of the three sequence formats (:doc:`representations`) can be passed; CIP
auto-detects via ``EOSSequenceFromFile``.

Live smoke test
---------------

The integration was validated end-to-end against a real GW170817 ``all.net``
(see ``examples/gw170817_tabular/SMOKE_TEST_RESULT.md``):

* legacy path reproduces the stored reference evidence
  (:math:`\ln L` 463.05 vs 464.71, GMM/MC scatter at low :math:`n_{\rm eff}`);
* the native ``NSSequence`` path runs the same marginalization through
  ``EOSSequenceNMB`` and converges (:math:`\ln L = 457.77 \pm 0.39` on a remapped
  12-EOS smoke configuration).

A *science-grade* run requires a sequence with the same cardinality as the
``all.net`` EOS indexing (e.g. :math:`10^4` draws from the bridge prior).

Per-EOS adapter
---------------

For single-EOS use inside RIFT-style codes,
:mod:`nmbackend.adapters` provides an ``EOSConcrete``-compatible adapter exposing
``lambda_from_m`` etc. from a backend EOS table.

Future
------

The 1-D :math:`\Lambda(m_{\rm ref})` ordering locks the inference to a single
stable branch. Because the native format stores everything versus :math:`h_c`, a
future driver can parameterize the likelihood directly along central enthalpy
(path integral in :math:`h_c`), representing disconnected twin-star branches
natively. Logged in :doc:`backlog`.
