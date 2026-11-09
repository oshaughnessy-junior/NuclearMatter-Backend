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

The v1 scalar accessors deliberately retain the historical **primary-branch**
contract.  ``NSSequence`` and ``PCASequence`` split central-coordinate samples into
contiguous stable, mass-rising runs and use the first run for ``lambda_of_m``,
``radius_of_m``, and ``m_max``; they never bridge an unstable interval into a later
twin-star branch.  ``n_stable_branches(e)`` (and the cached
``stable_branch_counts`` array) exposes discarded topology so callers can reject or
route multi-branch EOS explicitly.  Marginalizing over all branches requires the
central-enthalpy inference described below.

A secondary, opt-in multi-branch contract uses the same accessors with a
``branch`` argument.  ``branch="primary"`` (the default) is the v1 contract.  An
integer selects one stable run (0 is the primary one).  ``branch="all"`` makes
``m_max`` the maximum stable mass over every run, and makes ``lambda_of_m`` and
``radius_of_m`` return one value per run (NaN where the mass is outside it; an
empty array if the EOS has no run).  A negative integer counts from the densest
run.  Any other ``branch`` value raises ``ValueError``.  ``branches(e)`` returns
the runs themselves.

Where the runs come from, and their limits:

- ``NSSequence`` uses the stored ``stable`` flag, which ``write_nss`` sets from
  M(h_c) on the grid (``_sequence_stable``).  It excludes rows before the
  lightest row at or before the heaviest stable row.  If a twin-star dip is
  lighter than every earlier row and the later branch is heavier, the first
  branch is excluded as well, and ``n_stable_branches`` does not count it.
- ``PCASequence`` has no stored flag.  It splits the reconstructed M(h_c) into
  rising runs, so for such an EOS the two readers can disagree on the branches.
- The grid of ``write_nss`` ends two steps of a 40-point h_c scan past the last
  maximum of M(h_c), but never beyond ``frac_hi`` of the table's maximum
  enthalpy.  A maximum within two scan steps of that bound, or a stable branch
  narrower than one scan step beyond the last resolved maximum, is clipped.  On
  a table that ends before the TOV maximum, the stored ``m_max`` is the mass at
  the table end.

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
