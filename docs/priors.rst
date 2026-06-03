EOS priors
==========

Constrained Brownian-bridge prior (recommended)
-----------------------------------------------

:mod:`nmbackend.priors.bridge` implements the prior of Gorda, Komoltsev, Kurkela &
Sunde (`arXiv:2512.18044 <https://arxiv.org/abs/2512.18044>`_): nonparametric EOS
draws that are **mechanically stable, causal, and thermodynamically consistent by
construction**, with both the chiral-EFT-scale and perturbative-QCD constraints
built into the sampler (no shooting, no accept/reject against a QCD likelihood).

The three ingredients, faithful to the paper:

1. **KK 3-volume** (Komoltsev & Kurkela 2022): between two anchor triplets
   :math:`\beta = (\mu, n, p)`, every consistent intermediate point lies inside an
   analytic volume — :func:`nmbackend.priors.bridge.kk_volume`.
2. **Self-similar refinement**: recursive uniform sampling inside the volume
   (:func:`~nmbackend.priors.bridge.fractal_refine`), filling the entire allowed
   function space with structure on all scales.
3. **Diffusion smoothing**: heat-kernel smoothing with log-constant correlation
   length :math:`\sigma/n = c`. We formulate it as constant-coefficient diffusion
   of :math:`\ln\mu` versus :math:`\ln n` — mathematically equivalent to the
   paper's :math:`D(n) \propto n^2` choice, but with an **exact maximum principle
   for** :math:`c_s^2 = d\ln\mu/d\ln n`: the sound speed provably never leaves
   :math:`[0, 1]`. Chemical-potential anchors are pinned exactly (Dirichlet); the
   pQCD pressure endpoint moves by :math:`\sim` a few per cent — well inside the
   renormalization-scale band, as in the paper.

Anchors: the low end is **this backend's own nuclear band** (metamodel at
:math:`2\,n_{\rm sat}` with NEPs sampled per draw), making the prior self-contained;
the high end is the partial-N3LO pQCD point at :math:`\mu = 2.6` GeV (Gorda et al.
2021 values), with the scale parameter :math:`X` sampled log-uniformly in
:math:`[1, 4]`.

.. code-block:: bash

   python scripts/draw_bridge_prior.py --n 10000 --seed 1 \
       --out bridge_prior.nss.h5 --pca bridge_prior.pca.h5

Recovering the published LCEHL prior
------------------------------------

For paper comparisons, ``scripts/convert_landry_to_nss.py`` replays the published
LCEHL ``EOSSequenceLandry`` files through the same pipeline — either by repackaging
the published TOV results directly (fast) or by re-solving the LCEHL microphysical
tables with this backend's TOV (``--mode resolve``, apples-to-apples).

Parametric priors
-----------------

``scripts/build_eos_sequence.py`` draws metamodel / RMFT parameter grids or explicit
parameter files — useful for controlled studies and for the Fisher forecasting in
:doc:`inference`.
