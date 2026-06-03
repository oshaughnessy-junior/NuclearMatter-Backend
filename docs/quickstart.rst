Quickstart
==========

One star, one EOS
-----------------

.. code-block:: python

   from nmbackend.eos import MetaModelEOS
   from nmbackend.tov.solver import solve_structure

   mm = MetaModelEOS()
   table = mm.eos_table({"L_sym": 60.0, "cs2_tail": 0.5})
   pt = solve_structure(table, 0.20 * float(table.h[-1]))   # central enthalpy
   print(pt.M_msun, pt.R_km, pt.Lambda)

A mass–radius–:math:`\Lambda` family
------------------------------------

.. code-block:: python

   from nmbackend.tov import build_family, m_max, lambda_of_mass
   from nmbackend.tov.family import default_hc_grid

   fam = build_family(table, default_hc_grid(table, n_models=120))
   print(m_max(fam), lambda_of_mass(fam, 1.4))

Gradients (the point of the package)
------------------------------------

.. code-block:: python

   import jax
   dLam_dLsym = jax.jacfwd(
       lambda L: solve_structure(mm.eos_table({"L_sym": L}), 0.19).Lambda)(60.0)

For fixed-mass sensitivities use :func:`nmbackend.inference.sensitivity_matrix`
(implicit-function theorem; see :doc:`inference`) rather than differentiating
through a family interpolation.

An EOS sequence for RIFT
------------------------

.. code-block:: bash

   # parametric grid -> tabular + legacy files
   python scripts/build_eos_sequence.py --model metamodel \
       --grid "L_sym=40,50,60,70;cs2_tail=0.3,0.4,0.5" \
       --out seq.nss.h5 --legacy seq.landry.h5

   # modern nonparametric prior (bridge to pQCD), with compressed PCA copy
   python scripts/draw_bridge_prior.py --n 1000 --out prior.nss.h5 \
       --pca prior.pca.h5

   # hand either file to the RIFT tabular pipeline
   util_RIFT_pseudo_pipe.py --assume-matter \
       --internal-tabular-eos-file prior.nss.h5 ...

See :doc:`rift` for the full pipeline call chain and the GW170817 example.
