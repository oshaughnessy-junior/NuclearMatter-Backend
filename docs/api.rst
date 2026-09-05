API reference
=============

EOS models
----------

.. automodule:: nmbackend.eos.base
   :members: EOSTable, EOSModel, hermite_eos_at_h, build_h_and_cs2, extend_to_surface

.. automodule:: nmbackend.eos.metamodel
   :members: MetaModelEOS

.. automodule:: nmbackend.eos.rmft
   :members: RMFTEOS, saturation_properties

.. automodule:: nmbackend.eos.hybrid
   :members: HybridEOS, TransitionInfo

.. automodule:: nmbackend.eos.compose
   :members: load_compose, load_simple, load_cmf, write_compose,
             eos_table_from_arrays, TabularEOS

TOV solver and families
-----------------------

.. automodule:: nmbackend.tov.solver
   :members: solve_structure, TOVPoint

.. automodule:: nmbackend.tov.family
   :members: build_family, Family, m_max, lambda_of_mass, radius_of_mass,
             default_hc_grid, dhc_dM

Priors
------

.. automodule:: nmbackend.priors.bridge
   :members: kk_volume, sample_point_in_volume, fractal_refine, diffuse_mu_of_n,
             draw_bridge_eos, draw_prior_ensemble, pqcd_anchor

Sequence representations and RIFT I/O
-------------------------------------

.. automodule:: nmbackend.rift.sequence
   :members: SequenceRepresentation, open_sequence, write_nss, NSSequence,
             write_landry_legacy

.. automodule:: nmbackend.rift.compressed
   :members: fit_sequence_pca, write_pca, write_pca_from_nss, PCASequence,
             compression_ratio

Inference
---------

.. automodule:: nmbackend.inference.fisher
   :members: eos_observables, sensitivity_matrix, fisher_matrix, fisher_covariance

Laboratory likelihoods (MB-N2)
------------------------------

.. automodule:: nmbackend.lab_likelihood
   :members: psym, DensityTaggedConstraint, SigmaModelPrior,
             DensityTaggedPressureLikelihood, CorrelatedSymGaussian,
             load_scenario_ladder, scenario_by_name, gaussian_from_scenario,
             density_tagged_from_scenario
