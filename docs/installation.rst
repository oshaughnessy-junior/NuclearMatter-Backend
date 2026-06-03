Installation
============

Requirements
------------

* Python :math:`\ge` 3.10
* `jax <https://jax.readthedocs.io>`_ (CPU is fine; float64 is enabled by the
  package), `diffrax <https://docs.kidger.site/diffrax/>`_, ``numpy``, ``h5py``.
* Optional: ``pyreprimand`` for the independent reference-solver validation;
  a RIFT + LALSuite environment for the tabular-pipeline integration.

From source
-----------

.. code-block:: bash

   git clone <repo-url> NuclearMatter-Backend
   cd NuclearMatter-Backend
   pip install -e .

or, without installing, prepend the source tree:

.. code-block:: bash

   export PYTHONPATH=/path/to/NuclearMatter-Backend/src:$PYTHONPATH

Verifying the install
---------------------

.. code-block:: bash

   python -m pytest -q tests/

The suite covers solver validation against an independent radius-domain integrator,
AD-gradient checks against finite differences, RMFT saturation properties, phase
transitions, sequence-format round-trips, the Brownian-bridge prior invariants, and
the Fisher utility.
