# Rotating equilibrium and conditional transport

`nmbackend.rotating` builds and runs an **external** pinned RNS v2 solver.
It does not vendor upstream source and does not differentiate the C process.
The initial supported runner is a relativistic polytrope in **K=G=c=1** units.
Its central input is total energy density. The coordinate radius and
circumferential equatorial radius are distinct outputs.

```
git clone https://github.com/cgca/rns.git
git -C rns checkout 87ea68246ee990256f94aa91a0506f54d78dd8d0
PYTHONPATH=src python examples/rns_reference.py rns/source/rns.v2.0 reference-run
RNS_SOURCE=$PWD/rns/source/rns.v2.0 RNS_REQUIRED=1 python -m pytest tests/test_rotating_rns.py
PYTHONPATH=src python examples/rotating_transport.py reference-run/run/rns_rotating.csv transport.json
```

Builds write a separate generated driver, compiler log and source/binary hash
manifest. Runs use fresh output directories and record arguments and exported
metadata. The loader rejects missing fields, nonfinite values, bad grid ordering,
superluminal matter, inconsistent metric/velocity conventions and configurations
above the exported mass-shedding frequency. These checks do not certify
convergence, stability or a continuum accuracy target. A failure in the external
process remains a failure; there is no fabricated fallback export.

The imported `nmbackend.thermal.diffusion` prototype advances scalar transport
on frozen axisymmetric geometry. Its supplied coefficients are temperature
independent and its boundaries insulated. The example uses a spherical inner
cutout, **not the oblate stellar surface**, with constant dimensionless Cv and K.
Conditional transport derivatives do not include EOS-to-equilibrium response.
Whole-star cooling, envelopes and rotating mode spectra retain separate gates.

The independent JAX PINO development follows Ray, arXiv:2611.05790v1.
RNS is an independent reference path. Neither a trained approximation nor a
JAX gradient is admitted to inference without matched forward and constrained
derivative checks; model artifacts must carry their training/domain manifest.

See [the independent implementation and qualification procedure](rotating-pino-development.md)
and `scripts/rotation/README.md` for the bounded development workflow.
