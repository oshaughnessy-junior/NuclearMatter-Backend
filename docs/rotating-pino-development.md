# Independent rotating PINO development

This is our independent JAX implementation of the physics-informed procedure
in [Ray, arXiv:2611.05790v1](https://arxiv.org/html/2611.05790v1). No upstream
code or weights are needed. **The current trained pilot is not qualified for
EOS inference.** Conventional RNS remains the independent reference path.

## Implemented procedure

- A functional branch/trunk neural operator predicts three scaled metric
  potentials from sampled EOS functions, central enthalpy and coordinate axis
  ratio. Our architecture adds a smooth centre and regular radial derivative
  to the paper's symmetry and static frame-dragging constraints, and enforces
  the gamma/rho/omega far-field powers r^-2/r^-1/r^-3. It is
  an independently chosen variant, not a bitwise reproduction of its network.
- Hydrostationary centre/pole/equator relations determine the coordinate
  radius, angular speed and connected central matter region. Matter outside
  that region is vacuum; exterior gravitational source terms remain included.
- JAX evaluates field derivatives, the first-order alpha reconstruction,
  nonlinear metric sources and the Gegenbauer Green integrals over the
  compactified exterior. Residuals compare the predicted and mapped fields.
- Training uses **only physics residuals**, with fresh parameter points held
  out for validation. Adam with cosine restarts is followed optionally by a
  bounded, full-batch L-BFGS refinement. Invalid/nonfinite trials are failures,
  not omitted stars. Failed runs are retained.
- Checkpoints include configuration, prior, train/validation cases, library
  source and training-driver hashes. The evaluator rejects changed library
  code. It reports mass, radius, angular speed and metric-field errors against
  separate RNS models, and flags cases outside the training prior.

No new dependency is required beyond the backend's JAX/NumPy/SciPy stack.
The initial EOS adapter covers relativistic polytropes with K=G=c=1; the
central energy-to-enthalpy root has an implicit JVP. General nuclear-EOS
sampling, phase-transition handling and physical unit exports remain work.

## Reproduce a bounded pilot

Run science on an authorized compute host selected through infra-atlas, not
on an access point. From a configured backend checkout:

```
export PYTHONPATH=$PWD/src JAX_ENABLE_X64=true JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m pytest tests/test_rotating_pino.py -ra
python scripts/rotation/train_pino.py pilot \
  --steps 12000 --width 32 --modes 16 --nr 48 --nmu 12 --harmonics 4 \
  --batch 16 --train-cases 48 --coverage-grid --learning-rate .0003 \
  --wall-seconds 600 --lbfgs-steps 1500
python scripts/rotation/evaluate_pino.py pilot rns-results pilot/reference-comparison.json
```

Output paths must be fresh. `rns-results` holds the returned compressed
RNS jobs described in the runbook. `best.npz` is selected on physics validation
loss, never on the RNS comparison errors. Do not tune a checkpoint repeatedly against
the same RNS cases and continue calling them an untouched final test set.

Analytic tests check relativistic enthalpy identities and its implicit
derivative, exact uniform-ball Green inversions in three operator dimensions,
symmetries, centre regularity and a nonzero network derivative. The conventional
RNS checks gate the actual field equations before training.

## RNS derivative reconstruction lesson

The initial reference check differentiated a cubic interpolant through every
exported RNS radial node. The resulting alpha discrepancy was about 0.02 and
failed the unchanged 0.01 development gate. RNS's Simpson radial integration
leaves an alternating-node artifact: field values agree well while second
derivatives of that interpolant do not represent the smooth equilibrium.
Using consistent radial nodes (stride two) reduces the alpha error to about
2e-5 on the 257-by-129 reference, and the same gate passes for spherical and
q=0.8 backgrounds. Stride one remains the diagnostic default to preserve
the failure; reproduction of the passing check must explicitly request:

```
python scripts/rotation/check_keh.py reference/run/rns_rotating.csv keh.json --radial-stride 2
```

At 128-by-16 collocation, the q=0.8 development check has metric Green-map
RMS errors below 3.5e-4 and mass discrepancy about 1e-4. This qualifies a
bounded reference check, not all EOS/spins, continuum accuracy or derivatives.
Both failing and passing receipts must accompany further qualification.

## Remaining acceptance gates

Refine collocation and angular truncation, test new EOS/spin/central-density
points, reproduce static and slow-spin limits, and assess the surface and
near-Kepler regime. Compare integral quantities and metric fields separately.
Only after matched forward gates should EOS and constrained-sequence
derivatives be compared with fresh RNS finite differences. Fixed central
energy/axis ratio, observed mass/spin and fixed baryon-mass sequences are
different derivatives. Training loss or automatic differentiation alone does
not close any of those gates. Whole-star cooling and modes remain separate.
