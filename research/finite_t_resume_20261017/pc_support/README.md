# Soft positron-capture conditional-shell reference

Frozen author slice for a DIFFERENT review. It uses exact source copies of PR55's reviewed full free Guo current and fixed T2 chemical bath. It does not modify PR55, production code, previous outputs, generic drivers, or public claims.

The exact prior Eν=0.1 MeV call is reproduced in `ORIGINAL_FAILURE.json`. It fails the original strict final-shell guard at a genuine support point with initial proton energy about 3.64 million MeV. The problem is subtraction of nearly equal squared ultrarelativistic four-vector components, not absent physical support. That original exception and source are preserved.

`conditional_pc.py` changes integration coordinates to incident-antineutrino absorption on an incoming moving proton, followed by exact two-body final-state phase space. It computes both PC emission and true inverse absorption using their actual FD products and the same named current. It does not generate an inverse from KMS, set underflowed rates to zero, clip a physical discriminant, or introduce a surrogate. `REPORT.md` derives the support and absolute measure.

Selected results:

| Eν (MeV) | log j (s⁻¹) | log true χ (s⁻¹) |
| --- | ---: | ---: |
| 0.1 | −3786.58485976718 | −3787.939541852008 |
| 0.15 | −2381.239481736936 | −2382.569163821738 |

Both are positive finite logarithms whose ordinary float64 values underflow. Their ordinary representation is `None`, never physical zero.

Reproduce with one numerical thread:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 reproduce_original.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_convergence.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_semi_infinite.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 phase_jacobian.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 assess.py
```

Run these from this directory, or use their absolute script paths. Individual evaluations impose an 85-second wall budget and finish in seconds at the selected rules. No dependency installs are needed.

`UNMAPPED_INCOMING_ENDPOINT_RAW.json` and the prior candidate preserve an initially insufficient incoming-angle rule: direct u quadrature converged algebraically because the CM final momentum vanishes as sqrt(1−u). The exact u=1−v² map removes that coordinate corner; this is not threshold smoothing. Original diagnostic extraction also initially referenced a module global as a frame local; that failed instrumentation attempt did not return scientific values, and the saved reproducer fixes the extraction without changing the original numerical source.

An independent local coordinate-Jacobian probe connects the new phase measure to PR55's original El/cos/azimuth delta-shell measure. Literal external-spinor sums check the invariant current. These are distinct anchors, but neither is a separately authored integrated phase oracle. That stronger review remains pending. Only two soft incident energies are admitted by this author evidence. No whole-energy moment, transport, EOS binding, gradient or empirical current-model accuracy claim is made.
