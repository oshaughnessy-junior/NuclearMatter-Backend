# Separate compatibility and thermal functional-domain review

**Static minimum-API repair ACCEPTED. Returned-state thermal-increment completion remains BLOCKED at two resolved selected states.** This review has140 passing checks and2 functional completion failures. It does not relabel matched old/new rejections as successful thermal evaluations. The prior207-PASS/4-forward-FAIL initial review and665-PASS/1-static-FAIL repair review remain unchanged.

Reviewed source`53518a39d9b4030745285d774e028e8814485ca25d48f8fa45587d5a14e9af9e`, candidate manifest`bdb62cfe3e4192d583d648315ffeaa8219eb1a2d0fea4621396b6c3a4714bb0d`, author report`6f60d2c654ec101b1342da2df681f7f3f8507d0f825ba3fd97fbb692a25be4e6`, checkpoint`7568eee921a13457c6055b6c5f903eab82af4430c421928623d13f41b52d590b`. No author/upstream files were edited.

## Compatibility finding resolved

The only numerical-source change replaces the unavailable SciPy import by private `_log_expit(x)=-np.logaddexp(0,-x)`. The [NumPy1.21 primary documentation](https://numpy.org/doc/1.21/reference/generated/numpy.logaddexp.html) includes this function and records its introduction in1.3.0. Together with the earlier SciPy1.7 quad_vec API inspection, the specific missing-SciPy1.8-import incompatibility is closed statically.

Fresh independent scalar helper checks use stable math.log1p formulas at signed zero, tiny arguments, ordinary positive/negative values, dilute tails and +/-1e300. They pass2e-15 relative gates. A fresh legitimate package-relative candidate import succeeds when only scipy.special.log_expit is temporarily absent from the existing SciPy module; the export is restored in finally. This is a missing-export simulation, not installed SciPy1.7 execution or a fake package.

Every pre-existing function is AST-identical to the preserved preceding candidate after normalizing only the helper name; the new helper is the sole added function. Public signatures/defaults, rest masses/constants, pair-flavor independence, output fields, units, density root, thermal formula and quadrature policies do not change. Selected new warm/dilute/degenerate states and tiny signed roots pass old/new comparisons; successfully evaluated thermal increments also agree. New states include electron T=.87,mu=-1.31; muon T17.8,mu93.7 and T=.01,mu151.3; and symmetric electron m/T40,600. Prior independently accepted665 numerical/provenance checks support the unchanged integral implementation; they were not all rerun or added to this140-check count.

## Two physical completion gates remain failed

At the two returned electron states

`T=.0008516649166666665 MeV, m=.51099895 MeV, mu=+/- .0004513824058333333 MeV`,

both the preserved preceding candidate and the compatibility-repaired candidate return valid state_mu thermodynamics but raise

`ArithmeticError: Unresolved finite-temperature lepton quadrature`

from thermal_energy_density. This is a **functional-domain failure**, not a wrong returned scalar and not a new regression caused by the compatibility helper. The conservative exception avoids returning an unsupported number; it also means this requested thermal utility cannot complete on these already admitted resolved states. Its generic unresolved-quadrature exception does not supply positive forward evidence at these points.

The independent Bessel dilute formula gives each particle population with FD corrections negligible at m/T600. Let N=m²T K2(m/T)/[pi² HC³], alpha=mu/T, n_net=2N sinh(alpha), and mean energy=3T+m K1/K2. Then the rest-inclusive total energy is2N cosh(alpha)*mean. The cold energy is m*abs(n_net) plus an unrepresented tiny cold kinetic correction. Thus the thermal increment is

`2.4114807641831635e-274 MeV fm^-3`.

It is NORMAL float64, far above the minimum normal~2.225e-308. Independently reconstructed energy and net number agree with state_mu within the same3e-10 gates. There is no near-total cancellation or total-heat underflow here. The initial two matched rejections in the author follow-up therefore cannot be counted as physical thermal PASS.

## Independent bound on the troublesome positive hole contribution

In the dimensionless coordinate x=(E-m)/T, let v=m/T and x_c be the cold kinetic edge. Here x_c=2.487918144260715e-174. The unscaled dimensionless hole integrand is

`sqrt[x(x+2v)]*(v+x)*(x_c-x)*(1-f)` on0<=x<=x_c.

Using0<=1-f<=1 and the monotonic geometric factors,

`I_hole <= (4/15)*sqrt(x_c+2v)*(v+x_c)*x_c^(5/2)`.

My log upper bound is-990.7256875339907 before dimensional restoration. Multiplying by T^4/[pi² HC³] gives log physical upper bound-1037.1430033561905. Both are well below log minimum float64 subnormal=-744.4400719213812. This explains why the zero-epsabs component quadrature can report value/error0 with unresolved status. It proves that THIS positive component is mathematically unrepresentable as an ordinary float, while the sum of other positive thermal contributions remains represented. The separation is essential: there is no justification for a density floor, replacing total thermal energy by zero, or silently accepting all zero quadratures.

For a reliable returned-state thermal utility, hold functional admission until this case is handled with a controlled component policy and independently rechecked. A narrow repair can be assessed against the explicit bound and representable total; no such repair was made by this reviewer. State_mu acceptance and the successful thermal states remain separate from this completion gap.

## Preservation, environment and remaining gates

Full current author-tree before/after hashes are identical. All pinned previous initial and quadrature review files verify, including historical failures. Exactly five staged production paths remain proposed; this review does not authorize adding campaign artifacts or changing dependencies. Key candidate files/report/checkpoint/manifest are copied in `reviewed_compatibility_candidate/`. The previous compatibility/static failure and the author oracle/reporting failures remain preserved.

This review runs in existing Python3.12.12/NumPy2.3.5/SciPy1.15.2/JAX0.10.1, without installs. The NumPy version is outside the separate validation environment's<2.3 pin. Minimum-version/supported-version execution, admitted repository CI/full suite, packaging and broad-domain qualification remain OPEN. Static API availability and missing-export simulation cannot be presented as minimum-runtime testing. No JAX AD/JIT/Hessian replacement, full EOS, cooling, empirical uncertainty or uniform error certificate follows.

Exact bounded reproduction from Genesis:

```
JAX_ENABLE_X64=true OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/envs/junior_tools/bin/python -I -B outputs/nmb-finite-t-20261015/review_production_lepton_compatibility/probe.py
```

`results.json` separately stores passing checks, both failed functional gates, old/new exceptions, independent expected values and component bounds. No production/paper/git/PR change, author source repair or relaxed gate occurred.
