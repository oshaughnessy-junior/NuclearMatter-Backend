# First-order implicit AD for the finite-temperature GM1 state

15 October 2026. This is a narrow differentiable interface to the already accepted uniform-matter GM1 numerical reference. It evaluates the **same nonlinear equations and quadrature**, then supplies an analytic implicit first derivative. It does not fit an emulator, modify the frozen reference, promote a production EOS, or resolve the missing matched external finite-T benchmark.

## Reproduction and measured checkpoint

Default Python on this host lacks JAX. An existing environment was found and used without installing anything:

```bash
/Users/rossma/miniconda3/envs/junior_tools/bin/python \
  outputs/nmb-finite-t-20261015/rmf_ad/validate_ad.py
```

The measured environment is Python 3.12.12, JAX 0.10.1, NumPy 2.3.5 and SciPy 1.15.2 on the CPU. The test report records the actual executable, versions and frozen dependency hashes. The earlier non-JAX references used the default Python environment; the primal gates are therefore rerun here before derivative comparisons.

Measured result: **304/304 checks pass**. The first 295-check run took 46.8 seconds; the expanded suite takes roughly 48 seconds. See `validation_results.json` for actual values, all Jacobians, directional step ladders, residuals and derivative-order behavior. No new dependencies, production changes or frozen-reference changes were made.

Three locally stable warm states are tested:

```
(nB [fm^-3], T [MeV], Yp)
(0.16,10,0.1)
(0.24,20,0.3)
(0.32,30,0.1).
```

These states and their perturbations qualify this checkpoint; they are not a certificate over the entire rectangular range between them.

## Public research interface

`state_ad(x)` takes a real float64 vector `x=(nB,T,Yp)` with `T=k_B T_physical` in MeV. It returns twelve quantities, with ordering exported as `OUTPUTS`:

| Index | Quantity | Units |
|---|---|---|
|0|Scalar field Phi|MeV|
|1|Vector field W|MeV|
|2|Isovector field R|MeV|
|3|Dirac mass mD|MeV|
|4–5|Kinetic chemical potentials nu_n,nu_p|MeV|
|6–7|Total chemical potentials mu_n,mu_p|MeV|
|8|Total nuclear energy density e|MeV fm^-3|
|9|Pressure P|MeV fm^-3|
|10|Helmholtz free-energy density f|MeV fm^-3|
|11|Entropy density s|fm^-3, k_B=1|

Masses and chemical potentials retain the frozen GM1 common bare mass convention `M=939 MeV`, including rest energies. Derivative columns are ordered `(nB,T,Yp)`, and each matrix entry carries output-units divided by its input-units.

`oracle_value_jacobian(x,return_diagnostics=True)` is a strict ordinary-Python preflight that returns the converged values, analytic 12x3 Jacobian and diagnostics. It rejects invalid inputs and ill-conditioned/singular scalar-number response. `state_ad` wraps those computations in JAX callbacks with a supplied `custom_jvp` rule. Its first derivative is the matrix-vector product `J@dx`; reverse mode transposes that linear map.

Runtime exceptions are not a valid error-reporting contract for `jax.pure_callback`. Numerical/domain failures inside the staged callbacks therefore return fixed-shape NaN sentinels. The strict host oracle provides the explanatory error before staging. Tests verify invalid-temperature sentinels in a JIT primal and a JVP, and strict host rejection. This is a research interface with a stated validity/failure convention, not a robust exported status-bearing production API.

## What is differentiated

The primal is the frozen `gm1_finite_t.state`: adaptive quadrature, scalar/number root, field energies and occupation entropy. The new layer does **not** differentiate through root iterations, bracket selection or adaptive quadrature decisions. It differentiates the converged equations on the same local smooth branch.

The frozen nonlinear solver may use a numerical solver Jacobian during its search. That is a primal implementation detail. The **returned AD derivative rule contains no finite differences**: its field/number Jacobian and temperature derivatives are analytic distribution integrals, combined through a 3x3 linear solve.

Let `z=(Phi,nu_n,nu_p)`, `m=M-Phi`, and write the three constraints in internal fm units as

```
F0=U'(Phi)-rho_s_n-rho_s_p
Fn=n_n(T,nu_n,m)-(1-Yp)nB
Fp=n_p(T,nu_p,m)-Yp nB.
```

Reuse the independently qualified gas responses from the stability checkpoint:

```
chi_i=dn_i/dnu_i|m,T
b_i=drho_s_i/dnu_i|m,T=-dn_i/dm|nu_i,T
q_i=drho_s_i/dm|nu_i,T.
```

Then the unknown-state Jacobian is

```
F_z = [[U''+q_n+q_p, -b_n, -b_p],
       [b_n,           chi_n, 0],
       [b_p,           0, chi_p]].
```

For the input density derivatives,

```
dn_n_target=(1-Yp)dnB-nB dYp
dn_p_target=Yp dnB+nB dYp.
```

The explicit temperature response enters `F_x`, and the derivative is obtained from

```
F_z dz = -F_x dx.
```

The canonical scalar curvature is `D=U''+sum(q_i+b_i²/chi_i)`. A nonpositive/near-singular D or excessively conditioned field/number Jacobian is rejected by the strict host oracle and produces NaN in a staged call. This scalar gate does not prove matter-density stability: the full chemical Hessian remains a separate necessary diagnostic.

## Analytic temperature moments

For each species define `E=sqrt(p²+m²)`, particle occupation `f`, antiparticle occupation `fbar`, and `w=f(1-f)`, `wbar=fbar(1-fbar)`. In fm units, holding `nu,m` fixed:

```
n_T = integral p²[(E-nu)w-(E+nu)wbar] dp /(pi² T²)
rho_T = integral p²(m/E)[(E-nu)w+(E+nu)wbar] dp /(pi² T²)
s_T = integral p²[(E-nu)²w+(E+nu)²wbar] dp /(pi² T³).
```

The new deterministic quadrature computes these moments directly. Particle/antiparticle factors are handled with stable logistic functions. The entropy Maxwell identities are `s_nu=n_T` and `s_m=-rho_T`, giving

```
ds=sum_i[s_T_i dT+n_T_i dnu_i]+sum_i rho_T_i dPhi.
```

After conversion back to MeV, thermodynamic differentials generate the remaining output derivatives:

```
df=-s dT+mu_n dn_n+mu_p dn_p
de=T ds+mu_n dn_n+mu_p dn_p
dP=s dT+n_n dmu_n+n_p dmu_p.
```

The field contributions cancel through stationarity; vector shifts retain their density/composition derivatives. No chemical energy term is added on top of the total-energy convention.

A separate eliminated-constraint calculation supplies useful temperature checks. With

```
K=sum_i(rho_T_i-b_i n_T_i/chi_i),
dPhi/dT=K/D
Cv|nB,Yp = T[sum_i(s_T_i-n_T_i²/chi_i)+K²/D].
```

These formulas agree with the general implicit linear solve. All T in these two internal formulas are in fm^-1; the returned heat capacity in fm^-3 equals `de[MeV fm^-3]/dT[MeV]`.

## Measured first derivatives

| nB [fm^-3] | T [MeV] | Yp | dmD/dT | Cv at fixed(nB,Yp) [fm^-3] | dmu_p/dYp [MeV] |
|---|---:|---:|---:|---:|---:|
|0.16|10|0.1|0.232738887|0.116645716|211.357495|
|0.24|20|0.3|0.467218167|0.253057371|225.810005|
|0.32|30|0.1|0.497331453|0.341200035|490.783977|

The field/number implicit matrices have condition numbers about 22.46, 17.55 and 12.12 at these points. Maximum nonlinear residuals are below 5e-15, and linearized-equation residuals are below 5e-16 in the recorded conventions. Those measured margins apply to the tested states; a nominal cutoff is not an all-domain conditioning theorem.

The validation sequence is:

1. Match all twelve primal outputs directly to the frozen nonlinear reference and verify local positive chemical-Hessian eigenvalues at the three selected states.
2. Verify each new analytic temperature moment against an ordinary finite difference of the original gas integrals at fixed kinetic chemical potential and Dirac mass. Eighteen such checks have a largest measured relative discrepancy about 1.20e-7.
3. Compare JAX `jacfwd` and `jacrev` to the supplied analytic Jacobian, and refine the quadrature/root evaluation. The largest measured Jacobian refinement discrepancy in the first run was below 2e-14 on row-wise scales.
4. Independently compare density/composition chemical-potential derivatives to the accepted two-species Hessian; verify temperature/scalar/Cv identities and the known free-energy coordinate derivatives.
5. Test four directions: density-only, temperature-only, composition-only and mixed. At each of three decreasing dimensionless steps `(0.02,0.01,0.005)`, independently rerun both perturbed nonlinear states, gate their residuals and retain the raw finite differences. Compare the Richardson derivative with the analytic JVP for every output. The 144 output/direction checks had a largest scaled error about 1.82e-10 in the first full run; absolute floors prevent zero derivatives from generating meaningless relative ratios.
6. Verify a JIT primal, JIT first JVP and reverse gradient of a weighted scalar functional. These checks qualify transformation behavior; they do not constitute an independent physical model comparison.

## Derivative order and physical boundaries

**Qualified operations:** first-order JVP, VJP/`grad`, `jacfwd`, `jacrev`, and the tested JIT combinations, for a scalar three-input call on these smooth warm branches. Smooth downstream JAX functions can use those first derivatives.

**Unsupported:** nested AD through the state Jacobian, arbitrary-order gradients, coupling/parameter derivatives, derivatives through phase selection, cold/asymptotic matching, multidimensional table interpolation or broad vectorized/GPU/distributed operation. The test explicitly attempts `jacfwd(jacrev(state_ad))` and confirms a JAX `ValueError` for the unruled callback derivative. It does not silently return zero or a purported Hessian.

A first derivative of an output chemical potential corresponds physically to a second derivative of free energy and is supplied/validated here. That does **not** mean that taking `jax.grad` twice through the free-energy callback is supported. Similarly, this checkpoint does not qualify differentiating a stiff time integrator whose implementation internally differentiates the EOS and then requires derivatives of that differentiated operation.

A spinodal can possess well-defined local analytic derivatives. AD does not remove the negative chemical-Hessian eigenvalue or supply a global phase construction. The callback scalar-conditioning gate is therefore deliberately distinguished from the complete physical stability/neutrality check. Stable warm points were selected explicitly for this qualification; it must not be inferred that every accepted scalar root is a stable fluid.

## Next implementation choices

This reference is sufficient to test small first-order state sensitivities and begin a differentiated local response experiment under the controlled-ansatz program. Before production promotion, obtain a matched external finite-T GM1 forward reference, retain independent stability/phase qualification, and review the callback/units/implicit algebra.

If higher derivatives become necessary, implement differentiable distribution moments and an explicit higher-order implicit rule, or a fully JAX quadrature/root path with the same converged forward oracle. Do not approximate the missing derivative by silently differentiating an opaque callback or training an EOS emulator. A native deterministic implementation can be evaluated against the archived primal and Jacobian data before replacing this slower CPU reference.

The [official JAX external-callback documentation](https://docs.jax.dev/en/latest/external-callbacks.html) describes host callbacks, their purity/error semantics and combination with custom derivative rules; the [`custom_jvp` reference](https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html) documents the transformation interface. The actual executed version is pinned in the result metadata, rather than inferred from the latest web documentation.
