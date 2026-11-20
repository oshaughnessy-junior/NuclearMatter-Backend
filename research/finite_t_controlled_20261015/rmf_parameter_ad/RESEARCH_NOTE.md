# First-order finite-T RMF coupling sensitivities, with parameter work

15 October 2026. This leaf extends the accepted local finite-T GM1 numerical reference to **five selected nuclear couplings**. Bare nucleon mass remains 939 MeV. The state-only AD prototype, susceptibility/stability module and underlying nonlinear reference remain unchanged. The model is the same controlled uniform Hartree ansatz; new sensitivities do not establish a calibration, physical uncertainty family or externally validated finite-T EOS.

The important thermodynamic point is that **varying the Hamiltonian parameters changes Gibbs-Duhem**. Reusing the fixed-parameter pressure differential gives incorrect coupling gradients. The vector-coupling examples below make the error an exact factor of two.

## Reproduce and inspect

```bash
/Users/rossma/miniconda3/envs/junior_tools/bin/python \
  outputs/nmb-finite-t-20261015/rmf_parameter_ad/validate_parameters.py
```

The existing Python 3.12/JAX 0.10.1 CPU environment is reused; no installation is required. Measured result: **504/504 checks pass**, including 180 coupling/output finite-difference checks, 45 mixed Hellmann-Feynman/Maxwell checks, source-defined explicit partials and primal/branch gates at perturbed couplings. Full matrices, raw ladders and actual environment versions are recorded in `validation_results.json`.

No production files, frozen research leaves, dependencies or git state were changed. The checkpoint manifest hashes the imported frozen implementations and all files in this leaf.

## Inputs and the qualified scope

`state_parameter_ad(x)` takes the raw float64 input vector

```
x=(nB,T_MeV,Yp,C_sigma,C_omega,C_rho,b,c).
```

The first three inputs match the state-only prototype; `C_sigma,C_omega,C_rho` have units fm², while `b,c` are dimensionless. The bare mass is fixed to 939 MeV. `pack_state` accepts only those five named coupling overrides, rejecting extra model choices instead of silently varying an unsupported parameter.

The twelve outputs retain the frozen state order: scalar/vector/isovector fields, Dirac mass, kinetic and total neutron/proton chemical potentials, energy density, pressure, free energy and entropy. `INPUTS`, `PARAMETERS` and `OUTPUTS` export the order. The Jacobian is 12x8. For the C columns, a field/chemical-potential derivative has units MeV fm^-2 and an energy-density/pressure/free-energy derivative has units MeV fm^-5. For b/c columns, derivatives have the same units as their outputs.

Only first-order derivatives are qualified, at these three warm states and the small recorded coupling perturbations:

```
(nB,T_MeV,Yp)=(0.16,10,0.1),(0.24,20,0.3),(0.32,30,0.1).
```

The baseline couplings are the exact rounded NMB GM1 values at snapshot `25e68182f726e5bd9c0b372c9a667536940fbd07`: `C_sigma=11.785`, `C_omega=7.148`, `C_rho=4.410`, `b=0.002947`, `c=-0.001070`, with zero omega-rho cross coupling. The [immutable NMB source](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/eos/rmft.py) and its archived bytes define these conventions. No beta-equilibrium composition solve is differentiated. These are conditional sensitivities at independently fixed baryon density, temperature and proton fraction.

## Derive the parameter terms before differentiating

Use the frozen reference's internal fm units, with `M=939/HBARC`, `mD=M-Phi`, and

```
U(Phi)=Phi²/(2Cs)+b M Phi³/3+c Phi⁴/4,
W=Co nB,
R=(Cr/2)n3,
n3=np-nn.
```

After algebraically eliminating the vector fields, the canonical free-energy density is

```
f=sum_i f_kin(T,n_i,mD)+U(Phi)+Co nB²/2+Cr n3²/8 .
```

The scalar field obeys `U'(Phi)=rho_s_n+rho_s_p`. Its implicit variation therefore cancels in the first derivative of this canonical potential at fixed T and species densities. The explicit **reduced-canonical** derivatives are

```
f_Cs= -Phi²/(2Cs²)
f_Co= +nB²/2
f_Cr= +n3²/8
f_b = +M Phi³/3
f_c = +Phi⁴/4 .
```

Multiply these internal derivatives by `HBARC` for returned free-energy-density units. These formulas refer to the canonical potential after eliminating vectors; they are not obtained by incorrectly freezing an independent W while changing Co. The scalar field can be held fixed for the explicit partial because stationarity removes its implicit contribution.

The parameter-extended first law and pressure differential are

```
df=-s dT+mu_n dn_n+mu_p dn_p+sum_theta f_theta dtheta
de= T ds+mu_n dn_n+mu_p dn_p+sum_theta f_theta dtheta
dP= s dT+n_n dmu_n+n_p dmu_p-sum_theta f_theta dtheta.
```

The last line follows by differentiating `P=sum_i mu_i n_i-f`, and is essential. It is not optional parameter bookkeeping. These relations also state how a controlled model should account for work if it deliberately changes an interaction parameter; ordinary physical evolution usually holds those parameters fixed.

## Implicit state differentiation

The converged root is still `z=(Phi,nu_n,nu_p)`, with the scalar gap and neutron/proton number constraints. The accepted state-only analytic matrix is reused in form, now evaluated at the supplied couplings:

```
F_z=[[U''+q_n+q_p,-b_n,-b_p],
     [b_n,chi_n,0],
     [b_p,0,chi_p]],
```

where here `b_i=drho_s_i/dnu_i` is a response symbol, distinct from the scalar coupling b. The quantities `chi_i,q_i` and the new temperature moments come from the frozen, already checked distribution-response integrals.

At fixed `(Phi,nu_n,nu_p,T)`, the explicit scalar-equation parameter row is

```
F0_theta=(-Phi/Cs²,0,0,M Phi²,Phi³),
```

and the explicit number-equation coupling rows vanish. Solve `F_z dz=-F_x dx`, including the state-density and temperature columns. Vector variations retain their direct parameter terms:

```
dW=Co dnB+nB dCo
dR=(Cr/2)dn3+(n3/2)dCr.
```

The chemical potentials include those direct shifts. Entropy has no explicit coupling contribution at fixed kinetic gas state; its parameter dependence follows through the differentiated scalar field and kinetic chemical potentials. Energy, pressure and free-energy derivatives then use the parameter-work formulas above.

The strict host oracle checks scalar/number conditioning and returns a local chemical-Hessian diagnostic evaluated with the **perturbed couplings**, rather than accidentally reusing GM1 vector coefficients. Tests require positive eigenvalues on all perturbed states before interpreting coupling finite differences. This is a local uniform-branch gate; it does not establish global coexistence or validity of large parameter excursions.

## Independent source and finite-difference gates

The numerical validation follows the forward-before-derivative rule:

1. At each baseline state, match all primal values and the first three Jacobian columns to the frozen state-only oracle. JAX forward and reverse Jacobians must reproduce the supplied 8-input analytic Jacobian.
2. Read a restricted AST selection of the actual archived NMB `rmft.py` formulas. Hold the scalar field and densities fixed and vary one coupling in the source-defined energy. This independently checks all five explicit Hellmann-Feynman partials, including signs, rho normalization and HBARC factors. The source's kinetic term is cold, but is unchanged by a coupling variation at fixed Phi and density; the explicit coupling partial is therefore the same at finite T.
3. For each coupling, independently resolve the full finite-T state at relative steps `(0.002,0.001,0.0005)` in each direction. Gate nonlinear residuals and the full two-species local stability before comparing derivatives. Store all raw steps and a Richardson extrapolant.
4. Compare all twelve analytic output derivatives with those re-solved finite differences. The 180 component checks have a largest measured scaled error below 9e-10 on the recorded states, with stated absolute floors for vanishing derivatives. All perturbed root residuals are below 7e-15.
5. Check the parameter first law, explicit free-energy partials, pressure-work correction, exact Co/Cr pressure sensitivities and vanishing vector-coupling entropy response at frozen composition.
6. Verify a mixed state/coupling JVP, a JIT reverse gradient of a scalar functional, an invalid-coupling NaN sentinel, and explicit rejection of nested/higher-order AD.

A useful validation lesson appeared in the source-partial comparison: the first single central-difference check used a 0.001 relative step for Cs and showed the exactly expected 1e-6 relative truncation of differentiating 1/Cs. Applying the same decreasing-step/Richardson policy removed that test artifact. No physical derivative formula or tolerance was loosened to hide it.

The solver's internal numerical search Jacobian remains a primal detail. The new AD rule itself contains **no finite-difference derivative**. As in the state-only leaf, deterministic callbacks evaluate the converged numerical equations and analytic moments; they do not substitute a trained representation.

## Mixed parameter/state Maxwell checks

The explicit Hellmann-Feynman partials are also differentiated with respect to state using the **frozen state-only scalar tangent**. This supplies 45 checks of

```
df_theta/dT= -ds/dtheta
df_theta/dnB= (1-Yp)dmu_n/dtheta+Yp dmu_p/dtheta
df_theta/dYp= nB(dmu_p/dtheta-dmu_n/dtheta).
```

For example, `d f_Cs/dx=-HBARC Phi/Cs² * dPhi/dx`, while `d f_Co/dnB=HBARC nB`. These cross relations independently connect parameter terms in the constraint Jacobian with thermodynamic work and the already qualified state-coordinate tangent. They are mathematical second derivatives of the potential supplied through separate first-derivative identities; nested JAX differentiation of the opaque callback remains unsupported.

## Measured parameter derivatives and the pressure correction

At `nB=0.16 fm^-3,T=10 MeV,Yp=0.1`:

| Coupling | df/dtheta | dP/dtheta | Incorrect value from fixed-parameter Gibbs-Duhem |
|---|---:|---:|---:|
|Cs [fm²]|-1.50109574|-0.906381629|-2.40747737|
|Co [fm²]|+2.52578535|+2.52578535|+5.05157070|
|Cr [fm²]|+0.404125656|+0.404125656|+0.808251312|
|b|+961.415180|+1351.48055|+2312.89573|
|c|+220.266607|+486.266438|+706.533045|

The C derivatives in this table are in MeV fm^-5; the b/c derivatives are in MeV fm^-3. The incorrect column is `nn*dmu_n/dtheta+np*dmu_p/dtheta`, which omits `-f_theta`. It is included to make the convention error concrete rather than just stating an identity.

Co/Cr provide especially simple exact checks. At fixed state, their kinetic fields, masses and entropy do not change; only vector energies and chemical-potential shifts vary. Thus

```
dP/dCo=df/dCo=HBARC nB²/2
dP/dCr=df/dCr=HBARC n3²/8.
```

Summing `n_i*dmu_i/dtheta` alone gives twice those values. The agreement of the corrected derivative with the separately re-solved primal pressure is a direct test of the parameter-work sign.

## Boundaries and next use

Qualified inputs are the three state coordinates and five raw couplings, fixed common M=939 MeV, within the tested small perturbations on smooth locally stable branches. This does not differentiate a mapping from empirical saturation observables to couplings, reimpose beta equilibrium, construct a stellar profile, or quantify a posterior uncertainty. Large coupling changes can destroy calibration, change branches or render the uniform state unstable. The original negative quartic scalar coefficient also retains the model's global-potential limitation.

Only first-order JVP/VJP, `jacfwd`/`jacrev` and the tested JIT combinations are qualified. Higher derivatives and gradients through this returned Jacobian are explicitly unsupported and tested to raise at the transformation boundary. Staged numerical/domain failures return NaN; use the strict host oracle first for diagnostics. No claim is made for a full differentiated time integrator, phase selection, arbitrary batching, GPU execution or broad parameter domain.

The first useful application is a controlled local sensitivity study: vary selected nuclear couplings at fixed composition/temperature, compare induced thermodynamic and quasiparticle changes, then propagate them into a separately matched weak-response calculation. A matched external finite-T forward comparison and review remain prerequisites for production promotion. The response does not supply missing spin correlations, pairing or weak matrix elements merely because the bulk EOS can now be differentiated.
