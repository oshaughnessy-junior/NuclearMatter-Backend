# Uniform finite-temperature GM1 reference at independent composition

15 October 2026. This is the campaign's next bounded reference slice: **one explicitly controlled uniform nucleonic Hartree mean-field ansatz**, evaluated accurately on a small set of states. Its purpose is to connect thermodynamic state, effective masses and eventually weak kinematics in transparent calculations. It is not a phase-complete stellar EOS, a finite-temperature NMB production release, or an endorsement of GM1 as a modern uncertainty model.

## Actual source provenance

The NMB source was retrieved through the connected GitHub file API at immutable commit `25e68182f726e5bd9c0b372c9a667536940fbd07`:

- [src/nmbackend/eos/rmft.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/eos/rmft.py), Git blob `1edba14f7422f37f1ae6e4eb58167dc2c2140061`.
- [src/nmbackend/constants.py](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/25e68182f726e5bd9c0b372c9a667536940fbd07/src/nmbackend/constants.py), Git blob `cbd0b9009e1f333c6c447930670bcbfedaefd68e`.

Both full files are preserved in `source_snapshot/`; the checkpoint manifest records content hashes. `gm1_finite_t.py` follows the source's GM1 convention rather than converting a separately tabulated meson coupling. Parameters are `M=939 MeV`, `C_sigma=11.785 fm²`, `C_omega=7.148 fm²`, `C_rho=4.410 fm²`, `b=0.002947`, `c=-0.001070`, with no omega-rho cross coupling. The validation script reads the archived parameter assignment and constants directly and compares every parameter.

The cold comparison executes a restricted AST selection of the **actual pinned source's pure formulas** using NumPy as their array API. It evaluates kinetic/scalar densities, field energies, pressure, chemical potential and symmetry energy without importing NMB or claiming that its full test suite was run. The independent scalar root is supplied by this reference. This is a source-convention regression, not an independent physical theory comparison.

## Reproducible checkpoint

```bash
python outputs/nmb-finite-t-20261015/rmf_reference/validate_gm1.py
```

Measured result: **197/197 checks pass**, including 50 comparisons against a separately implemented momentum-variable fixed Gauss-Legendre quadrature. The solver uses adaptive kinetic-energy-variable quadrature. Actual numerical outputs, tolerances, sample states and runtime are in `validation_results.json`; the complete suite takes roughly 16 seconds on this host. No dependencies were installed; no production files or git state were changed.

The checkpoint consists of the reference implementation, validation driver, JSON results, this note, provenance manifest and archived source. It can be rerun with the already available Python, NumPy and SciPy environment. It does not require JAX, GPUs, NMB installation or downloaded EOS tables.

## State contract and equations

Inputs are baryon **net** density `nB` in fm^-3, temperature `T=k_B T_physical` in MeV, and independent proton fraction `0<Yp<1`. Thus `nn=(1-Yp)nB`, `np=Yp nB`. No beta equilibrium or charge-neutrality solve is imposed. Leptons, photons, nuclei, pairing and neutrinos are absent.

The internal convention is the source's `hbar*c=1`, with masses and momenta in fm^-1. Divide MeV by `HBARC=197.3269804 MeV fm`; energy/pressure densities in fm^-4 are multiplied by `HBARC` on return. Entropy density is in fm^-3 with `k_B=1`.

For the scalar field `Phi=g_sigma sigma`, define

```
mD = M-Phi
U(Phi) = Phi²/(2 C_sigma) + b M Phi³/3 + c Phi⁴/4
W = C_omega nB
R = (C_rho/2)(np-nn)
Uvector = W²/(2 C_omega) + R²/(2 C_rho)
Vn = W-R/2,    Vp = W+R/2 .
```

For each species, `nu_i=mu_i-V_i` is the **kinetic chemical potential including its effective rest mass**, and `E*=sqrt(p²+mD²)`. Occupations are

```
f_i(p)    = 1/[exp((E*-nu_i)/T)+1]
fbar_i(p) = 1/[exp((E*+nu_i)/T)+1].
```

All quantities in these internal formulas, including `T`, are in fm units. With spin degeneracy two,

```
n_i = (1/pi²) integral p² (f_i-fbar_i) dp
rho_s_i = (1/pi²) integral p² (mD/E*) (f_i+fbar_i) dp
e_kin_i = (1/pi²) integral p² E* (f_i+fbar_i) dp
P_kin_i = (1/3pi²) integral p⁴/E* (f_i+fbar_i) dp.
```

Entropy is the usual occupation entropy of both particles and antiparticles. Thermal antiparticles are retained; the vacuum Dirac sea is omitted, as part of the specified Hartree ansatz. The code solves three simultaneous equations for `(Phi,nu_n,nu_p)`: two number constraints and

```
Phi/C_sigma + b M Phi² + c Phi³ = rho_s_n + rho_s_p .
```

On the solution,

```
e = e_kin_n + e_kin_p + U + Uvector
P = P_kin_n + P_kin_p - U + Uvector
s = s_n+s_p
f = e-Ts
mu_n = nu_n+W-R/2,    mu_p = nu_p+W+R/2.
```

The pressure is computed from the independent kinetic and field contributions, so the Euler identity `e+P=Ts+mu_n nn+mu_p np` is a useful numerical gate rather than its definition. The returned scalar, vector and kinetic energy decomposition sums back to `e`.

Returned Dirac mass is `mD`, whereas the diagnostic density-defined Landau masses are `sqrt(kF_i²+mD²)` at `kF_i=(3pi²n_i)^(1/3)`. At finite temperature there is no sharp occupied Fermi surface; these values report the quasiparticle slope at the density-defined momentum, rather than an asserted universal thermal transport mass. The vector fields shift single-particle energies without changing that momentum slope.

## Cold anchor and measured states

The cold solution selects the first scalar negative-to-positive residual crossing in `0<Phi<0.985M`, then applies bracketed root finding. The warm least-squares solve starts on that branch or a supplied nearby state. Its largest normalized equation residual must be below `2e-9`. This approach establishes a narrow tested local branch; it does not prove a unique physical root at arbitrary density or temperature.

From the source's rounded GM1 parameters, solving symmetric-matter `P=0` continuously gives:

| Quantity | Measured | NMB source's approximate GM1 target |
|---|---:|---:|
| n0 [fm^-3] | 0.1531429434 | 0.153 |
| E/A-M [MeV] | -16.3260551 | -16.3 |
| K [MeV] | 300.193040 | 300 |
| mD/M | 0.700109701 | 0.70 |
| J [MeV] | 32.5072472 | 32.5 |

This verifies the intended existing calibration within rounding. It is not a fresh fit, uncertainty estimate or comparison with nuclear data. The independent public [CompOSE GM1 catalogue entry](https://compose.obspm.fr/eos/54) identifies GM1 as a cold beta-equilibrated nucleon/electron EOS and lists the same nominal saturation-density, incompressibility and symmetry-energy values. Its cold table has not been downloaded or used as a finite-T reference here.

Sampled finite-temperature nuclear-only results:

| nB [fm^-3] | T [MeV] | Yp | mD [MeV] | mu_n [MeV] | mu_p [MeV] | e [MeV fm^-3] | P [MeV fm^-3] | s [fm^-3] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.16 | 1 | 0.5 | 647.183120 | 924.256088 | 924.256088 | 147.641130 | 0.255782564 | 0.015938416 |
| 0.16 | 10 | 0.1 | 652.158479 | 978.526264 | 861.822729 | 151.793127 | 4.23032934 | 0.132651075 |
| 0.32 | 30 | 0.1 | 472.073612 | 1119.52467 | 878.367485 | 320.926754 | 41.5630198 | 0.398630329 |
| 0.8 | 20 | 0.05 | 244.152129 | 1890.91027 | 1285.73365 | 1029.69519 | 465.773083 | 0.347356119 |
| 0.08 | 5 | 0.5 | 777.734803 | 917.252985 | 917.252985 | 74.3039677 | -0.572812742 | 0.070183241 |

The final row is intentionally retained as a uniform-matter value, not declared to be a stable physical phase. A separate explicit spinodal example at `nB=0.06 fm^-3,T=1 MeV,Yp=0.5` has `dP/dnB|T,Yp=-11.3409205 MeV`. It satisfies the local scalar equations while being mechanically unstable. Every returned state labels phase coexistence/mechanical stability as unassessed; users must not replace a phase construction with scalar-solver convergence.

## Independent identities and numerical qualification

At each of the five finite-T sample states, tests include:

- Number/scalar equation residuals, Euler identity, and exact energy decomposition.
- Adaptive quadrature refinement from relative tolerance `2e-11` to `2e-12` and tail cutoff from 60 T to 80 T.
- Independent 96-point Gauss-Legendre momentum-panel calculations of both species' net density, scalar density, energy, pressure and entropy.
- `s=-df/dT` at fixed `(nB,Yp)`; `df/dnB=(1-Yp)mu_n+Yp mu_p`; `df/dYp=nB(mu_p-mu_n)`.
- Pressure from the density derivative of free energy; `ds/dnB=-d[(1-Yp)mu_n+Yp mu_p]/dT`.
- Positive frozen-composition `de/dT`, with chemical composition held fixed rather than in beta equilibrium.
- Stationarity of free energy against scalar-field variation while **separately reinverting each species' density** through a scalar bracketed root. The local scalar curvature is positive at the tested points.

Two asymmetric warm states also pass cross-species Maxwell symmetry `dmu_n/dnp=dmu_p/dnn`. Three states at `T=0.01 MeV` reproduce the cold source-convention values before warm derivative tests are interpreted. One direct warm solve agrees with a temperature-continuation solve.

The independent integration rules test numerical implementation of the same specified physics. Thermodynamic identities test internal consistency. Neither establishes that the physical approximation matches an external finite-temperature GM1 calculation. That matched external forward comparison remains a separate acceptance gate before promoting the slice into production or claiming a precision EOS.

## A first compact product derived from free energy

`sommerfeld_thermal` provides the leading low-temperature approximation directly from the cold local branch, with density-of-states coefficient

```
gamma(nB,Yp) = sum_i kF_i E*F_i /(3 HBARC)
delta_f = -gamma T²/2
s = gamma T
Cv|nB,Yp = gamma T
delta_e = gamma T²/2
delta_P = (gamma-nB dgamma/dnB) T²/2
Gamma_thermal = 2-nB (dgamma/dnB)/gamma .
```

Here `kF_i,E*F_i` are in fm^-1, `T` is in MeV, and the density derivative is at fixed `Yp`. The derivative includes the **self-consistent cold density dependence of the Dirac mass**; omitting it generally gives the wrong thermal pressure. The implementation uses a documented central finite difference for that derivative. This is a reference asymptotic approximation, without fitted coefficients or AD claims.

The evaluator rejects temperatures above 0.03 times either species' cold kinetic Fermi energy `E*F_i-mD`. Three sample states at `T=0.1 MeV` pass entropy, thermal-energy, free-energy and thermal-pressure comparisons within `2e-4` relative error. The numerical gate is measured on those states, while the 0.03 threshold is a conservative proposed operating restriction, not a proof of an error bound throughout the allowed region. At tiny temperatures use the asymptotic increments rather than subtracting large full energies. At higher temperatures retain the converged reference and construct a separately qualified potential fit or table.

## Consequences for the next weak/transport work

This supplies the minimal interacting-state vertical slice: independently specified composition, temperature-dependent scalar field, kinetic chemical potentials, vector shifts and thermodynamic derivatives that can be checked. The next controlled problem should use these values in EOS-consistent charged-current kinematics, then compare composition relaxation and thermal losses in a homogeneous element.

Do not insert this state's chemical-potential difference into a vacuum static-target rate as a generic Q-value correction. Consistent energy conservation uses each nucleon's dispersion,

```
En(p)=sqrt(p²+mD²)+W-R/2
Ep(p)=sqrt(p²+mD²)+W+R/2,
```

together with momentum conservation, final-state blocking and the chosen weak matrix element. The current separate static-target kernel has its own classical-nucleon and physical neutron-proton mass-splitting convention. NMB's GM1 instead uses one bare `939 MeV` nucleon mass for both species. Adding the vacuum `1.293 MeV` splitting on top without changing the EOS convention would silently mix models. Keep the existing ideal-target closure for its conservative application tests; build a distinct matched RMF rate implementation.

The next bounded acceptance tasks are an independent code/derivation review, a matched finite-temperature external GM1 point or table with precisely the same conventions, and an implicit-differentiation/JAX implementation checked against these converged primal values. Only then expose EOS-consistent weak source terms. Full response functions, spin correlations, screening, collision widths, paired matter and nuclear composition are additional models, not inferred from the free-energy curve alone.

For low-density disks or outflows, use the separate classical/NSE/composition layer where appropriate. GM1 uniform matter does not provide a dilute nuclear composition, surface/Coulomb energies or a phase coexistence construction. For dense merger/cooling applications, causality, mechanical/compositional stability and model uncertainty need their own coverage; no all-state validity envelope is asserted here.

## Additional primary references

- [Glendenning & Moszkowski 1991, Phys. Rev. Lett. 67,2414](https://doi.org/10.1103/PhysRevLett.67.2414), the GM family reference cited by NMB and CompOSE.
- [Yang et al. 2024, Phys. Rev. C109,014908](https://doi.org/10.1103/PhysRevC.109.014908), a finite-temperature nonlinear Walecka calculation showing the grand potential, number/scalar densities with fermions and antifermions, and mean-field stationarity. It uses its own parameterization and is a formalism reference, not this checkpoint's numerical benchmark.
- [Serot & Walecka, Recent Progress in Quantum Hadrodynamics](https://arxiv.org/abs/nucl-th/9701058), underlying mean-field framework and model limitations.
