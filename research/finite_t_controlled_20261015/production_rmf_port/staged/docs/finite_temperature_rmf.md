# Uniform finite-temperature GM1 CPU reference

`nmbackend.reference.finite_t_rmf` is an explicitly imported scalar NumPy/SciPy
reference for one restricted uniform nuclear model. It does not change the
cold JAX RMFT/lepton/EOSTable/TOV/cooling APIs. It is not a full stellar EOS,
phase construction, neutrino backend or AD implementation.

```python
from nmbackend.reference.finite_t_rmf import state_density, diagnose_density
s = state_density(10., .16, .1) # T MeV, nB fm^-3, independent Yp
unstable = diagnose_density(1., .06, .1) # explicitly unadmitted spinodal
assert not unstable.locally_admitted
```

## Exact model and species

The fixed NMB-rounded GM1 convention is common bare neutron/proton mass939MeV,
Cs11.785,Co7.148,Cr4.410fm²,b.002947,c−.001070,HBARC197.3269804MeVfm.
It is spin-two relativistic Hartree n/p with thermal antibaryons and zero
Dirac vacuum sea. Number is NET particles minus antiparticles, with targets
nn=(1−Yp)nB,np=Yp*nB. Scalar density, energy, pressure and entropy add both.
No leptons, photons, beta/charge-equilibrium solve, nuclei, crust, pairing,
extra couplings, parameter family or physical vacuum n−p splitting is added.
Unknown parameter kwargs reject. Do not reuse the physical free-nucleon masses
or manually add a1.293MeV Q to this common-mass EOS. Weak-current mass/vertex
choices are not selected by this module.

## API, units and actual admission

`state_density(temperature_mev, baryon_density_fm3, proton_fraction, *,
controls=SolveControls(), initial=None)` requires a resolved locally convex
NUCLEAR two-charge branch. `diagnose_density` solves the same equations but
returns a labelled unadmitted spinodal/marginal state. Neither proves global
phase stability. Numerical/moment/branch failure raises ArithmeticError;
invalid inputs or strict physical admission failure raises ValueError.

`SolveControls(rtol=2e-11,tail=60.)` accepts finite non-Boolean real controls,
0<rtol<1e−4,30<=tail<=100. States require finite non-Boolean real scalars in
T1..50MeV,nB.02...4fm^-3,Yp.05...5. These bounded request guards and per-state
checks are not uniform accuracy/physical-domain certificates. ZeroT,zero
species, arbitrary arrays and other parameterizations are unsupported.

Outputs are immutable `UniformRMFState`; `asdict` returns named fields:

| Fields | Meaning / units |
|---|---|
|temperature,phi,omega,rho,dirac_mass,kinetic_mu_n/p,mu_n/p|MeV; kinetic mu includes mD, before vector shift; total mu is rest-inclusive|
|baryon_density,neutron_density,proton_density,scalar_density|fm^-3; species NUMBER is net, scalar density is summed particle+anti|
|energy_density,pressure,free_energy_density|MeVfm^-3,rest-inclusive; f=e−T*s|
|kinetic_energy_density|Dirac-GAS energy including mD rest; NOT a rest-subtracted ideal-particle kinetic increment|
|scalar_field_energy_density,vector_field_energy_density|MeVfm^-3; gas+scalar+vector sums to total energy|
|entropy_density,cv_fixed_density_composition|fm^-3,kB1; Cv at FIXED(nB,Yp), including scalar relaxation|
|hessian_n_p|2x2 dmu_i/dn_j at fixed T, order(neutron,proton),MeVfm³|
|density_defined_landau_n/p properties|sqrt[(HBARC*kFi)²+mD²] MeV; finiteT density-defined slope diagnostics, not a universal transport mass|
|scalar_curvature_fm_minus2,response_condition|canonical scalar curvatureD fm^-2; condition of row-scaled three-constraint Jacobian, dimensionless|
|hessian_min_eigenvalue,euler_residual|MeVfm³ and MeVfm^-3 respectively|
|phi_temperature_derivative|dPhi_MeV/dT_MeV at fixednB,Yp,dimensionless|
|antiparticle_density_n/p,log_antiparticle_density_n/p|ordinary fm^-3 when represented, otherwise None; logs mean ln[nbar/(1fm^-3)] and remain finite|
|cold_scalar_bracket,warm_scalar_bracket|internal numerical Phi bracket endpoints fm^-1, not publicMeV fields|
|locally_admitted,local_label,branch,global_phase,model|explicit local physical/numerical scope and fixed model identity|

Ordinary tiny antiparticles can underflow while bulk state is resolved; None
with finite log is a representation limitation, not exact physical absence.
No density floor, artificial threshold or zero positive bulk state is returned.
At exactly kineticnu=0, net density vanishes by particle/anti symmetry; the
number-root bracket uses this exact identity without asking an unrepresentable
positive cold pair gas to return zero bulk moments.

## Stationarity and thermodynamics

Internally energies are fm^-1,t=T/HBARC,M=939/HBARC,mD=M−Phi:

```
U=Phi²/(2Cs)+b*M*Phi³/3+c*Phi⁴/4
W=Co*nB; R=Cr*(np−nn)/2
Un=W−R/2; Up=W+R/2; mu_i=nu_i+Ui
U'=rhos_n+rhos_p
Vfield=Co*nB²/2+Cr*(np−nn)²/8
f=sum fgas(t,n_i,mD)+U+Vfield
```

The canonical free energy is stationary at fixed net species numbers. Gas
occupations are FD with kinetic total nu before vector shifts. Energy is
gas+U+Vfield; pressure independently integrates gas−U+Vfield. Summing
quasiparticle TOTAL energies then adding U+Vfield would double count vector
energy, because sum Ui*ni=2Vfield. No extra chemical/rest energy correction
is added.

Private moments include scalar, mass and thermal responses. For chi=n_nu,
B=rhos_nu=−n_m,Q=rhos_m|nu and canonicalD=U''+sum(Q+B²/chi),a=B/chi,

```
H=diag(1/chi)−a*a^T/D+[[Co+Cr/4,Co−Cr/4],[Co−Cr/4,Co+Cr/4]]
K=sum(rhos_t−B*n_t/chi)
Cv=t*[sum(s_t−n_t²/chi)+K²/D]
```

All partials here use INTERNAL t,nu,m. ConvertH byHBARC; Cv has fm^-3 units
for either internal or public temperature derivative. The positive scalar
relaxation term is required. Plugging mD into a public fixed-mass particle Cv
or summing separately fixed-particle and fixed-antiparticle Cv is incorrect.
These fixed-coupling responses are not density-dependent-RMF rearrangement,
parameter sensitivities or JAX derivatives.

## Branch and local stability

A cold scalar scan verifies an actual unique upcrossing in0<Phi<.985M.
Warm three-constraint continuation steps are at most5MeV. A separate48-interval
fixed-density scalar scan/nested number root at the final T verifies the same
canonical crossing. No missing bracket uses a zero index/clamp fallback.
An initial object must match this model but is never trusted as a solved-state
shortcut: cold-connected branch checks always rerun.

Normalized scalar/number residuals must be<=2e−9; nested number residual<=2e−10;
canonicalD>1e−8fm^-2; row-scaled Jacobian condition<1e8; finite positiveCv;
H symmetry at1e−9relative. Strict admission requires minimumH eigenvalue>
1e−7*max(1,||H||2) in returned units. These numerical margins are not a
physical uncertainty model. Negative/marginalH states are only diagnostic.
Positive fixed-Y compressibility alone is insufficient: the(.06,1,.1)
example is mechanically positive but has a negative coupledH eigenvalue.

Negative scalar quartic makes the unrestricted potential unbounded; this
is an explicitly restricted local effective-model branch. Local nuclear
convexity is not Gibbs coexistence, Coulomb/lepton constrained stability,
cluster/nuclear composition or a crust boundary. No metastable-state global
free-energy claim follows from this branch label.

## Evidence and open gates

Tests use an independently coded original-momentum scalar grand potential
and nested scalar/number roots before response comparisons. Saved fixtures
are explicitly INTERNAL-only; they are regression data, not external native
O2scl outputs. Selected forward states, tail/tolerance controls, pair-sensitive
mass moments, re-solved Cv/H/Maxwell/free-energy checks, spinodal and failed
bracket/numerical guard regressions are separate evidence. Detailed physical
quadrature/root is always the default; no surrogate fit replaces it.

The matched native O2scl forward gate remains OPEN: the pinned native fixture
has not executed on the available host. Same-equation numerical agreement is
not empirical calibration or model uncertainty. Full supported/minimum runtime
and normal admitted repository CI also remain separate gates. No JAX/AD/JIT,
Hessian-through-callback, cooling/evolution, weak rate, production transport,
phase-complete matter or broad domain certificate is supplied. Existing cold
APIs and exports are unchanged. Run the example or focused tests explicitly.

Model normalization/rounded parameters follow the [pinned original NMB cold RMF source](https://github.com/oshaughnessy-junior/NuclearMatter-Backend/blob/ffb5cc72949f2e5401251847aef4c9c7a98dde78/src/nmbackend/eos/rmft.py). A shared GM1 label or nominal saturation point is not an exact finite-T external benchmark.
