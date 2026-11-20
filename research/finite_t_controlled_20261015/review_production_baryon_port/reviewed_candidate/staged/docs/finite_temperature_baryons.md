# Particle-only finite-temperature baryon reference

`nmbackend.reference.finite_t_baryons` is a separately imported scalar CPU
NumPy/SciPy utility for one ideal relativistic spin-two **particle** Fermi gas.
It is useful for controlled free-npe reference calculations. It omits
antiparticles, interactions, mean fields, composition constraints and weak
rates. Existing cold/JAX/EOSTable/cooling APIs and exports are unchanged.

```python
from nmbackend.constants import M_N, M_P
from nmbackend.reference.finite_t_baryons import state_mu, state_density

neutron = state_density(2., .001, mass_mev=M_N)
proton = state_density(2., .0002, mass_mev=M_P)
```

Each call is independent. `particle_density` is the positive particle NUMBER
in fm^-3, never a particle-minus-antiparticle baryon density. No neutrality,
neutron/proton relation, beta equilibrium or inter-species chemical potential
is imposed. The model remains mathematically defined where a complete
pair-equilibrated physical gas would also need antiparticles; their absence
is a model choice, not a claim that they are negligible at arbitrary T/m.
There is no hidden antiparticle diagnostic veto. Physical application must
justify omission of those species separately.

## Public API and units

`state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_N, *,
rtol=2e-12, tail=60.0)` returns immutable `ParticleState`.

`state_density(temperature_mev, particle_density_fm3, mass_mev=M_N, *,
rtol=2e-12, tail=60.0)` inverts strictly positive particle number. Zero or
negative particle density is rejected; neither is converted to a small floor.

| Field | Meaning / units |
|---|---|
| temperature, mass, chemical_potential | MeV; T means kB times temperature, mu includes rest mass |
| kinetic_chemical_potential | mu−mass, MeV; not total mu or an interacting Hartree potential |
| particle_density | Positive particle NUMBER, fm^-3; spin-two normalization already included |
| kinetic_energy_density | Direct integral of E−mass, MeV fm^-3; not obtained by subtracting two rest-dominated energies |
| energy_density | mass*particle_density+kinetic_energy_density, MeV fm^-3 |
| pressure, free_energy_density | MeV fm^-3; F=mu*n−P=E−T*s |
| entropy_density, cv_fixed_particle_density | fm^-3, kB=1 |
| dn_dmu_fixed_temperature, dn_dtemperature_fixed_mu | fm^-3 MeV^-1 |
| dmu_dtemperature_fixed_particle_density | Dimensionless |

Every thermodynamic response holds the supplied mass fixed. Any finite
positive declared bare or quasiparticle mass can parameterize this ideal gas.
That does not supply density-/temperature-dependent mass derivatives,
rearrangement terms, calibrated interactions, or effective-mass thermodynamics.
This module never selects or replaces a weak-current/spinor mass convention.

The exact package HBARC, M_N and M_P constants are used. Current NMB neutron/
proton masses are939.565421/938.272088 MeV; some research capture conventions
use939.56542052/938.27208816 MeV. Pass those explicit masses when comparing
that convention. Neither convention is silently substituted for the other.
To convert entropy/Cv to erg cm^-3 K^-1 multiply by1e39*kB(erg/K); energy/
pressure to erg cm^-3 multiply by1e39*MeV_to_erg. These are not geometric TOV
units or fm^-4 energies.

## Equations and numerical policy

For E=sqrt(p²+m²), f=expit((mu−E)/T), the spin-two measure is
`p² dp/[pi² HBARC³]`. Particle number uses f, total energy uses E*f,
kinetic energy uses (E−m)*f, and pressure uses p²/(3E)*f with that measure.
Entropy directly integrates−f log(f)−(1−f)log(1−f); no subtraction of large
rest/chemical terms is needed for it.

The implementation uses x=(E−m)/T, a separately normalized occupation and
dimensionless geometry/moment scales. Occupation and dimensional factors
are combined in logarithmic restoration before ordinary float64 rounding.
Relative adaptive quadrature uses zero absolute tolerance, with panels around
the Fermi scale and selected tails. No Maxwell fallback or surrogate is used.

With w=f(1−f), A0=integral(w), A1=integral[(E−mu)w], the responses are
chi=A0/T, n_T=A1/T² and mu_T|n=−A1/(T*A0). Cv is evaluated as a SECOND
positive centered integral `integral[(x−mean_w(x))²*w]`, equivalent to the
fixed-particle-number covariance divided by T². This avoids subtraction of
two large dilution/rest-dominated second moments. A resolved positive
susceptibility/Cv is required; unresolved values are not replaced by floors.

Density inversion solves a logarithmic density residual in the dimensionless
kinetic chemical potential. An NR estimate is used only to start its bracket,
not to replace the FD source. The final returned ordinary total mu must
reconstruct the requested density to1e−9 relative. If mu−m cannot be encoded
in that total mu, the forward residual rejects rather than claiming a tiny
root. There is no public log-density or arbitrary-precision API.

Inputs must be finite non-Boolean real scalars, T>0,m>0,0<rtol<1e−4 and
30<=tail<=100. Invalid inputs raise ValueError. Unresolved quadrature,
nonfinite outputs, unrepresented positive density/susceptibility/Cv or root
reconstruction failure raise ArithmeticError. Ordinary kinetic energy or
pressure can underflow; a zero/subnormal field is not a relative-precision
certificate or a statement of exact physical absence. The tolerance controls
scaled quadrature, not uniform relative accuracy of every conditioned output.
No T=0, zero-particle branch or full-domain error rectangle is certified.

## Selected validation and remaining limits

Forward capture-bath checks use the accepted free particle EOS with exactly
matched masses, before response tests. Separate original-momentum integrals
check forward quantities and responses at selected dilute, warm, degenerate,
relativistic and declared-mass states. Scaled Bessel formulas check resolved
small populations; cold/Sommerfeld and nonrelativistic formulas check limits.
Legendre identities, pressure differentiation, Maxwell response relations,
fixed-density temperature differences, tiny positive roots, dimensionful
T/m scaling, tail/tolerance controls and total-mu representability guards
are tested. This is selected numerical evidence inside an ideal-gas model.
It is not empirical uncertainty, a complete matter EOS, a continuous-domain
certificate or qualification of differentiating through quadrature/roots.

Run `PYTHONPATH=src python examples/finite_t_baryons.py` or the focused
`PYTHONPATH=src pytest tests/test_finite_t_baryons.py`. This CPU module does
not provide JAX AD/JIT/vmap, baryon-antibaryon thermodynamics, composition or
energy inversion, a cold-energy subtraction API, source transport or cooling.
Minimum/supported dependency-version execution and normal package CI remain
separate release gates from any scalar benchmark results.
