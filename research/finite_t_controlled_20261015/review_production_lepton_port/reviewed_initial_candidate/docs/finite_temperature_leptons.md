# Finite-temperature ideal lepton reference

`nmbackend.reference.finite_t_leptons` supplies scalar CPU NumPy/SciPy states
for an ideal relativistic spin-two lepton/antilepton gas. It is an installed
reference utility, separate from the cold JAX EOS and NSCool cooling kernels.
It does not provide JAX AD, JIT, vmap, an interacting matter EOS or weak rates.

```python
from nmbackend.constants import M_E, M_MU
from nmbackend.reference.finite_t_leptons import state_mu, state_density

electron = state_density(20., 1e-6, mass_mev=M_E)
muon = state_density(20., .005, mass_mev=M_MU)
pairs = state_mu(20., 0., mass_mev=M_MU)
```

Each flavor has its own total chemical potential or signed net **number**
density. Net number is the negatively charged lepton population minus the
positively charged antiparticle population: electric charge density is
`-e*n_net`. These calls do not impose neutrality, beta equilibrium or
`mu_e=mu_mu`. Pair equilibrium within a flavor sets `mu_anti=-mu`. Net zero
retains thermal pairs; setting a muon net number to zero does not disable
muons. Matter composition, baryons, photons and any further constraints must
be supplied separately.

## API and units

`state_mu(temperature_mev, chemical_potential_mev, mass_mev=M_E, *,
rtol=2e-12, tail=60.0)` returns an immutable `LeptonState`.

`state_density(temperature_mev, net_density_fm3, mass_mev=M_E, *,
rtol=2e-12, tail=60.0)` independently inverts the signed net number. It checks
reconstruction to 1e-9 relative and has a distinct exactly-zero pair branch.
The initial scale uses the pair susceptibility, so very small signed net
numbers are not discarded by an absolute-MeV root tolerance.

| Field | Meaning / units |
|---|---|
| `temperature`, `chemical_potential`, `mass` | MeV; T means kB times physical temperature; mu includes rest mass |
| `net_density` | n_minus−n_plus, fm−3 |
| `particle_density`, `antiparticle_density` | Separate nonnegative populations, fm−3 |
| `energy_density` | Total rest-inclusive energy of both populations, MeV fm−3 |
| `pressure`, `free_energy_density` | MeV fm−3; free energy is energy−T entropy |
| `entropy_density`, `cv_fixed_net_density` | fm−3 with kB=1; Cv holds net number fixed while pairs equilibrate |
| `dn_dmu_fixed_temperature` | Positive number susceptibility, fm−3 MeV−1 |
| `dn_dtemperature_fixed_mu` | Net-number derivative, fm−3 MeV−1 |
| `dmu_dtemperature_fixed_net_density` | Dimensionless |

`thermal_energy_density(state, *, rtol=2e-12, tail=60.0)` returns
`e(T,n_net)−e(0,n_net)` in MeV fm−3 without subtracting two large cold
energies. Use a state returned by this module. This is one flavor's thermal
increment at fixed net number, not total-matter heat. No extra rest-mass or
reaction-Q source belongs in its energy accounting.

To convert entropy or Cv to erg cm−3 K−1 multiply by `1e39*kB` with kB in
erg/K. Convert energy and pressure using `1e39*MeV_to_erg`. The cold
`eos.leptons.lepton_sector` uses mu in fm−1 and energy in fm−4: multiply
those quantities by HBARC to compare with the MeV conventions here. It also
imposes cold inter-flavor chemical equilibrium, unlike these independent
component states. Existing cold and NSCool APIs are unchanged.

## Equations and numerical contract

For E=sqrt(p²+m²), occupations are `f_minus=expit((mu-E)/T)` and
`f_plus=expit((-mu-E)/T)`. The spin-two measure is
`p² dp/[pi² (hbar*c)³]`. Net number uses their difference; total energy and
pressure use their sum. The implementation integrates kinetic energy
`k=E-m`, splitting at the Fermi scale and thermal tails. It uses stable
near-pair-symmetry subtraction and occupation entropy.

Writing w=f(1−f), the response moments are
`A0=integral(w_minus+w_plus)`,
`A1=integral[(E-mu)w_minus−(E+mu)w_plus]`, and
`A2=integral[(E-mu)²w_minus+(E+mu)²w_plus]` with the same measure.
Then `chi=A0/T`, `n_T=A1/T²`, `mu_T|n=−A1/(T*A0)` and
`Cv|n=(A2−A1²/A0)/T²`. These are thermodynamic response fields, not
qualifications of differentiation through adaptive quadrature or the root.

Finite real, non-Boolean scalar inputs are required: T>0, m>=0,
`0<rtol<1e-4` and `30<=tail<=100`. T=0 is rejected; use the existing cold
EOS for that limit. Invalid inputs raise `ValueError`. Unresolved quadrature,
net-number residuals, nonfinite responses or susceptibility underflow raise
`ArithmeticError`. There is no density floor or silent classical fallback.

A returned ordinary antiparticle population can underflow to zero while the
state susceptibility remains resolved. That zero is not a claim of exact
absence, threshold closure or relative tiny-population accuracy. The module
does not supply log-population asymptotics. Very cold dilute massive-pair
states with unresolved susceptibility are rejected. No uniform parameter
rectangle, broad T→0 asymptotic scheme or arbitrary precision is certified.

Tests cover exact massless formulas, independent scalar original-momentum
integrals at selected electron/muon states and both signs, zero-net pairs,
very small signed massless net numbers, nonrelativistic Maxwell–Boltzmann
and massive cold/Sommerfeld limits, fixed-net-number finite differences,
thermodynamic identities and selected tail/tolerance refinements. They
establish selected ideal-gas numerical benchmarks, not interacting-matter
uncertainty or empirical validation. The package's existing HBARC, M_E and
M_MU constants are used directly.

The ideal-gas formulas and response identities are independently written;
no external source code, EOS table or campaign runtime file is needed.
Background conventions: [Timmes Fermi–Dirac integrals](https://cococubed.com/code_pages/fermi_dirac.shtml)
and [Timmes & Swesty 2000](https://doi.org/10.1086/313304). In particular,
rest-mass-subtracted degeneracy variables used by some EOSs must be
converted before comparison with this module's total mu.

Run the example with `PYTHONPATH=src python examples/finite_t_leptons.py`.
Focused tests are `PYTHONPATH=src pytest tests/test_finite_t_leptons.py`;
existing cold derivative/unit tests remain separate compatibility gates.
