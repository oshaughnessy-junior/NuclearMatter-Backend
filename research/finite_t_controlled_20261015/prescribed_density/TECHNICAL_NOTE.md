# Prescribed-density homogeneous element

15 October 2026. Numerical research reference pending independent review. Density is an imposed function of time; no hydrodynamic momentum, geometry, gravity or transport equation is solved. The purpose is to verify compression work, dilution, thermodynamic derivatives and weak composition/energy bookkeeping within the existing warm dilute model.

## Common EOS and held-fixed variables

Matter contains classical free neutrons and protons with a common 939 MeV kinetic mass and a retained neutron-proton rest offset Q, ideal finite-mass electrons/positrons, and photons. Local electromagnetic thermal/pair equilibrium is imposed; photons comove with matter and do not escape. Neutrinos escape freely when weak reactions are enabled. Transparency is an ansatz rather than a parcel-size opacity calculation.

For baryon density n, electron fraction Y and temperature T in energy units, charge neutrality gives net electron density `n_l=Yn`, and

```
u(T,n,Y) = (1-Y)Q + 3T/2 + [epsilon_l(T,Yn)+a T^4]/n,
P(T,n,Y) = nT + P_l(T,Yn) + a T^4/3,
a = pi^2/[15(hbar*c)^3].
```

u is energy per baryon after subtraction of one constant common nucleon rest energy. All relevant species-dependent rest energy, including electron/positron mass, remains. There is no additional nuclear Q heat.

Let `mu_T=(partial mu_l/partial T)_n_l`. At independent `(T,n,Y)`, define `A_l=mu_l-T*mu_T`. The consistent energy derivatives are

```
u_T = 3/2 + Cv_l,fixed_net_density/n + 4 a T^3/n,
u_Y = -Q + A_l,
u_n = [n Y A_l - epsilon_l - a T^4]/n^2.
```

The `u_n` derivative is at fixed T and Y, so net electron density varies when n changes. It is not the derivative at fixed electron density. This distinction controls compression heating of electron/positron-rich matter.

## First law, dilution and reservoirs

For externally specified `n(t)`, signed mechanical work per baryon is `dot(W)=P*dot(n)/n^2`, positive under compression. With positive neutrino energy escape per baryon q and net weak composition source F, evolve

```
dot(u) = P*dot(n)/n^2 - q,
dot(Y) = F,
dot(T) = { [P/n^2-u_n]*dot(n) - q-u_Y*F }/u_T.
```

The numerical variables `(T,mu_l)` avoid repeated charge inversions. Charge neutrality requires the density term as well as thermal and reaction terms:

```
n_mu*dot(mu_l) = n*F + Y*dot(n) - n_T*dot(T).
```

Here `n_mu=(partial n_l/partial mu_l)_T` and `n_T=(partial n_l/partial T)_mu_l`. Omitting `Y*dot(n)` would incorrectly change charge per baryon during a weak-off expansion or compression. The RHS verifies this transformed charge identity and the differential energy equation at every evaluation.

The integrals of emitted flavor counts, escaped energy and signed mechanical work obey

```
u(t)-u(0)+E_escaped(t)/baryon-W_on_matter(t)/baryon = 0,
Y(t)-Y(0)+N_nue,escaped(t)/baryon-N_antinue,escaped(t)/baryon = 0.
```

Mechanical work is an external reservoir. Matter energy alone is not conserved, and weak-off expansion still removes energy through work. Global baryon number is fixed for this Lagrangian parcel, with volume inversely proportional to n; no baryon source is applied.

## Weak-off entropy benchmark before weak-on execution

The classical thermal wavelength is `lambda=[2*pi*(hbar*c)^2/(939*T)]^(1/2)`, with spin degeneracy two for each nucleon. Specific entropy and free energy are evaluated independently as

```
s_bar = sum_i Y_i [5/2-ln(n Y_i lambda^3/2)] + [s_l+4aT^3/3]/n,
f_bar = (1-Y)Q + T sum_i Y_i [ln(nY_i lambda^3/2)-1]
        + [f_l-aT^4/3]/n,
```

where `Y_p=Y`, `Y_n=1-Y`. These satisfy `u=f_bar+T*s_bar`. The implementation first checks u_n by density finite differences, pressure by `P=n^2*(partial f_bar/partial n)_T,Y`, and the weak-off adiabatic slope by independent entropy differences:

```
(dT/dn)_s,Y = -s_n/s_T = [P/n^2-u_n]/u_T.
```

Three states use a decreasing density-step ladder. Weak-off expansion and compression then verify both specific entropy conservation and integrated work/energy balance. Only after those tests complete does the script enable weak reactions.

## Executed trajectories and numerical gates

Density is prescribed as

```
n(t)=n0 exp[ln(density_factor)*t/duration].
```

Examples start at `(T,n,Y)=(2 MeV,1e-7 fm^-3,0.3)`, run for 0.1 s and change density by factors 1/2 and 2. Both weak-off paths use ODE tolerances `2e-6` and `2e-9`. Both weak-on paths additionally refine the weak quadrature from order 96 to 160 at fine ODE tolerance. The channels are electron capture, positron capture and neutron decay from the existing static target kernel. No kinetic or thermodynamic equilibrium is forced.

The bounded evaluation guard is `0.25<T<8 MeV`, physical open-interval composition, and `max(n_n,n_p)*lambda^3/2<0.02`. A 0.8 MeV terminal event is present; the demonstration requires reaching the prescribed density endpoint without hitting it. This is a small tested trajectory domain, not broad validation of the classical/static physics.

EOS density and pressure derivative errors must be below `1e-6`, entropy-slope error below `1e-6`, weak-off relative entropy drift below `1e-7`, and global scaled energy/lepton ledgers below `1e-6`. Matched-time ODE and weak-order refinements must agree within `1e-5`. The energy scale excludes the constant common baryon rest energy.

Run:

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/prescribed_density/element.py
```

The report records source/dependency hashes checked at both ends, command, constants, library versions, derivative ladders, per-run energy/entropy/lepton diagnostics, refinement measurements and small physical histories. A provenance drift rejects publication of the run. New results require independent review before being treated as accepted campaign evidence.

## Physical restrictions

This is a controlled homogeneous work/cooling experiment with classical static nucleons. It excludes finite recoil, thermal target velocities, nucleon Pauli blocking, mean fields, interactions, pairing, nuclei, muons and magnetic fields. Its physical accuracy is not established by small conservation residuals. Photon retention, electron/positron thermal equilibrium and neutrino transparency are explicit ansatz. The disconnected antineutrino support of the static kernel remains; this transparent escaped-radiation bookkeeping does not imply global radiation thermalization. Matter entropy is conserved only in the weak-off reversible benchmark; the weak-on system emits neutrino energy and entropy, so matter entropy alone is not a closed-system entropy budget.

## Measured numerical results, pending independent review

All ten trajectory runs pass, with three derivative-validation states and six solver/quadrature comparisons. The refined endpoint results are:

| Density factor | Weak reactions | Final T [MeV] | Final Ye | Escaped energy [MeV/baryon] | Work on matter [MeV/baryon] |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5 | off | 1.581552694 | 0.3 | 0 | -8.884152236 |
| 2 | off | 2.529828699 | 0.3 | 0 | 11.388042204 |
| 0.5 | on | 1.499304859 | 0.532985139 | 6.823675174 | -7.978737998 |
| 2 | on | 2.182391832 | 0.536274130 | 19.725418314 | 9.112207607 |

Largest scaled energy and lepton residuals across the suite are `1.70e-12` and `2.91e-12`. Weak-off relative specific-entropy drift is below `1.14e-15`. Largest matched-history refinement difference is `3.22e-10`. These exceptionally small weak-off errors describe a smooth, short, well-resolved benchmark and do not establish comparable accuracy for larger density excursions or different physical models.

The separate `entropy_oracle.py` solves constant specific entropy for temperature as a function of density, rather than integrating the temperature/chemical-potential ODE. It then independently integrates `P/n*dln(n)` on 9- and 17-point paths and compares that work with the EOS endpoint energy difference. Run it after the main script:

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/prescribed_density/entropy_oracle.py
```

Both directions pass. The largest ODE-to-entropy-root relative temperature difference is `6.66e-14`. The independent refined work integral differs from the EOS energy difference by less than `3.12e-10` on its work scale; grid refinement changes it by less than `4.71e-9`. Its report includes result and code hashes. This is independent numerical formulation of the same EOS, not independent interaction physics.

`initial_failure.json` preserves a corrected implementation error: a scalar lepton-ledger variable initially shadowed the imported lepton module during weak-off output processing. Renaming it repaired execution without changing physical equations or numerical gates. The failed run did not reach weak-on execution and supplied no published results.
