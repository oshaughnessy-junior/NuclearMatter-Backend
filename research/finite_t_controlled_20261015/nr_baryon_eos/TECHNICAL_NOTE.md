# Matched nonrelativistic Fermi-Dirac free-nucleon EOS

15 October 2026. New research reference pending independent review. This EOS matches the common-mass quadratic nucleon dispersion used by `response_reference/free_density_response.py` and the allowed-current optical construction. It replaces classical occupation assumptions where nucleon Pauli blocking is retained; it does not add interactions or relativistic kinematics.

## Dispersion, rest offsets and normalization

Each species has spin degeneracy two and dispersion

```
E_i(p)=O_i+p^2/(2M),   nu_i=mu_i-O_i,
f_i(p)=[1+exp((p^2/(2M)-nu_i)/T)]^-1.
```

The common kinetic mass is M=939 MeV. The default neutron/proton offsets are `(Q,0)` after subtraction of a constant common baryon rest energy. This retains the physical chosen neutron-proton energy difference without assigning distinct kinetic masses. Offsets may be changed explicitly but must be constants of the chosen state model; this free EOS does not include density-dependent mean fields or rearrangement terms. Chemical potentials supplied to the response include the offsets.

Using kinetic energy k, spin-summed density of states is

```
D(k)=sqrt(2) M^(3/2) sqrt(k)/[pi^2(hbar*c)^3].
```

The code integrates n=`integral D f dk`, kinetic energy K=`integral D k f dk` and entropy `s=-integral D[f ln f+(1-f)ln(1-f)]dk`. Total species energy density is `e=O*n+K`, pressure is `P=2K/3`, and free energy density is `F=e-Ts`. Energies and chemical potentials are MeV, densities fm^-3 and extensive thermodynamic densities MeV fm^-3.

A root solve in dimensionless `nu/T` imposes each positive species density separately. It does not impose weak equilibrium, charge neutrality between nucleons alone, or any relation between neutron/proton chemical potentials. Exact vacuum and extreme density/susceptibility underflow are rejected by the finite-T reference; `cold_state` supplies a separate exact T=0 formula for positive density.

## Positive fluctuation moments and fixed-density derivatives

Define `w=f(1-f)` and

```
A0=integral D w dk,
A1=integral D(k-nu) w dk,
A2=integral D(k-nu)^2 w dk.
```

The reference exposes the moments, the positive number susceptibility, fixed-density heat capacity and implicit temperature derivative:

```
(partial n/partial mu)_T = A0/T,
(partial n/partial T)_mu = A1/T^2,
(partial mu/partial T)_n = -A1/(T*A0),
Cv_n = [A2-A1^2/A0]/T^2.
```

The A matrix is a weighted Gram matrix. Cauchy-Schwarz gives `A2-A1^2/A0 >= 0`, with strict positivity for finite-T nontrivial support. This covariance form supplies the thermodynamic response of the same occupied dispersion used by Pauli-blocked transitions. A negative numerical variance or underflowed A0 raises an error rather than being silently repaired. Constant rest offsets cancel from Cv and number susceptibilities.

The identity `(partial e/partial mu)_T = mu*n_mu+T*n_T` includes the rest offset. The chemical potentials, energy and susceptibilities therefore have one common convention suitable for a later energy-equation derivation. No fluid equation is integrated in this batch.

## Reusing charged leptons and constructing response states

`matter_state(T,nn,np)` solves both FD nucleon species and calls the existing `thermodynamics/lepton_reference.py` density inversion for net electron density np. It introduces no duplicate lepton calculation. The returned `Matter` exposes baryon energy, pressure and fixed-independent-density heat capacity; it intentionally adds no photon contribution.

`Matter.pair('neutron')` returns the existing `PairState` for an n→p transition, with exact EOS chemical potentials and offsets. The reverse returns p→n. `Matter.optical('nue')` uses n→p and total electron chemical potential; `Matter.optical('antinue')` uses p→n and minus that electron chemical potential. Thus the equilibrium neutrino parameters satisfy

```
mu_nue = mu_e + mu_p - mu_n,
mu_antinue = -mu_nue.
```

This is an interface convention, not a claim that the state is in weak equilibrium or that all optical channels have been supplied. The factory makes no collision-moment calls and no claim about completeness of neutron-decay/crossed channels. It prepares thermodynamically matched inputs for those kernels once independently reviewed.

Example factory point: T=2 MeV, nn=`7e-5 fm^-3`, np=`3e-5 fm^-3`, giving total electron chemical potential `18.2819793680 MeV`. With the declared offsets, equilibrium neutrino parameter is `15.2730449987 MeV`; the opposite flavor is its negative. This off-equilibrium test point is deliberately not forced onto a beta-equilibrium sequence.

## Numerical checks actually run

Run:

```
python3 /Users/rossma/Genesis_LANL/outputs/nmb-finite-t-20261015/nr_baryon_eos/check_reference.py
```

All 96 author checks pass. Species states sample `(T[MeV],n[fm^-3])=(2,1e-7),(1,0.001),(0.05,0.005)`. Checks include:

- Density inversion residuals; Euler identity `e+P=Ts+mu*n`; pressure/kinetic virial relation; positive susceptibility and Cv.
- Independent scalar momentum-space quadrature for n and K, compared with the energy-space vector integration.
- Tail/tolerance refinement of kinetic chemical potential, energy, entropy, Cv and susceptibility.
- Three-step finite differences of fixed-density Cv and chemical-potential temperature response, free-energy density derivative and potential-derived pressure.
- Maxwell relation `s_mu=n_T`, checked by independent entropy differences.
- Dilute classical chemical potential, pressure, energy, Cv and susceptibility.
- T/EF=`0.01` and `0.003` cold approach, Sommerfeld chemical potential and leading degenerate heat capacity.
- Constant rest-offset shifts and charge-neutral response/optical factory signs.

The largest density residual is `2.67e-15`; Euler residual `2.09e-15`; independent momentum-density difference `2.45e-15`; fixed-density Cv finite-difference relative error `1.35e-8`; pressure potential-derivative relative error `5.34e-8`. Largest Maxwell finite-difference relative error across the step ladder is `1.98e-6`. The leading Sommerfeld expression is an asymptotic limit, so its discrepancy has the expected retained-order scale, rather than an arbitrary numerical-integration tolerance.

The JSON records each measured check and its gate, source/dependency SHA-256 at both ends, exact command, constants through serialized states, library versions, domain diagnostics and factory sample. Source drift rejects the run. It retains `publication_status: Requires independent review; no fluid integration`.

## Physical domain and omissions

The sampled species have pF/M approximately `0.00302`, `0.0650` and `0.1112`, with T/M respectively `0.00213`, `0.00106` and `0.0000532`. These diagnostics describe the NR expansion parameters; they do not establish a uniform error bound on observables or justify use at arbitrary neutron-star density. Finite-T high-momentum tails also exist. A later comparison with relativistic free gases should quantify the actual observable correction where required.

No nucleon antiparticles, interactions, clusters, pairing, mean fields, magnetic fields or effective-mass dependence are included. Number susceptibility is not a complete interacting dynamical weak response. A common 939 MeV kinetic mass plus a finite Q is a deliberately approximate dispersion, not exact neutron/proton relativistic mass kinematics. Conserved energy in any later coupled calculation must use the same occupied FD EOS and rest offsets, rather than the previous `3T/2` classical baryon energy. Crossed optical completeness and consistent collision-energy moments require independent review before a coupled fluid demonstration.
