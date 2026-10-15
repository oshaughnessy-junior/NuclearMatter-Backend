# Actual free relativistic charged-current particle-hole tensor

15 October2026. **Author-frozen, independent review required.** This leaf implements all16 complex components of the free relativistic particle-hole response and selected differential absorption/emission coefficients. It does not yet integrate the charged-lepton phase space, supply a total neutrino opacity or emissivity, use density-root matter compositions, or implement GM1. Inputs are specified total chemical potentials, masses and temperature; they are not claimed charge-neutral stellar states.

The implementation follows the independently reviewed [response contract](../relativistic_response_contract/RESEARCH_CONTRACT.md), [Guo et al.2020](https://arxiv.org/html/2006.12051v2) current/phase-space conventions, and [Roberts & Reddy2017](https://arxiv.org/html/1612.02764v2) tensor normalization. The accepted source-defined vacuum `vertices` building block is reused without invoking its vacuum `trace` function. Code, raw tensors, controls, oracle comparison and failure snapshots are retained here. Other contributors' frozen files remain unchanged.

## Explicit state/current API

`State(T,mass_a,mass_b,mu_a,mu_b,weak_mass)` describes an ordered **a→b absorption** transition with free positive-energy dispersions `Ei=sqrt(p²+mi²)`, no vector potential, and total chemical potentials including rest mass. No baryon-antibaryon states or timelike pair-creation cuts are included. Particle-only thermodynamic validity must be checked separately before using a self-consistent EOS state.

`weak_mass` is an independent bare vertex normalization; it defaults to the physical average938.91875434MeV. Spinors use mass_a,mass_b. Neither the density state nor a changed spinor mass silently changes the WM/PS normalization. `FormFactors` explicitly supplies GV,GA,F2,GP and their switches. Physical q is also kinetic q in this free U=0 leaf. In-medium q/qstar mapping is absent; passing effective masses does not make this a GM1 current.

The full vertex is `GV gamma^mu−GA gamma^mu gamma5+iF2 sigma^(mu nu)q_nu/(2Mweak)−GP gamma5 q^mu/Mweak`. Couplings, dipole/pion-pole forms, bare-mass factors and antineutrino signs retain the separate full-current leaf's conventions. The full unaveraged baryon trace is

`T^(mu nu)=Tr[(slash pb+mb)Gamma^mu(slash pa+ma)barGamma^nu]`.

No target1/2 average is inserted. The trace already contains the target spin-two sum. GF and Vud are absent from the returned W. The charged-lepton tensor has one incident active neutrino helicity; for antineutrinos its antisymmetric part is reversed through transposition and its outgoing spin sum has slash l−ml. The effective charged-lepton chemical potential is an explicit input: use−mu_l for the positive lepton in antineutrino absorption.

## Exact q-axis reduction and support

With `omega=Enu−El`, q along z and `pb=pa+q`, the energy delta fixes

`x0=cos(pa,q)=[ma²−mb²+omega²−q²+2omega Ea]/(2pq)`.

The absorption tensor is

`Wabs^(mu nu)=1/(16pi²q) ∫allowed dEa ∫0^2pi dphi T^(mu nu) fa(1−fb)`.

The separately integrated emission tensor has the **same forward current orientation** and occupation product `fb(1−fa)`. It is not constructed by multiplying Wabs by an exponential. The angular delta Jacobian Eb/(pq) cancels the state-normalization Eb; the full trace, spin convention and azimuth give the stated prefactor.

Allowed energy intervals satisfy positive energies `Ea>=ma`, `Ea+omega>=mb`, and

`A Ea²+B Ea+D>=0`,
`A=q²−omega²`, `C=ma²−mb²−A`, `B=−omega C`, `D=−(C²/4+q²ma²)`.

`support` uses the quadratic sign to choose intervals: spacelike A>0 has an unbounded physical branch; timelike A<0 may give a finite interval or empty support. A==0 is treated as its actual linear inequality, not division by a tiny quadratic coefficient. Stable factored differences and quadratic roots reduce cancellation. Positive-energy constraints are applied to the intervals. q=0 is explicitly rejected as a singular coordinate, not assigned zero scattering.

306 direct angle/energy sample checks cover physical and synthetic spacelike/timelike/linear/empty cases. These supplement the analytic derivation rather than proving arbitrary floating-point input coverage. Selected Gauss nodes reconstruct angles within their support; a materially out-of-domain angle rejects the result. A roundoff-only angle clamp is recorded, not used to introduce physical support or width.

The free current's azimuth dependence is at most quadratic in transverse momentum. Periodic8/16-point integration retains every complex component, including antisymmetric ones. No current-conservation identity removes components; unequal masses are retained explicitly.

## Thermal integration and representation

Energy Gauss panels split exact support boundaries and both species' Fermi regions. Positive semi-infinite intervals are truncated beyond the larger relevant Fermi energy/support lower edge by an explicit thermal tail. A finite timelike upper endpoint is retained. Energy16→24→32, phi8→16 and tail45→60→80 refinements are recorded. No smoothing of the response or Pauli factors is introduced.

Absorption and emission use separate log occupation weights and separately selected scales. The returned `scaled` complex tensor and finite `log_scale` are the authoritative representation; optional ordinary tensors have explicit representability flags. Each component combines its log magnitude with the scale **before** exponentiation, so a small scale is not prematurely declared underflow when the spin trace makes a component representable. A pure helper check exercises that case. Empty support has exact structural zero metadata. Nonfinite occupation logarithms reject the request. Neither floors nor arbitrary replacements for unrepresentable rates are used.

For `differential`, the covariant contraction gives

`dchi/(dEl dc)=GF²Vud²/(32pi² hbar) * pl/Enu * (1−fl) L_mu nu Wabs^(mu nu)`,
`dj/(dEl dc)=GF²Vud²/(32pi² hbar) * pl/Enu * fl L_mu nu Wem^(mu nu)`.

Values have units **s^-1MeV^-1 per unit cosine**; W is MeV². No extra c or target spin factor appears. The output includes its reconstructed lepton four-vectors, complex L, complete response, full inputs and quadrature controls. This is not yet an angle/charged-lepton-energy-integrated j or chi. Crossed three-body channels need their own lepton trace and kinematics before a complete warm inventory can be claimed.

## Measured forward points

All masses are physical vacuum mn939.56542052/mp938.27208816, Mweak938.91875434MeV, with the selected source-defined full current. Each pair is ordered for absorption; mu_l below is the effective chemical potential of the charged particle in that channel.

| Point | T; mu_a,mu_b [MeV] | Enu,El [MeV]; cosine; mu_l | differential j | differential chi |
| --- | --- | --- | ---: | ---: |
| n→p, electron |4;945,938|40,38;.2;20|1594.87801399914|1362119.54301627|
| n→p, muon |12;950,940|150,120;−.3;80|5744.79464664325|4514098.74591987|
| p→n, antimuon |12;940,950|160,125;.4;−30|.286482341830150|936515.752349251|
| n→p, timelike electron |4;945,938|.05,1.25;.8;20|9247.80384903809|363.086602645664|

The final point has omega=−1.2MeV,q=1.10118919132MeV and **finite support942.607056364≤Ea≤11875.7769907MeV**; it would be lost by an unjustified spacelike-only support assumption. The thermal tail truncates the negligible occupied portion well before its high kinematic endpoint, with explicit tail refinement.

A separate **exact lightlike** response test uses omega=−.5,q=.5MeV at T20 with physical masses and mu_a950,mu_b940. Both occupancy branches are nonempty. Energy/phi/tail refinement changes the tensor norm by at most1.33e−12. Its support solver uses the linear branch directly; no limiting omega or q displacement is inserted.

## Internal checks, absolute limits and independent oracle

`validate.py` reports **422/422 author checks in.59s**: the four differential points, separate j/chi KMS, all-component refinement, Hermiticity, positive-semidefinite W, positive physical contractions, exact support cases, actual lightlike integration, independent weak-mass parameter, representation behavior and matched NR limits. KMS is checked only after separate occupancy integrations, using `log(j/chi)=−[Enu−(mu_l+mu_b−mu_a)]/T`.

For a density-only point vertex with common mass and fixed kinetic chemical potentials, W00 approaches the frozen spin-two NR response multiplied by(hbarc)^3. The relative differences decrease7.41e−4→7.37e−5 for M1e4→1e5MeV along a controlled thermal momentum/energy path. For the point allowed-current differential coefficients, the absolute prefactor comparison with the frozen NR kernel gives relative differences1.56e−4→1.48e−5 for M1e5→1e6. These are finite-mass model-limit differences, not integration errors; the original declared3e−4/3e−5 selected coefficient gates are retained. Full form factors are not compared to a contact NR current as if they were the same model.

The independent numerical worker supplied a separately authored, frozen forward oracle at the three agreed spacelike points. Only point inputs and frozen **data** were exchanged; its implementation was not used to construct this tensor. `compare_oracle.py` checks full16 complex Wabs,Wem,L and j,chi, with exact shared state/lepton inputs: **51/51 checks pass**, maximum forward data difference1.66e−14 relative. This is a strong forward gate between separate calculations, not automatic independent review or a qualification of all tensor inputs. The timelike/lightlike extensions still need independent numerical witnesses.

## Preserved failed NR-limit test

The first414-check study had one failed comparison: at fixed omega=.5,q20MeV, M1e5 gave a−.00664062 relative NR density discrepancy against a3e−4 gate. This result, original driver and exact original tensor source are preserved. Its source snapshot hash matches the hash recorded in the failed report.

The test did not take a controlled NR limit: fixed omega/q keeps the selected initial phase velocity finite as M grows, while the activated minimum kinetic energy and relative sensitivity to relativistic corrections grow. The repair changes the comparison path to `omega=.3 q sqrt(T/M)+q²/(2M)` while keeping the physical tensor, NR formula and acceptance gates unchanged. This shrinks the velocity with the thermal scale and passes. The old finite-M discrepancy remains true and failed; it is not relabeled as numerical error or hidden by a tolerance change.

Later finite-log input hardening changes error handling only. A separate optional-representation check ensures trace magnitudes are combined with scales before exponentiation. No physical forward formula, source coupling, support width or normalization was calibrated.

## Reproduction and remaining gates

Run `python outputs/nmb-finite-t-20261015/relativistic_tensor/validate.py`, then `compare_oracle.py`, using the existing runtime. Individual numerical runs are below1s; no build/install or production change is required. Raw complex arrays use explicit real/imag fields in JSON; the manifest pins all sources and oracle data.

Independent review is required. Next gates are a separate timelike/lightlike oracle, more systematic state/current/support coverage, connection to independently checked free density roots and independent electron/muon composition, full charged-lepton integration/tail control, inverse/crossed inventory and conserved number/energy moments. GM1 requires an explicitly reviewed q/qstar, field-energy and weak-bare-mass mapping; none is implemented here. No lepton-integrated opacity, material source, EOS external-forward gate or production claim follows from these differential results.
