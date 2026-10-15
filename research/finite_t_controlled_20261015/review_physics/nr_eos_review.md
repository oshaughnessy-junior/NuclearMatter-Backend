# Independent physics review: matched free-FD nucleon EOS

15 October 2026. Reviewed `nr_baryon_eos/free_nucleons.py`, technical note and recorded 96-check evidence. No author files changed. Review budget/account/window matched at start (4% used).

**Accept this positive-density finite-T common-mass free-FD baryon EOS and its supplied `matter_state` response/optical factories as a matched reference.** No thermodynamic or chemical-convention blocker was found. Its purpose is to replace classical baryon thermodynamics when Pauli-blocked finite-recoil responses are used. It is not an interacting or relativistic matter EOS and does not establish full composition equilibrium with nuclei.

## Density measure, offsets and responses

Starting from spin-two `n=integral p²dp f/[pi²(hbarc)³]` and `k=p²/(2M)`, the energy-space measure is `sqrt(2) M^(3/2) sqrt(k) dk/[pi²(hbarc)³]`, as coded. The pressure virial relation `P=2K/3` applies to this nonrelativistic quadratic dispersion. Total energy includes constant offset `O*n`, and total chemical potential is `mu=O+nu`; both are passed consistently into the reviewed dynamical response.

The factory's default `(O_n,O_p)=(Q,0)` subtracts a constant common rest energy without losing reaction composition energy. It does not give neutron/proton different kinetic masses. No extra Q source is warranted when this energy is evolved. If offsets become density- or temperature-dependent, the present derivatives and free-energy construction cease to include their rearrangement terms: such inputs are outside this constant-offset model.

The positive moment matrix gives `n_mu=A0/T`, `n_T=A1/T²`, `mu_T|n=−A1/(T A0)` and `Cv|n=(A2−A1²/A0)/T²`. The formulas are correct at fixed **total mu** because O is constant. Constant offsets cancel from susceptibility and Cv. The total-energy density derivative is `A_i=mu_i−T mu_i,T|n_i`; using only the kinetic chemical potential would lose offset/composition energy.

`matter_state` solves neutron/proton densities independently and imposes lepton net charge np through the already reviewed finite-mass pair EOS. No beta equilibrium is imposed. Its PairState/OpticalState signs are correct: `mu_nue=mu_p+mu_e−mu_n`, with the opposite antineutrino value. Photons are deliberately not included by this factory.

The public dataclasses can technically be assembled manually with different species temperatures or masses; the supplied factory guarantees matching. A future production interface should validate common T/M and lepton neutrality, or keep construction behind the factory. Acceptance here covers factory-generated states, not arbitrary inconsistent manual dataclass combinations.

## Independent shared-EOS derivative checks

`crossed_eos_probes.py` writes `nr_eos_probe_results.json`. At `T=2 MeV, nB=1e-4 fm^-3, Ye=0.3`, independently differentiating the complete returned baryon-plus-lepton free energy and energy gives:

- `(partial f_bar/partial Ye)_T,nB = mu_p+mu_e−mu_n`, agreeing within **7.46e-9 scaled**. This connects the EOS thermodynamic affinity to the weak factory rather than simply checking factory signs.
- `(partial u/partial Ye)_T,nB = A_p−A_n+A_e`, agreeing within **3.95e-9 scaled**.
- Frozen-composition `u_T=(Cv_n+Cv_p+Cv_e)/nB`, agreeing within **3.67e-10 scaled**.

All three pass. The same factory's finite-M capture/crossed chemical KMS checks also pass as recorded in `crossed_probe_results.json`. The author's broader 96 checks provide momentum/energy integration, Euler, Maxwell, free-energy pressure, cold/MB and derivative evidence on selected states. Shared theory and constants mean these remain free-gas numerical/identity qualifications, not independent interacting physics.

## Coupling formulas required before finite-M fluid use

For baryon density n, neutron fraction 1−Y, proton/electron fraction Y, constant M/O, and optional local photons, define

```
u = [e_n+e_p+e_e+aT^4]/n
P = P_n+P_p+P_e+aT^4/3
u_T = [Cv_n+Cv_p+Cv_e+4aT³]/n
u_Y = A_p−A_n+A_e
u_n = [n_n A_n+n_p A_p+n_e,net A_e−e_n−e_p−e_e−aT^4]/n².
```

Use these derivatives in the already reviewed first law. They reduce to the classical formulas only in the appropriate dilute limit. Simply replacing chemical potentials inside weak rates while retaining a `3T/2` baryon energy/heat capacity is not thermodynamically matched when degeneracy matters. The neutrino chemical parameter is the free-energy composition derivative; it is not generally the total-energy derivative `u_Y`.

## Domain and next gate

The reported pF/M and T/M diagnose the quadratic-dispersion expansion; finite-T tails additionally probe momenta above pF. Arbitrary degeneracy is not arbitrary relativity. Exact vacuum, susceptibility underflow and very cold floating-point cancellation are explicitly outside the finite-T reference; its separate cold-state formulas do not validate numerical interpolation across all intermediate states.

Before a finite-M fluid demonstration: derive/check the full mixture energy/density/composition derivatives above; establish optical energy/angle/neutrino-moment convergence including the known imported capture electron-threshold corner; then test one weak-off adiabatic path and one weak-on transparent path with the same complete named channel inventory. Record NR momentum/temperature diagnostics and keep photon-retention/neutrino-escape closures explicit. Compare free relativistic thermodynamics at representative states when an actual physical error estimate is needed. No interaction uncertainty or neutron-star density reliability follows from the current numerical residuals.
