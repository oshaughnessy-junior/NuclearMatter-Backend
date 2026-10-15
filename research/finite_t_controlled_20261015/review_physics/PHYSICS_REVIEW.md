# Independent physics review: initial finite-temperature slices

15 October 2026. Reviewer was not an author of either reference implementation. This audit owns only `review_physics/`; no author source was changed. Scope: source/equation audit plus 23 selected independent probes, all passing. This is acceptance of restricted reference calculations and a coupling recommendation, not qualification for neutron-star matter or arbitrary finite-temperature Urca physics.

## Disposition

**Accept** the pair-equilibrated ideal electron/muon thermodynamics as a CPU reference on the demonstrated domain. **Accept** the static classical free-nucleon electron charged-current kernel, with its finite-mass lepton phase space, as an internally consistent restricted collision model. **Accept** a transparent electron-only fluid element as the next integration target. The supplied controlled-model energy and lepton ledgers have the right signs and variables.

**Require a correction to the prospective closed-box acceptance contract:** these static charged currents cannot thermalize every antineutrino energy. This is an exact invariant of the specified reaction operator, not an integration defect. Conservation tests can proceed immediately, but a claim of relaxation from arbitrary spectra to a global Fermi–Dirac spectrum must wait for either an appropriate redistribution process or a qualification of the disconnected modes.

## Findings

1. **Finite-mass lepton responses are correct.** With net charge fixed and pairs equilibrated, the response moments imply `n_mu=A0/T`, `n_T=A1/T²`, `mu_T|n=−A1/(T A0)` and `e_T|n=(A2−A1²/A0)/T²`. The latter follows by eliminating `dmu` between the number and energy derivatives; rest-energy and chemical-energy terms cancel. Particle/antiparticle signs and spin degeneracy two are consistent. The thermal increment subtracts `mu0` times the fixed-charge constraint correctly, including at net zero. Heat capacity and entropy in the chosen `T=kBT` convention have numerical units fm^-3; their physical cgs conversion needs the declared Boltzmann factor.

2. **The weak normalization and inverse channels are consistent with the declared model.** `A` has units s^-1 MeV^-5; `D E² dE` is the one-helicity neutrino number measure. Hence `(A/D) n_target E_e p_e` has units s^-1 and reconstructs the target moments without a spin factor of two. Electron capture has `E_e=E_nu+Q`; positron capture has `E_e=E_antinu−Q`; neutron decay has `E_e=Q−E_antinu`. The final-state vacancies and initial occupations are assigned correctly. Inverse neutron decay includes an occupied electron and incoming antineutrino; the implementation does not confuse it with positron absorption.

3. **Detailed balance is internally consistent.** For shared nucleon thermal wavelength, `mu_n−mu_p=Q+T ln(n_n/n_p)`. Each channel yields `ln(j/chi)=−(E−mu_nu)/T`, with `mu_nue=mu_e−Q+T ln(n_p/n_n)` and the opposite antineutrino chemical potential. The lepton chemical potentials include rest mass. Do not substitute the commonly tabulated rest-subtracted electron degeneracy parameter in these equations. The ratio is undefined when both coefficients vanish; test the collision residual there instead.

4. **Important disconnected antineutrino interval.** Positron capture and its inverse act only for `E_antinu >= Q+m_e`; neutron decay and its inverse act only for `0 <= E_antinu <= Q−m_e`. Thus `Q−m_e < E_antinu < Q+m_e`, numerically **0.78233340931–1.80433131069 MeV**, has `j=chi=0` identically in this model. Every occupation in this interval is a conserved mode in a closed box. In a fixed slab these modes stream without these charged-current interactions. They must not be filled by interpolation across the thresholds. A global equilibrium root using only total energy and total electron lepton number would omit these constraints. It is valid to initialize these bins at equilibrium and test the connected modes, or to hold them fixed and solve the constrained equilibrium of the active bins. It is invalid to expect arbitrary initial occupations there to relax. Elastic isoenergetic scattering alone does not remove this energy-space obstruction; an energy-changing or appropriate pair process can.

5. **Shared-mass classical nucleons plus finite-mass leptons form a coherent leading static ansatz.** Use one kinetic mass `M` and spin degeneracy two in both nucleon thermal wavelengths, while retaining different neutron/proton rest energies. The common thermal chemical term cancels in the reaction affinity. The common `3T/2` kinetic energy per converted baryon also cancels in the composition energy derivative. This makes the static-Q collision model and the stated EOS thermodynamically compatible at their declared order. It is not a finite-recoil model merely because the matter EOS has a finite thermal mass. Thermal Doppler/recoil corrections are of omitted order in `T/M` and lepton energies over `M`; finite degeneracy and interactions are independent limitations. Report `n_i lambda_M^3/2` in run manifests to establish whether the classical approximation is actually appropriate. No specific dense-matter validity follows from passing these checks.

6. **Transparent rate balance is not thermal equilibrium.** The source sum, rather than `mu_nue=0`, defines transparent kinetic balance. Even at that root, gross energy emission remains. An isothermal composition experiment requires an explicit thermostat ledger; a cooling element must evolve temperature with composition. Existing cooling/paired dense-matter physics must remain separately qualified.

## Recommended first coupling

Use independent variables `(nB,T,Y)` for electron-only free nucleons, with `n_p=Y nB`, `n_n=(1−Y)nB`, and `n_e,net=Y nB`. Photons can be an explicitly switchable equilibrium sector. Preserve pair equilibrium in the lepton calculation. Define total matter energy per baryon, subtracting a constant reference neutron rest mass if desired,

```
u = −Q Y + 3T/2 + e_lepton(T,Y nB)/nB + e_gamma(T)/nB
P = nB T + P_lepton + P_gamma
u_T = 3/2 + Cv_lepton/nB + (de_gamma/dT)/nB
u_Y = −Q + mu_e − T (dmu_e/dT)|n_e
```

The last relation follows from `de/dn|T=mu−T dmu/dT|n`; replacing it by `mu` or by just `−Q` is incorrect. The supplied lepton state already returns every quantity needed. The independent finite-difference probe reproduced this composition derivative.

At prescribed density evolution,

```
dot(Y) = −(R_nue−R_antinue)/nB
u_T dot(T) = [P/nB²−u_nB] dot(nB) − u_Y dot(Y) − Q_nu/nB + Q_ext/nB.
```

Use the **same discrete spectral collision moments** for composition, radiation energy and external escape ledgers. Include the rest-energy offset in `u_Y` exactly once; do not add a separate reaction-Q heat source. At fixed density the energy law is simply `dot(u)=−Q_nu/nB+Q_ext/nB`. A thermally evolving element should first test this against an independent derivative of the full `u(T,Y)` function, including the composition term.

For the first closed box, evolve matter `u,Y` and neutrino occupations, recovering `T` monotonically from `u` with positive `u_T`. Share discrete weights between all exchange ledgers. This reduces cancellation from integrating separate matter/radiation sources, but the antineutrino disconnected modes still need the explicit constrained treatment above. A pointwise entropy-production probe verifies `C[ln((1−f)/f)−(E−mu_nu)/T]>=0` for selected states. It checks local microscopic reversibility, not a complete integrated box H theorem.

## Independent probes and limits of this review

`python3 outputs/nmb-finite-t-20261015/review_physics/independent_probes.py` writes `probe_results.json`. All **23 probes passed**: direct momentum-space number/energy integrals for positive/negative electron charge and massive muons; channel-resolved logarithmic detailed balance; selected pointwise entropy production; exact disconnected support; and the total matter composition-energy derivative. Momentum quadratures independently change variables from the author's kinetic-energy integration but share physical constants and ideal-gas theory. These are independent numerical/identity checks, not an independent many-body theory comparison. Author test suites were inspected as context but not rerun here.

No numerical domain sweep, EOS interaction validation, muon weak rate, nucleon Pauli blocking, recoil/mean-field Urca, pairing, or transport scattering was established. In particular the antineutrino gap is a feature of this leading static approximation and must not be exported as a physical statement about realistic finite-recoil nuclear matter.

The primary equation source was inspected: [Chiesa et al., Appendix A, beta processes](https://arxiv.org/html/2412.04570v2#A1). That source supplies the wider reaction context, while the concrete gap and coupling conclusions here are derived directly from the audited restricted code. No source implementation was installed or executed.
