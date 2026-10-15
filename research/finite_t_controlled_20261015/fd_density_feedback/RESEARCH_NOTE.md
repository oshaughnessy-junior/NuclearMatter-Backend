# Prescribed-density finite-Fermi-gas element with actual weak feedback

This research leaf couples the reviewed free NR Fermi n,p and relativistic e± EOS to the complete allowed-current charged-current emission inventory. It studies a transparent Lagrangian fluid element with prescribed density, rather than solving hydrodynamics. Photons are absent. Bare nucleon mass splitting, electron rest energy, and composition changes are included. The nuclear EOS is the matched free gas, not GM1. Current-recoil, weak magnetism, correlations, nuclear interactions, nuclei, muons, and radiation feedback are outside this ansatz.

For baryon density n, proton/electron fraction Y, and energy per baryon u(T,n,Y), use emitted one-helicity neutrino number rates Rν,Ranti [fm⁻³ s⁻¹] and energy rates Qν,Qanti [MeV fm⁻³ s⁻¹]. The inventory is electron capture plus its inverse for νe, and positron capture and crossed neutron decay plus their inverses for antiνe. Empty incident radiation sets the collision source to j. The physical evolution equations are

    F = dY/dt = (Ranti − Rν)/n,
    q = (Qν + Qanti)/n,
    du/dt = P/n² dn/dt − q,
    dT/dt = [(P/n² − u_n) dn/dt − u_Y F − q]/u_T.

The derivatives are those of the full FD EOS. In particular u_Y = A_p−A_n+A_e with A_i=µ_i−T(∂µ_i/∂T)_ni, and n²u_n=P−T P_T. No classical heat capacity is substituted. With a=µ_p+µ_e−µ_n and material entropy s per baryon, T ds/dt=−q−aF. Escaping total energy does not require falling temperature: composition energy release can heat this state.

The prescribed cycle is n(t)=10⁻⁴ exp[0.02 sin²(πt/τ)] fm⁻³ with τ=2×10⁻⁶ s, initially T=2 MeV and Y=0.3. Density rises 2.02% and returns. The integration is first-order operator splitting: a genuine changing-state full spectral source supplies a weak Euler kick in (u,Y) at fixed n; the full FD energy is inverted for T; then density changes along an exact material-entropy adiabat at the updated composition. Two and four steps are separate bounded physical experiments. This is a short-path step-sensitivity study, not a convergence-order or long-time certificate.

The mechanical counter is independently evaluated as ∫P/n dln n using Gauss nodes and full entropy-root EOS solves. It is compared with the EOS energy change, then used in u+Eescape−W=u_initial. The electron-lepton counter is Lescape=∫(Rν−Ranti)/n dt, giving Y+Lescape=Y_initial. A separate Gauss integral of (du−a dY)/T along each finite weak kick checks material entropy; the density substep preserves it. Material entropy alone is not total matter-plus-radiation entropy, and no full radiation entropy statement is made.

Physical kernels use internal order48/tail40 and segmented neutrino-energy Gauss order6 to100 MeV, with the exact threshold panels inherited from the reviewed isochoric leaf. Endpoint order10, selected peak-density internal72/tail50 coefficients, and order8 mechanical work are separate qualification refinements. These are selected-state checks; the inherited strict ordinary-coefficient underflow limitation and the excluded heavy-static T2 wrapper comparison remain open. No broader domain claim follows.

Each source record carries exact solved FD state, all energy edges/internal controls, full source and integrator hashes, and Python/NumPy/SciPy environment. Only the identical initial state reuses the previously qualified source, through its full current qualification contract. Every distinct evolving state recomputes both flavors with guarded mapped capture and explicit crossed decay. Source metadata and provenance are checked before use and again at the end. There is no fixed-rate feedback approximation or learned surrogate.

## Reproduction and results

Run `python run_feedback.py 2`, then `python run_feedback.py 4`, then `python qualify_feedback.py` from this directory (or use absolute script paths). Each process has its own bounded numerical experiment and writes raw partial checkpoints. Final numerical results and limitations are appended below after qualification. Frozen upstream sources are unchanged.

| Quantity | Two steps | Four steps |
|---|---:|---:|
| Final T [MeV] | 2.0044132484 | 2.00441327155 |
| Final Y | 0.294771019617 | 0.294800226214 |
| Final u [MeV/baryon] | 8.41416497485 | 8.41469770246 |
| Escaped energy [MeV/baryon] | 0.0880517355516 | 0.087515715481 |
| Escaped electron lepton number /baryon | 0.00522898038255 | 0.00519977378596 |
| Net pressure work [MeV/baryon] | 0.000295446839972 | 0.000292154380203 |
| Material entropy change [kB/baryon] | -0.00416716927967 | -0.00412143259763 |

All **126 author checks pass**: 19 two-step, 37 four-step, 41 postqualification and 29 new-state EOS/tail checks. Physical runs took 49.81 and 110.23 seconds; separate qualification experiments took 87.69 and 18.57 seconds. No numerical failure or budget stop occurred in this leaf. Independent review is still required.

At four steps the independently computed compression and expansion works are +0.0708986619446 and -0.0706065075644 MeV/baryon. Their small net difference is retained explicitly, rather than inferred from an energy subtraction. The maximum energy-ledger defect is 2.13e-14 MeV/baryon, lepton defect 5.55e-17, and material entropy-ledger defect 1.06e-15 kB/baryon. Work Gauss4→8 changes at most 3.87e-14 MeV/baryon. Small conservation defects do not measure physical trajectory accuracy.

Endpoint outer-energy order6→10 changes individual number/energy moments at most 8.36e-07 relative. Selected peak-density internal48/tail40→72/tail50 coefficient changes are at most 1.78e-15 relative. Independent 100–240 MeV extension panels at both actual peak and endpoint contribute at most 3.57e-15 of the respective same-flavor production moment. This is a numerical finite-cutoff check, not a mathematical bound for E>240. Genuine u_T,u_n,u_Y central-difference ladders at both states agree to 1.59e-09 relative.

Two→four steps change escaped energy by 0.612% and escaped electron-lepton number by 0.562%. These differences are substantially larger than the selected source-quadrature errors. The near coincidence of final temperatures must not be presented as uniform temporal convergence. Higher-order splitting, more steps and longer paths remain unqualified. Every changed state was recomputed physically; none of these feedback results use a fitted or frozen rate.

`results_2.json` and `results_4.json` retain all actual-state source requests, full spectra integrated panel sums, source origins, state/hash/environment fingerprints, and every step ledger. `qualification.json` retains endpoint and internal-kernel refinements; `tails_eos.json` retains derivative and cutoff probes. `MANIFEST.json` pins all leaf files and transitive physical/integrator source hashes. Reproduction includes an exact-provenance gate; future changes to upstream physics or the physical integrator require fresh qualification.
