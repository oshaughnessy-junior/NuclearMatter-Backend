# Independent physics review: log-domain emission-only moments

15 October 2026. Author confirmed freeze before probes. Pinned `emission_only/log_emission.py` SHA-256 **f73fd3504498b696614fb5153362f5d27816e659a8cb5bb43d4467ccbde28207**, checked before/after execution. Reviewed implementation, final 39-check result, 16-control result, research note and preserved failure contracts. Reviewer changed only new `review_physics/emission_*` files; no author/upstream files or old failures changed. Account/window matched at start (7% used).

**Accept this single prescribed-electron heavy-common-mass complete-named-antineutrino emission benchmark and its log-domain representation contract.** It supplies a successful complete two-channel static-normalization limit without the strict positive-ordinary-coefficient requirement of transfer. It **does not** retroactively pass the original different T=2 neutral-bath transfer experiment, qualify generic opacity/absorption, or provide a broad physical domain certificate.

## Inventory, occupations and normalization

The named antineutrino inventory contains positron capture on neutrons and crossed neutron beta decay. Both are evaluated at **every** positive energy node and retained through node/panel/channel log accumulation. Capture uses reversed baryons and positron chemical potential −mu_e; decay uses spontaneous n→p with final electron vacancy. These are the reviewed signs/directions. Inverse logarithms may be computed internally, but no ordinarily representable or strictly positive chi is required to integrate empty-radiation emission.

Number/energy integrands are `D E²(j_pc+j_nd)` and `D E³(j_pc+j_nd)`, with one-helicity `D=1/[2pi²(hbarc)³]`. The target neutron spin degeneracy is already in the baryon response. No additional factor two or symmetry factor is warranted. Output units fm^-3 s^-1 and MeV fm^-3 s^-1 follow directly. This source is for f_antinu=0; no radiation occupation/blocking or transport energy ledger is being evolved.

Every energy panel, including the interval between static endpoints Q−me and Q+me, is retained. The exact square-root mappings change integration variables and resolve endpoint corners without adding a physical smoothing scale. The reflected map below the decay endpoint has the correct positive absolute Jacobian. Log-sum accumulation includes both positive channel contributions even when their ordinary values underflow.

## Representation versus physical tail accuracy

`represented` uses actual IEEE exponential behavior: finite positive ordinary results, including rounded subnormals, are returned with explicit flags; zero-underflow or overflow returns `value=None` while retaining the finite logarithm. There is no rate floor, artificial zero support band, or silently discarded channel. Nonfinite log input is rejected. This is a finite-positive-T log interface, not a universally exact-zero/T=0 representation API.

At the documented gap witness the combined log is about −55933.95 while neither ordinary channel coefficient is representable. A new held-out gap point also retains finite logs with ordinary value unavailable. This validates representation/inventory bookkeeping. It does **not** independently establish relative physical or asymptotic numerical accuracy of extremely remote NR thermal tails. The note makes that limit explicit.

## Heavy-mass and prescribed-electron scope

The qualified state is T=.3 MeV, M=1e7 MeV, nn=7e-5 and np=3e-5 fm^-3, prescribed total mu_e=0. The nucleon density solves remain FD. Charge neutrality is **not** imposed; this is a normalization experiment with a supplied electron/positron reservoir, not a self-consistent neutral material state for fluid evolution.

At fixed densities and temperature, increasing common M makes the nucleon fugacities scale toward zero. Therefore the dynamical FD phase space tends to the classical static-target response at fixed **actual** density. At finite M=1e7 the small remaining recoil and Pauli corrections are legitimate model differences from the exact limiting formulas; they must not be calibrated away by replacing actual density with its classical fugacity approximation or adjusting the weak prefactor.

The static neutron-decay reference here retains finite-T electron blocking at mu_e=0. It is **not** the vacuum unblocked approximately 947.37 s bare lifetime check, nor a measured-lifetime calibration. Positron capture is significant and decay contributes about 21% of the combined number source, so this is a useful balanced two-channel test rather than a sum dominated entirely by one channel.

## Independent targeted checks

`emission_probes.py` writes `emission_probe_results.json`; **13/13 probes pass** without replaying the heavy neutrino-energy integration:

- Direct adaptive scalar **charged-lepton-energy** quadrature independently reconstructs static PC and warm-blocked ND number/energy moments, with the bare prefactor written separately from `target_moments`. Each final heavy-M moment agrees within **6.27e-7 relative**. The independently summed complete number/energy references agree within **4.14e-7 / 5.90e-7**. These are same-theory normalization/limit checks, not independent interaction physics.
- Three new spectral energies verify both-channel logarithmic summation. A held-out point inside the static gap keeps finite combined log emission with explicit ordinary-underflow flag.
- The leading FD-versus-classical density correction `n_classical/n_actual−1≈z/2^(3/2)` is independently checked for each solved species, with absolute discrepancies below **4.60e-14**. This supports the distinction between an actual FD density solve and its asymptotic classical comparator.
- The original strict-wrapper failure file's SHA remains identical to the recorded value.

Author outer order/tail refinements and sampled internal checks provide additional selected-state numerical evidence; they do not prove uniform convergence across all formal tail energies or other bath parameters.

## Gate disposition and next use

The new benchmark closes a **selected complete-named-antineutrino emission-only heavy-M normalization test** at this supplied electron bath. The earlier original T=2 neutral-state strict transfer failure remains preserved and excluded. Generic nonempty-radiation exchange, normally representable j/chi opacities, exact-zero limits, other flavors/baths and broader energy domains need their own interfaces and independent checks.

No astrophysical meaning should be assigned to the artificial heavy mass. The inherited current is leading static V0/axial-spin with quadratic phase space, not a complete recoil-current or relativistic weak calculation. No interaction, collision width, pairing, nuclear species extension or external GM1 validation is established.

The next useful step is to apply the separately reviewed log emission API to a **physical-M, charge-neutral** matched FD state and qualify its full number/energy integration before a new fluid experiment. For irradiated matter, combine gain/loss in a separately reviewed log/zero-safe collision interface and preserve composition/energy bookkeeping; this emission-only success does not supply that extension automatically.
