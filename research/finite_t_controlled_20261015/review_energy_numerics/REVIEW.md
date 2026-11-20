# Independent numerical review: full-energy charged-current moments

15 October 2026. **The dilute complete-inventory volume moments and prescribed slab exchanges pass independent numerical integration. One dominant repaired moderate-bath panel also passes.** This review does not close the failed heavy-static complete-inventory gate or claim an independent full moderate-grid rerun.

Evidence: `probe_energy.py` → `energy_review.json` (**40 passing checks**,69.52 s); `probe_moderate_panel.py` → `moderate_panel_refined_review.json` (**8 passing checks**,7.62 s). The initial supplementary coarse-panel failure is preserved separately. Total actual experiments, including that failed 4.99 s coarse attempt, took about **82.2 s**; no installs, production edits or author-file changes occurred.

## Independent integration and normalization

The main review evaluates the physical coefficient wrapper at every fresh node for the dilute T=2 MeV, `(nn,np)=(7e-5,3e-5)` fm^-3 state and both neutrino flavors. It does **not** call the author's `integrate`, `spectral`, flux or E3 helper. It uses z=log(1+E), independent energy panels, two outer orders 8 and12, and an 80-node angular integration of the actual outgoing ray occupations.

The one-helicity neutrino measure is independently reconstructed as `E²/[2pi²(hbar c)³] dE`, including conversion by 1e39 from fm^-3 to cm^-3. Empty-radiation volume emissivity is this measure times spontaneous j; energy inserts one factor E. One outward face carries `c/2 int_0^1 mu f_out dmu`, giving the usual c/4 factor for a constant hemispheric occupation. Two prescribed incoming hemispheres are counted once each.

Independent dilute number/energy emissivities and incoming/outgoing/net slab exchanges agree with archived values to **8.02e-11 relative or better**. The largest log-energy order8→12 discrepancy is **9.76e-8**. Thus the agreement is not merely the same original energy nodes summed twice. Bath total-energy exchange has the negative radiation-gain sign, and the explicit MeV→erg factor is correct. Electron-lepton exchange has opposite flavor signs; the separate flavor checks pass.

Boundary number/energy differences also reproduce the directly integrated angular ray changes with a gross flux scale. These comparisons avoid interpreting cancellation of large total fluxes as an accuracy estimate for small signed exchange. The original implementation similarly retains positive exchange pieces separately; its physically meaningful conserved quantity is their net exchange.

## Moderate-bath convergence and preserved failure

The author correctly retains the initial 14 failed coarse moderate-grid gates. Its targeted replacement panels use orders32/48 around the degenerate edge, while the untouched remainder has its own comparisons. This review independently samples the important [40,70] MeV panel at the moderate T=2 MeV, `(nn,np)=(0.002,0.001)` state, again using log-energy and direct angular rays.

The review's first order12→20 comparison failed two slab exchange gates, by **9.47e-5 and1.08e-4** against the unchanged3e-5 target, even though its order20 values already agreed closely with the author's refined reference. This is a convergence failure of this review's **coarse** grid, not a newly discovered failure of the author's repaired panel. Preserve it in `initial_moderate_panel_failure.json` and `moderate_panel_review.json`.

Refining the same panel to log-energy orders20/28 resolves all eight gates without relaxing tolerance. Independent number/energy emissivity agrees with the author's fine48 panel within **8.44e-13**; net number/energy slab exchange agrees within **3.00e-10**. The largest20→28 change is **5.12e-8**. These results support the targeted degenerate-edge treatment; they are not a uniform order requirement for every state.

The global moderate result remains dependent on the author's separately retained remainder/cutoff/internal-node evidence. This review has directly tested only the specified panel, while the complete dilute case is independently integrated across the full energy interval.

## Important interpretation of positive exchange pieces

The transfer attenuation is `(j+chi)/c`, as required by `C=j(1-f)-chi f`. The positive `slab_emission` and `slab_absorbed_boundary` pieces are a **linear-solution decomposition** under that relaxation attenuation. They are not separately integrated gross microscopic event counters `int j(1-f)` and `int chi f` through the slab. In particular, the deficit of externally incident radiation relative to free transmission includes both true absorption and blocking of spontaneous emission. Their difference gives the correct net conserved exchange, which this review accepts. Do not identify the second piece with an event count using chi alone or infer material heating from it without the net ledger.

## Open gates and reproducibility caveat

- **Heavy-static complete inventory remains excluded.** The preserved failure at E=0.8313293913 MeV has crossed log j≈−8402 and capture log j≈−1.87e6. Exponentiating those finite-T tails underflows float64, and the wrapper deliberately raises. A selected-energy static test cannot replace this unresolved energy-integrated gate. A log-domain or carefully absolute-error-controlled interface needs separate review; no missing channel, zero floor or rate fit is introduced here.
- Sampled internal energy/angle/tail checks and high-energy cutoffs are strong selected-state evidence, not a global convergence certificate. An evolving fluid also needs consistent FD baryon/rest-offset energy derivatives and actual radiation evolution; the fixed-bath ledger does not supply these automatically.
- **Explicit resume is not automatically provenance-safe.** Source inspection finds that `run_study.py --resume` selects cached runs using flavor and outer_order only; it does not enforce a fingerprint of state, upstream source, internal controls and energy edges. Existing frozen resumed quadratures have been checked numerically here, and their known reporting-only interruption does not invalidate them. However, the current CLI must not be treated as guaranteeing fresh-source/state equivalence for arbitrary future resumes. A manifest validation should reject mismatches before reusing cached runs. Fresh reproduction is the default and remains the qualified workflow.

The source/dependency snapshots are recorded in this review's JSON and manifest. No full physical external-code comparison, interacting EOS, relativistic current, scattering/redistribution or hydro claim is added.
