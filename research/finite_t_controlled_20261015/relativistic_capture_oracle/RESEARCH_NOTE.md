# Moving-target CM event oracle for total relativistic capture

15 October2026. **Selected free-baryon total capture/inverse reference, independently ordered from the tensor/lepton integral.** This leaf uses target momentum, its angle to the neutrino, outgoing two-body CM polar angle and outgoing CM azimuth. It does not call or import the new response/tensor or lepton-integration implementation. Only bath/test-point inputs were coordinated. Physical masses are mn939.56542052,mp938.27208816,mmu105.6583755MeV, with bare Mweak938.91875434MeV and the reviewed source-defined full Guo current.

## Event phase space and absolute factors

The bath is isotropic, with particle-only baryon f_a(Ea) per spin and total chemical potentials including rest mass. Fix the incident neutrino along z and the initial target in the xz plane. Integrate target direction azimuth2pi analytically. For each target momentum and incident-target cosine c, construct s=ma²+2Eν(Ea−pa c), the exact center-of-momentum boost beta=(k+pa)_vec/(k+pa)_0, and the exact final two-body CM momentum. CM angles are defined relative to the **boosted incident neutrino**, not the bath z axis. Boost the outgoing charged lepton and baryon separately back to the bath, then evaluate actual FD blocking at their bath-frame energies. No final-lepton-rest or static-target replacement is made.

The target spin-averaged vacuum cross section is

`dσ/dΩCM = Mavg²/(64pi²s) * pf/pi`.

Own batched external u/v spinors and gamma/current matrices construct the full amplitude, with Mavg²=GF²Vud²/4 times the full summed current-product square: the1/2 target-spin average is explicit and there is no incident-neutrino average. The spin-two target measure and Møller flux then give

`chi_s = ∫p²dp dc/(2pi² hbar) [1−pa c/Ea] ∫dΩCM (dσ/dΩCM) fa(1−fb)(1−fl)`.

All momenta/densities here are natural MeV units; the displayed particle-density diagnostic divides by(hbarc)³. Thus no fitted normalization or extra c conversion enters the rate. The Møller flux is dimensionless; sigma is MeV^-2, the target measure MeV³, and chi has s^-1 after division by hbar.

For inverse emission **the same common invariant event measure** uses independently evaluated `fb fl(1−fa)`. It does not compute j from chi or a KMS ratio. Log occupations are combined with the positive event weights and accumulated with logsumexp, preserving dilute logj when ordinary j underflows. The neutrino occupation/vacancy factors belong outside these intrinsic coefficients. No support floor or physical broadening is introduced.

The full vector/axial/F2/GP vacuum vertex and q²-dependent source poles are reconstructed in `events.py`, without importing either new tensor author code or prior full-current contractions. Antineutrino v-spinor/conjugate-current algebra is implemented and independently checked at vacuum points, but the **total thermal event integral is only qualified for the listed neutrino points**. This is capture/inverse capture, not complete antineutrino decay inventory or total weak sources.

## Shared physical EOS inputs and selected values

Both tests use Eν150MeV, outgoing negative muon, physical n→p masses and the full source current. Bath chemical potentials came from the reviewed relativistic free EOS; no chemical equilibrium is imposed.

| bath | T | nB [fm^-3] | Ye,Ymu | mu_n | mu_p | mu_mu |
|---|---:|---:|---|---:|---:|---:|
| dilute |.2|1e−15|.3,0|934.5991075502667|933.136728749228|0|
| warm |2|1e−4|.29,.01|933.6807508996585|930.6761152463529|97.74668143468162|

The initial target densities are7.000000000001131e−16 and6.999999999999623e−5fm^-3. The warm state has separate net electron/muon fractions and Yp=.30; the dilute zero-net muon still has its pair-equilibrium chemical potential0.

| selected coefficient | dilute | warm |
|---|---:|---:|
| chi [s^-1] |2.3663596769113936e−5|2402841.5288913795|
| log chi |−10.651572691985756|14.692162565349395|
| j [s^-1] |unrepresented|2.407757975788224e−6|
| log j |−767.9634666971789|−12.936814543962601|

Finite dilute logj is retained, with ordinary value=None and representable=False. That is not physical zero. KMS is checked only **after** independent integration of both positive event inventories. The fine warm logj−logchi agrees with its thermodynamic exponent within2.14e−14. This relation tests occupation bookkeeping, not integration accuracy.

## Numerical gates and failures

Current evidence has **61 passing checks and2 retained coarse-grid failures**: main run12PASS/2FAIL, independent spinor/boost24PASS, dilute absolute anchor2PASS, targeted warm refinement4PASS, controls/representation19PASS. `results.json.passed` remains false because those coarse failures are retained. The initial pre-map run additionally retains10 failed gates; it is not part of the selected qualification.

Initial direct kinetic-energy Gauss quadrature undersampled the square-root lower endpoint: densities were wrong by~3.8e−5 and total-rate refinement failed. That source and all initial results remain preserved. The numerical substitution k=x², with exact2x Jacobian, resolves this endpoint without changing phase space or matrix physics. At mapped radial/incident/CMpolar/azimuth orders6/6/10/6,10/10/16/8,14/14/24/12 and tails30/45/60T:

- Dilute density and rate converge to essentially roundoff. The coarsest rate differs8.8e−10, below the unchanged3e−6 rate gate.
- Warm coarse6 differs4.24666e−6 and **fails** the same3e−6 gate. Medium10→14 differs2.81e−7 and passes.
- A separate warm20/20/32/16,tail80 refinement changes14's chi/j by2.09e−10. Its density residual is1.55e−15. No tolerance was relaxed.

The fine experiment finished its final panel in19.64s, but its subsequent JSON numpy-bool serializer failed. Every panel had already checkpointed positive accumulated sums; `recover_refinement.py` asserts that the retained checkpoint is the final panel4of5 and rebuilds only those completed rate/density comparisons. It does not invent missing fine shell-residual/event-count metadata. Original driver and completed checkpoint are retained; the serializer is corrected for future reproductions. An analogous cheap anchor reporting failure is preserved. All experiments were below120s, with roughly49s total physical numerical work including failed initial attempts.

Absolute normalization has two checks. First,24 own-spinor/source-invariant and Lorentz-boost checks pass at four new physical vacuum points (neutrino/antineutrino); arbitrary boosts leave squared amplitudes invariant. Second, a separate dilute collision-flux integral averages the **target-rest total vacuum cross section**, using the reviewed published scalar/source polynomial rather than this leaf's explicit spinor/event implementation. It agrees with the event chi within4.9e−15 relative. The finite-T dilute rate is.1801887% above the stationary-target cσ rate; that is target motion, not a fitted normalization correction. Final blocked momenta conserve four-momentum to the retained main-run gate2e−14.

## Limits and reproduction

Run `python run.py`, `python check_spinors.py`, `python dilute_anchor.py`, `python refine.py` and `python check_controls.py`. The main run deliberately reproduces its coarse warm failures. `recover_refinement.py` reproduces the saved-data recovery without rerunning the physical refinement. Full controls and bath metadata are in raw JSON. Parent/independent reviewers should compare frozen final rates with the separately authored tensor/lepton integral; neither integration path was tuned to the other's result.

This is a two-bath, one-incident-energy reference, not a generic integration-domain certificate. Simultaneous order/tail increases establish selected convergence but do not furnish a rigorous infinite-tail bound. Near-threshold moving-target support, stronger degeneracy/blocking, total antineutrino capture, electron flavor totals, crossed decay, broad energy moments, interacting/GM1 mappings and radiation/matter evolution remain separate gates. Finite log outputs do not establish arbitrary extremely suppressed-tail accuracy. The EOS is free particle-baryon plus pair leptons; baryon antiparticles, radiative corrections and medium currents are omitted. See [Guo et al., current and invariant-rate conventions](https://arxiv.org/html/2006.12051v2) and [Roberts & Reddy, spin/phase-space conventions](https://arxiv.org/html/1612.02764v2); no external finite-temperature interacting benchmark is claimed.
