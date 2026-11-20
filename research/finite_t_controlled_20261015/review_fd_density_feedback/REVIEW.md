# Independent physics review: prescribed-density FD weak feedback

15 October 2026. **Accept the selected2/4-step transparent density-feedback experiments within their stated free-FD/allowed-current model.** No physical blocker found. This is a conservative short-path operator-splitting demonstration, not a general temporal-accuracy, hydro or interacting-EOS qualification. All author sources/evidence were preserved.

Reviewed `engine.py`, `run_feedback.py`, source qualification/fingerprint logic, author raw trajectories and independent work/source/EOS refinements. New review-owned `probe.py` and `results.json` supply **61/61 checks passing in4.72s** under the author's recorded existing Python environment. We did not rerun either49.8s or110.2s trajectory.

## Actual physics and first-law signs

The source is the named transparent electron-family inventory: electron capture/inverse for nue, and positron capture plus crossed neutron decay/inverses for antinue. Empty radiation selects spontaneous j. The energy moment has one more E than the number moment, with one-helicity neutrino measure `E²dE/[2pi²(hbar c)³]`; j is s^-1. No extra c, target-spin degeneracy or anti-flavor multiplicity belongs in these integrated sources.

Positive emitted nue number reduces material Ye; emitted antinue raises it. Therefore `F=(Ranti−Rnu)/n`, and escaped electron-lepton source is−F. Both positive radiation energy moments remove material total energy, giving `du/dt=P dn/dt/n²−q`. At fixed baryon number, prescribed n changes the element volume, so compression work has positive sign. The two distinct signed work legs must be retained; the small net cyclic work is not necessarily zero after weak processes change composition/entropy.

The EOS subtracts a common constant nucleon rest contribution and retains neutron Q, electron rest energy and e± pair energy. The neutron/proton common939MeV kinetic mass is the declared NR ansatz, not exact relativistic bare-mass recoil. The finite Q offset is fixed across all density/composition states; it is not recalibrated to impose beta balance. No hidden density-dependent offset or rearrangement work appears. Photons are absent.

For `u(T,n,Y)`, correct full-FD derivatives give

`uT dT/dt=(P/n²−u_n)dn/dt−uY F−q`,
`uY=Ap−An+Ae`, `Ai=mu_i−T mu_i,T`, `n²u_n=P−T P_T`.

The code obtains temperature by full EOS inversion after the weak energy/composition increment, so it retains the same composition/rest-energy terms without adding Q a second time. The entropy identity uses the chemical affinity itself, `T ds/dt=−q−(mu_p+mu_e−mu_n)F`, rather than replacing it by uY. This sign/convention distinction is correct.

## Operator split and mechanical path

The first substep is an explicit weak Euler kick at fixed density, with q,F freshly calculated at its initial material state. The density move follows at fixed updated composition and full material entropy. This is a first-order split; neither the exact EOS inversion nor the entropy-preserving mechanical substep makes the weak source constant along the true continuous trajectory.

The independent pressure integral follows the same reversible mechanical path, evaluating `∫P/n dln n`. The entropy integral follows the **actual straight finite-kick path in u,Y**, evaluating `∫(du−affinity dY)/T` with full EOS inversions. It is a legitimate first-law check on that finite path, not an independent solution of continuously changing weak rates within the step. Summing this material entropy change does not furnish the escaping neutrino entropy or an H theorem for matter+radiation; the author note correctly makes no such claim.

The selected2/4-step density nodes split the sinusoidal compression and expansion into monotonic segments. Their endpoint-adiabatic construction represents the declared reversible mechanical substep. No mechanical momentum equation, gravity or microscopic equilibration-timescale certificate follows from the microsecond time label.

## New independent witnesses

1. **All raw source panels and step ledgers.** Independently sum number/energy panel moments for each flavor, reconstruct q,F and accumulate escaped E,Le and pressure work at all six recorded steps. These reproduce the source signs and conservation ledgers; largest reconstructed energy defect is2.13e−14MeV/baryon. Requests match each actual pre-kick T,n,Ye, and their fingerprints are reconstructed independently.
2. **A selected compression path with different quadrature/root methods.** For the second four-step mechanical substep, use a bracketed entropy root at midpoint/end and Simpson integration in log density, rather than the author's Newton/Gauss implementation. Independent pressure work is **.0355029168711335MeV/baryon**, versus author Gauss work **.0355029168711141**: about5.48e−13 relative. Reconstructed endpoint temperature is2.02689531366417MeV; entropy remains6.01754684834392 per baryon. Full EOS energy increase agrees with this independent pressure integral.
3. **Fresh physical coefficients at a new spectral energy.** At the actual changed peak-density state, evaluate both flavors at **E=7.3MeV**, internal48/tail40 versus64/tail50. Separate j,chi pass the original3e−6 refinement gate and EOS-affinity KMS passes3e−10. This supplements the selected node qualification; it does not certify a full density/energy domain.
4. **Exact state and provenance.** Recompute the peak full FD state and compare its complete serialized species state to the original request. It matches. Source/integrator hashes and recorded runtime environment also match under the successful replay.

## Provenance rejection preserved

The first review run used existing `junior_tools` Python3.12/NumPy2.3.5/SciPy1.15.2; all60 physical/reconstruction checks passed but the exact provenance check failed. The author environment is existing base Python3.13/NumPy2.4.6/SciPy1.17.1, with a different platform string and executable path. All physical-source/integrator hashes matched; this was an environment mismatch, not a changed source or an author physics failure.

`initial_environment_mismatch.json` preserves the rejected review. Replaying the same review driver with `/Users/rossma/miniconda3/bin/python` gives61/61PASS, with no weakened fingerprint or scientific tolerance. No dependency was installed. This demonstrates why exact initial-source reuse must remain gated by the full qualification contract, rather than just T,n,Ye.

The author implementation allows initial reuse only for exact T2,n1e−4,Y.3 and original controls, and verifies the earlier full-state/source/environment qualification. All distinct later states take the fresh full-source branch. The code and raw origin/request records support that distinction. The review does not treat old-state or wrong-environment rates as qualified changed-state physics.

## Accuracy disposition

**The two→four-step escaped-energy difference is0.612484%, and escaped electron-lepton difference is0.561690%.** The final T difference is only2.31458e−8MeV. The near-identical temperatures therefore conceal appreciable sensitivity in the emitted energy/composition evolution; they cannot establish uniform temporal convergence. Those source-integrated differences are much larger than selected quadrature errors and must remain explicit in paper-bound evidence.

Accept physical source feedback, independent work, EOS/rest-energy signs, conservation and the short controlled split demonstration. More steps/higher-order splitting, longer-time behavior, continuum cutoff/error bounds, radiation feedback and relativistic/interacting/correlated matter remain open. No author edits, production changes, gate relaxation or discarded author failure occurred.
