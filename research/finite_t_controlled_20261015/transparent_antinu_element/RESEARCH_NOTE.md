# Transparent antineutrino material element: two specified reactions

Research prototype, 2026-10-15. This is a short fixed-density material feedback experiment using **only** neutron beta decay `n → p + e− + antiνe` and positron capture `n + e+ → p + antiνe`. Escaping radiation has identically zero incoming occupation at every time. Both reactions increase net electron fraction. There is no electron capture, neutrino channel, muon, radiation evolution, or claim of full beta equilibrium. The empty-radiation prescription is imposed externally.

## Material and source equations

Use exact free relativistic particle neutron/proton FD states, physical masses, and the exact electron/positron pair EOS with neutrality `np = ne− − ne+ = nB Ye`. Pair chemical potentials are ±µe. The particle-only baryon EOS retains its explicit antinucleon diagnostic and exclusion. Here nB=10⁻⁷ fm⁻³, initially T=2 MeV and Ye=0.3. There are no photons, ions beyond free nucleons, mean fields, or extra rest-mass Q adjustments.

Let ε include the actual neutron/proton and both charged-lepton rest energies and u=ε/nB. Compute each channel's total actual moving-particle phase integral R (fm⁻³ s⁻¹) and escaping energy Q (MeV fm⁻³ s⁻¹), then

```
dYe/dt = (Rbeta + RPC)/nB
 du/dt = −(Qbeta + QPC)/nB
 uT = Cv,nB,Ye/nB
 uY = (µp − T∂Tµp|np) − (µn − T∂Tµn|nn)
      + (µe − T∂Tµe|net ne)
 dT/dt = [−Q/nB − uY R/nB]/uT .
```

The last equation diagnoses the genuine FD composition/temperature response; integration updates conserved u and Ye, then solves the actual EOS for T. No classical Cv is substituted. Explicit Euler source updates give a deliberately modest, measurable first-order time study. The energy root is bracketed in T=[0.5,4] MeV; this is a numerical bracket, not an accuracy certificate throughout that interval. The FD material ansatz assumes rapid rethermalization of the changing composition.

The beta source imports the accepted full-current rest-event/parent-boost implementation in `hot_beta_moments/hot_moments.py`. The PC source imports the independently reviewed vectorized Guo-invariant phase integral in `review_pc_moments/reference.py`: actual neutron tangent momentum, positron logarithmic momentum and scattering directions about the incident positron CM axis. This remains an explicit detailed phase integral, with recoil, physical masses, spin factors, nucleon blocking and electron/positron occupations. No source interpolation or fit is used. The imported PC function also computes prescribed FD-radiation scenarios; this experiment uses only its empty-radiation row. Shared full-current form-factor and omitted radiative/Coulomb/correlation assumptions remain inherited research qualifications.

Base PC rule is (neutron,positron,angle)=(28,44,10), tail60T; endpoint refinement is (32,52,12), tail70T. Base beta rule is (rest,parent,orientation,phi)=(16,12,6,8), refined=(24,20,10,12). Additional controls separately change PC tail and beta rest phase coordinate.

## Completed results

The initial recomputed source is R=1.11736186834×10⁻⁶ and Q=1.19946357969×10⁻⁵ in the units above, agreeing with the independently joined initial source within 4.1×10⁻¹⁰ relative. The refined initial source is R=1.11736186857×10⁻⁶, Q=1.19946358007×10⁻⁵. Both species-root and pair-charge forward gates precede integration.

Initial u=966.12690437 MeV/baryon, uT=49.2694621 and uY=−0.76733577 MeV. Initial dT/dt=−2.26047608 MeV/s: composition release offsets some escaping energy loss but does not reverse cooling in this selected pair-rich state.

| Steps over 0.002 s | Final Ye | Final T (MeV) | Final u (MeV/baryon) |
|---|---:|---:|---:|
| 1 | 0.322347237367 | 1.995453988831 | 965.887011653750 |
| 2 | 0.322073323444 | 1.995512587169 | 965.890077891730 |
| 4 | 0.321940128611 | 1.995541068003 | 965.891568379072 |

Observed refinement orders are 1.0402 for Ye, 1.0409 for T and 1.0407 for u. The first-order Richardson remaining-error estimates at four steps are 1.33×10⁻⁴ in Ye, 2.85×10⁻⁵ MeV in T and 0.00149 MeV/baryon in u. These are convergence estimates, not certified global error bounds or a high-accuracy trajectory. ΔYe=0.02194 is appreciable; this test does not rely on a tiny temperature change to establish feedback.

At the changed endpoint the refined source is R=1.06389786822×10⁻⁶, Q=1.13965514336×10⁻⁵, a decline of approximately 4.8–5.0% from the initial source. Every changed source is recomputed. Initial/endpoint source refinements, including each individual channel R and Q, differ by at most 5.24×10⁻⁹. The PC integrated neutron-density residual is at most 1.91×10⁻¹⁰ in this experiment. Genuine FD uT/uY derivatives are checked against separate EOS perturbations at initial and changed endpoint states.

Four-step escaped number is 2.19401286113×10⁻⁹ fm⁻³ and escaped energy 2.35335990559×10⁻⁸ MeV fm⁻³. Each channel is tracked separately. The discrete energy identity `εfinal + Eescaped = εinitial` closes within 3.09×10⁻¹⁰ relative to the escaped energy, including energy-root error. `nB(Ye_final−Ye_initial) = Nescaped` closes to roundoff. Material electron lepton number increases by Nescaped, escaped antineutrino electron lepton number is −Nescaped; baryon number and net charge neutrality are preserved. The beta channel contributes only 8.41×10⁻¹⁴ fm⁻³ events here but is retained explicitly.

There is no finite total radiation entropy diagnostic for exactly empty radiation: the derivative of FD entropy at zero occupation is singular. Material energy loss, composition release and net lepton bookkeeping are separate from any such entropy claim.

## Reproducibility, qualification and failures

Run `python study.py`, then `python extra_controls.py` from this leaf using the campaign's existing NumPy/SciPy Python. `results.json` contains complete states, sources, controls, time histories, separate channel escape totals, gates and the current-source/environment identity. `partial_trajectories.json` is saved after each completed time grid. Every source access checks current source/environment identity; an in-memory key includes exact binary T, Ye, nB and every integration control. No persisted joined source is loaded. All loaded campaign Python source files and conservative dependency-directory Python hashes are pinned, including `muon_vacuum/vacuum_cc.py`. No generic state-to-source table is used.

The first provenance list omitted that transitive vacuum-current module. Original element bytes/results are preserved as `initial_provenance_*`; the corrected fingerprint adds the module and loaded-source enumeration. Physics was rerun unchanged. A supplemental `from study import ...` collided with upstream `hot_beta_moments/study.py` after its path insertion: that unguarded validator executed and overwrote its own results before ImportError. The import failure and accidental output are preserved locally, and the parent was informed for archive restoration. Supplemental helpers now live locally with no ambiguous study import. No upstream Python source was edited. These operational failures are recorded rather than counted as physics passes.

This is an authored, bounded research checkpoint pending independent review. It qualifies selected-source accuracy and first-order convergence along this short restricted path. It does not establish a general source domain, full weak inventory, transparent kinetic equilibrium, physical rapid-thermalization mechanism, stiffness handling, long-time behavior, GR transport, interacting matter or external EOS/rate validation.

Final authored count: **65/65 checks pass** (55 main, 10 supplemental). Main batch 16.01 s, supplemental 3.42 s; individual trajectory experiments remain below 90 s. Separate PC tail changes affect R/Q by ≤4.47×10⁻¹¹; independent beta electron-energy versus pair-invariant rest coordinates agree within 4.10×10⁻¹³. Original provenance-version trajectories and final trajectories have identical physical endpoint values.
