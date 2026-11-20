# Independent review: selected conductive-slab spatial self-convergence

**ACCEPT the selected mild nonlinear spatial self-convergence and measured time-error separation.** This closes the previously absent selected-case dynamic spatial check; it is not a universal continuum certificate, a blanket second-order result in every norm, or qualification of a strongly temperature-dependent transport regime. There is no selected-scope blocking finding.

Frozen author `conductive_spatial_refinement/refine.py` SHA256 is `78364b0e3bd51214d962743aa15c127cfe97e1d2624ebc7dede137e43064e3cb`. The existing slab, Lorentz transport and EOS sources remain unchanged. **93 review checks pass**:16 independent initialization/RK2 checks and77 source/projection/norm/time/EOS checks. Source pins and related controls contribute to that count; it is not93 independent physical benchmarks. Probes/raw evidence are `probe.py`, `audit.py`, `results.json`, `audit.json`.

## Compatible initial state and projection

The initial condition is theta(x)=1+.04cos(πx), but finite-volume cell data represent averaged **conserved energy W(theta)**, not theta at the cell center. The author integrates W on the finest16cells and exactly projects those cell averages to8/4cells. The initial temperature of each uniform cell is then the actual EOS inversion of that average. Thus each grid represents the same integrated physical energy. The rest-energy subtraction in W is constant at fixedne and cancels from thermal changes. No classical or ion heat capacity enters.

The review independently performs adaptive Gauss–Kronrod quadrature of actual W on4/8cells, rather than the author's Gauss8/12-node rules, and compares both with finest-grid projection. These initial-energy checks pass. The projection is independently assembled from physical overlaps of fine and coarse cells, rather than reshape/mean. It preserves volume and integrated energy. Actual EOS temperatures are obtained with independent bracketed roots. Averaging fine temperatures instead would generally give a different coarse energy; this error is absent.

Each grid's entropy is evaluated after its energy inversion. Coarse averaging loses resolved thermal variation, so equal initial total entropy across resolutions is neither expected nor claimed. Equal initial conserved energy, not equal initial arithmetic temperature or entropy, is the relevant compatibility requirement here.

## Independent reconstruction of every spatial norm and rate

The review recomputes overlap projections and norms using separate arithmetic. L1 is the volume-weighted mean absolute W difference; L2 is the volume-weighted RMS difference; Linf is the maximum cell difference. These are self-convergence differences, not absolute errors against an exact solution:

| Grid pair | L1 | L2 | Linf |
|---|---:|---:|---:|
|4 vs projected8|1.4031406346240363e−5|1.530456353495366e−5|2.1808115109996085e−5|
|8 vs projected16|3.5769283428801427e−6|3.986110892954558e−6|6.216989252225136e−6|
|log2 ratio|1.971866493287951|1.94090815039004|1.8105770923725824|

These reproduce the raw data and note. The Linf rate is1.81 and must remain visible. Calling every norm second order without qualification would overstate the evidence. Only three selected coarse grids have been tested; no extrapolation to arbitrary amplitudes, densities, coefficient variation or boundary configurations follows.

## Time error separated from stability

The fine-grid16/32/64step differences independently reproduce the approximately.5 ratios expected for first-order Euler time stepping. For a smooth Euler expansion U(dt)=U(0)+a dt+..., the leading error at the coarse16stepdt is estimated by2[U16−U32]. The author projects this vector onto the corresponding coarse comparison grid before taking each norm. That is the appropriate comparison of cell-average quantities. It is an empirical Richardson estimate, not a rigorous bound on every grid's error.

The largest projected time estimate is4.8010162% of the8vs16 spatial difference and1.2277416% of4vs8. Independent recomputation of all6 fractions passes. Separately, changed-state4/8cell trajectories at32steps allow spatial differences at the common smallerdt to be formed. The maximum norm of the *change in the spatial residual vector*, divided by its original norm, is.0006118256=.0611826%. This is stronger evidence for selected time insensitivity than quoting CFL stability alone. Both results are reproduced directly from the frozen raw arrays, and the supplement's pinned original-data hash/source hashes pass.

## New independent nonlinear trajectories

The review freshly evolves4/8cells using second-order SSPRK2, actual changed-state conductivity, independently adaptive initial energy integrals and independent bracketed EOS inversions. Its spatial heat flux retains the physical harmonic-face discretization so that the time-integrator comparison addresses the same finite-volume operator. It neither calls the frozen `slab.evolve` nor interpolates EOS or conductivity.

The new16step RK2 endpoints agree with the Euler Richardson extrapolation2W32−W16 to at most2.03e−12 in any norm. This agreement is near the EOS inversion tolerance and should not be advertised as an independently known continuum solution at that precision. The actual reconstructed EOS energy ledger closes within2.504e−13 normalized units. Entropy increases at every saved time, temperatures remain within their initial bounds, and no boundary energy/entropy flux is introduced in the isolated problem. The fresh runs take10.67s and21.67s. No new16/32/64cell run or full58s author replay was needed.

For every saved author grid/time endpoint, the review also recomputes W and entropy from the actual EOS temperatures. These match the stored final energy arrays and isolated ledgers. Source input ne=5.3e−8fm−3 is independently gated before interpreting the new coefficient values. Thus array projection agreement is complemented by a physical forward/EOS check, rather than testing only bookkeeping identities.

## Mild nonlinearity and physical limits

Fresh source evaluations reproduce kappa/kappa0 andCv/Cv0 at theta=.96,1,1.04. Each changes across a range of about8%, while their diffusivity ratio spans only.0007988346≈.0798835%. This is an actual variable-coefficient EOS/conductivity calculation, but the selected diffusivity is almost constant. It is a useful mild nonlinear gate and cannot stand in for a strong nonlinear transport stress test. No fixed-coefficient surrogate is used to obtain the trajectories.

The prior physical qualifications remain: fixedne, unmagnetized open-circuit electron conduction, immobile uncorrelated external ions, no ionCv or mechanical work, and negligible equilibrium positrons under the electron-only gate. The local FD distribution is imposed. Elastic Lorentz scattering alone does not redistribute isotropic electron energy into FD, and this study supplies neither the missing local thermalization kernel nor its effect onkappa. The named physical model is not a calibrated crust/core transport prescription; same-model numerical agreement does not quantify physical model uncertainty.

The chosen100µm slab and6.25µm finest cells are much larger than the representative Fermi-surface mean free path. Those diagnostics support the selected angular-diffusion ordering, not all-energy kinetic accuracy or an energy-thermalization time. The isolated spatial study does not add new reservoir/field/ion energy dynamics. Previously reviewed reservoir signs and ledgers remain in the parent slab leaf.

## Retained records and remaining gates

The initial averaging-generator exception and exact initial source are preserved; it occurred before evolution and is not a failed physical result. No new review gate failed and no author source was modified. All frozen source/raw hashes and coefficient-data provenance pass.

The accepted evidence is selected spatial self-convergence with temporal contamination measured and small. Stronger nonlinear amplitudes, broader states, alternate conductivity physics, further spatial grids or an independent continuum reference remain later gates. A general large-contrast nonlinear boundary CFL theorem also does not follow from this isolated mild test. Physical FD energy redistribution remains OPEN. This does not qualify hydro, production NMB replacement, radiative/ion heat capacities or realistic correlated/mobile plasma conduction.
