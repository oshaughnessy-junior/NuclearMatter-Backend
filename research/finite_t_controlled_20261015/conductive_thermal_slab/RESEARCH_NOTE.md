# Fixed-background electron thermal diffusion slab

**A conservative finite-volume thermal slab now uses the actual screened Lorentz-gas open-circuit conductivity and matched electron energy/Cv. 53 author checks pass; independent review is pending.** The selected model is a fixed, neutral, unmagnetized electron background with immobile uncorrelated external ions, no ion heat capacity, no photons, hydro or neutrinos. It edits and replaces no existing NMB conductivity.

The reference state is T0=.041MeV, ne=5.3e−8fm^-3, Z=2. Positrons are excluded under the upstream electron-only pair gate; their reference density ratio is5.83e−34. Conductivity is the accepted screened Born/Mott local-susceptibility Lorentz reference. Its local potential, Born approximation and static-ion assumptions remain physical choices, not a realistic crust/core calibration.

## Evolution and closure

At fixed ne, with the local open-circuit charge constraint j=0,

\[
\partial_t u_e(T,n_e)=-\partial_x q,
\qquad q=-\kappa_{j=0}(T,n_e,Z)\partial_xT.
\]

The energy, entropy and fixed-density heat capacity come from the **same relativistic spin-two particle electron EOS used by the conductivity**, including the electron rest energy in u. That rest contribution is constant at fixed ne and cancels from thermal changes. There is no classical3ne/2 substitution and no ion Cv. At the reference, Cv=9.563014595970627e−9fm^-3 and kappa=107.25446830892398MeV². Two decreasing finite-difference steps of actual EOS energy reproduce Cv within2.2e−9 relative.

The local-FD distribution is **prescribed**. Elastic Lorentz collisions relax angular distortions but do not redistribute an arbitrary isotropic electron energy distribution into FD. This model assumes rapid local energy redistribution conserving n and u; no such collision kernel, its relaxation time or its possible change to kappa is calculated. The open-circuit electrochemical field is slaved to the local thermoelectric constraint, with quasineutral background and no Maxwell/field-energy evolution. This is a controlled thermal-closure ansatz.

Define

\[
\theta=T/T_0,\quad \xi=x/L,\quad \tau=t\kappa_0/(C_{V0}L^2),
\quad W(\theta)=[u_e(T)-u_e(T_0)]/(C_{V0}T_0),
\quad k(\theta)=\kappa(T)/\kappa_0.
\]

Then \(\partial_\tau W=\partial_\xi[k(\theta)\partial_\xi\theta]\), and dW/dtheta=Cv(T)/Cv0. The finite-volume scheme advances **W**, then inverts the actual EOS for T. Face conductivities are harmonic means; the explicit step uses a conservative CFL bound based on max(k)/min(Cv/Cv0). A Newton/bracket solve has a saved energy residual gate. The isolated case has zero face flux; the reservoir case uses fixed face temperatures with half-cell boundary distances.

Every changed state obtains actual EOS and conductivity quadratures. The in-memory cache uses exact floating-point temperature keys and rejects changed sources, Python/NumPy/SciPy, T0, ne, Z or quadrature controls before lookup. It is not an interpolant, emulator or persistent trajectory source cache. The final qualification used4845 actual EOS evaluations and2990 actual transport quadratures; cached values are reused only at identical states.

## Physical scale conversion and kinetic ordering

The primary solver is dimensionless. Physical scales are independently derived using hbar=6.582119569e−22MeVs and hbarc=197.3269804MeVfm, with1fm=1e−15m:

\[
L_{nat}=L_m/(\hbar c)_{\rm MeV\,m},\qquad
C_{V,nat}=C_{V,\rm fm^{-3}}(\hbar c)^3,
\]
\[
t_0=\hbar C_{V,nat}L_{nat}^2/\kappa_0,\qquad
D_{SI}=(\kappa_0/C_{V,nat})(\hbar c)_{\rm MeV\,m}^2/\hbar.
\]

For a **chosen100µm slab**, t0=1.1580616017067412e−7s and D=.08635119224454113m²/s; the simulated duration tau=.02 is2.3161e−9s. The independent hbar*c consistency check is2.23e−10 relative, reflecting the named rounded constants.

The reference Fermi-surface collision time is3.0096e−18s, mean free path8.8066e−10m and Kn=lambda/L=8.8066e−6. The screening length is8.8247e−13m and pF*lambda in natural units is10233.9. These establish a useful selected-state separation of screening, angular collision and slab scales. They do **not** establish the missing local energy-thermalization rate, certify every high-energy electron's mean free path, or make the static-ion conductivity a realistic medium model.

## Independent analytic diffusion gate

With prescribed constant k=1 and Cv/Cv0=1, fixed equal boundary temperatures and a small sine perturbation, the continuum amplitude is A exp(−pi²tau). The cell-centered spatial eigenvalue is

\[
\lambda_h=4\sin^2(\pi/(2N))/\Delta\xi^2,
\]

so explicit-Euler amplitude is independently known as A(1−lambda_h Delta_tau)^steps. The numerical calculation matches that discrete prediction and preserves the sine shape without fitting a decay rate. N=8,16,32 spatial errors decrease by factors about.248/.250, and time halving at fixed N gives the expected first-order decay-error reduction.

This tests the transport discretization separately from the nonlinear physical coefficients. It does not declare a finite-amplitude variable-coefficient problem to be exactly linear.

## Nonlinear actual-physics runs and ledgers

Eight cells evolve to tau=.02 with16,32,64steps. The isolated initial condition is theta=1+.12cos(pi xi). The reservoir run starts at theta=1 with fixed left/right faces1.08/.92. Both stay within the initial/reservoir temperature bounds. Time-step differences between16/32 and32/64steps decrease by factors.496 (isolated) and.516 (reservoir), consistent with first-order time evolution.

Material energy is independently reconstructed from the inverted EOS, not only the updated conserved array. Incoming and outgoing face energy are integrated **separately**. The maximum normalized material-energy minus integrated boundary-flux residual is8.39e−13 across both runs. For the64step reservoir run,

- incoming energy integral=.012591430297622044;
- outgoing energy integral=.011619236963321676;
- actual material energy change=.0009721933340890568.

These are normalized per-area quantities in units Cv0*T0*L; they are not CGS fluxes. The closed isolated run has initial/final normalized material energy.00358555824069854/.003585558240631037. Conserved energy does not imply a conserved arithmetic mean temperature for nonlinear u(T).

Using normalized entropy [s(T)−s(T0)]/Cv0, the physical boundary entropy input is

\[
\int d\tau\,[q_L/\theta_L-q_R/\theta_R].
\]

The actual material entropy change minus that input is positive at every saved time. At64steps, total entropy production is.0011661374132883886 (isolated) and.0014163048101171315 (reservoir). The reservoir material entropy change is.0004454311722597127 and boundary entropy input is−.0009708736378574189; reservoir entropy must not be omitted when interpreting the sign.

A separate face sum of q_face times the jump in1/theta approximates the instantaneous entropy-production integral. It is **not set equal by construction** to the finite-step entropy change: their discrepancies are1.83e−6 and1.34e−5 at64steps and decrease approximately linearly with the time step. This distinguishes the entropy balance from a falsely exact finite-time Euler identity.

## Independent constant-flux steady reference

The continuum steady state is derived from the actual conductivity potential

\[
G(\theta)=\int_{.92}^{\theta}k(z)dz,\qquad
G(\theta(\xi))=G(1.08)(1-\xi).
\]

Independent quadrature and scalar roots give constant normalized flux **.16000305759222203**. Sampling that profile on4/8cells, the independently assembled finite-volume face fluxes differ from it by at most4.6932e−4/1.2807e−4 relative, consistent with the selected spatial refinement. The transient reservoir run at tau=.02 is not claimed steady.

`thermal_profiles.png/.pdf` shows the actual eight-cell short-time profiles and the independent steady integral; its measured-data provenance is saved. Lines join calculated cell values. Dynamic spatial convergence beyond the analytic constant-coefficient gate and these steady checks remains open; eight-cell transient results are not a continuum accuracy certificate.

## Preserved failure, limits and reproduction

`initial_entropy_gate_failure.json` and the exact initial validation/source bytes preserve six failed author assertions: the error-style test mistakenly compared **positive** entropy production against zero with an absolute-error gate. The physical entropy values were positive. The assertion was corrected without changing its physical inequality; the experiment was rerun. A cache fingerprint omission of T0 and a safer CFL bound were also corrected before freeze. All53 final checks pass. Initial/final full experiments took65.04/66.12s, each within120s; final physical runs each took under14s.

The next gates are an independent review, dynamic spatial refinement with independently controlled time error, broader temperature/density transport qualification, a physical energy-thermalization model, and eventual realistic electron-ion/electron-electron conductivity choices. No hydro, ion heat capacity, magnetized transport, photons, neutrinos or NMB production replacement is supplied.

Reproduce from the repository root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/conductive_thermal_slab/validate.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/conductive_thermal_slab/make_figure.py
```
