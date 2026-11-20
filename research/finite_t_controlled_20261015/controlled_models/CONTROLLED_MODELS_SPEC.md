# NMB finite-T controlled configurations: implementation specification

15 October 2026. Independent review of `docs/NMB_FINITE_T_RESEARCH_PLAN_2026-10-15.md`, incorporating the PI's clarification: detailed microphysics within controlled theoretical ansatz, without the throughput requirement of a GRMHD radiation backend. All thresholds below are proposed acceptance criteria, except the separately reported exact slab test results.

## Scope and ordering

Build a library of small, inspectable physical experiments. A converged quadrature taking seconds or minutes per state is useful here; throughput is a later question prompted by actual application profiles. Preserve a clear separation between matter model, collision operator, imposed geometry/kinematics, radiation closure and numerical representation. Analytical fits are scientific products and optional accelerators, not a prerequisite to exploring the detailed model.

Recommended order: (1) fixed-state coefficient and spectral comparisons; (2) prescribed fluid element with electron charged currents; (3) conservative closed matter-neutrino box; (4) exact and discrete-ordinate spectral slab; (5) prescribed compression/disk and fixed-background cooling comparisons. Muon channels are important, but need not delay an electron-only vertical slice with its species limitation explicit. Warm modified Urca and collision-broadened prescriptions remain distinct physical alternatives requiring careful matching.

No new stellar structure, accretion dynamics, turbulence model or general relativistic radiation hydrodynamics solver is required. A fixed TOV metric is an optional prescribed background. A Newtonian baseline should pass before adding redshift transport.

## Shared conventions and conservation contract

Use baryon density `n`, total matter energy per baryon `u(n,T,Y)` including a declared composition-dependent rest-mass/binding convention, and matter pressure `P`. For prescribed expansion `theta=-dot(n)/n`, the Lagrangian first law is

`dot(u) = P dot(n)/n^2 + (Q_ext-Q_nu)/n`.

Here `Q_nu` is positive for energy transferred from matter to the explicitly evolved or escaped radiation; it is signed when there is absorption. The temperature equation must follow the same EOS:

`u_T dot(T) = [P/n^2-u_n] dot(n) - sum_a u_Ya dot(Ya) + (Q_ext-Q_nu)/n`.

All partial derivatives hold the other independent variables fixed. The explicit `u_Y dot(Y)` term is essential. Do not add a separate nuclear Q heating term when its energy is already included in `u`; when an alternate energy convention omits it, transform the source explicitly.

For each neutrino flavor `a`, define occupation-number collision source `C_a(E,Omega)` in inverse seconds. For massless neutrinos with one helicity, phase-space measure is `dPi=d^3p/(2pi hbar)^3`, with `p=E/c`. Then

`R_a = integral C_a dPi`, `Q_a = integral E C_a dPi`.

Electron charged-current matter source is `n dot(Ye)=-(R_nue-R_antinue)`; muon charged currents have the analogous expression. Pair production conserves net flavor lepton number but transfers energy. Neutral-current scattering conserves neutrino number; inelastic scattering exchanges energy with matter. State explicitly which channels can exchange electron and muon flavor lepton numbers.

Emission/absorption is `C=j(1-f)-chi f`. Both `j` and true absorption `chi` have units inverse time. At thermal equilibrium `j/chi=exp[-(E-mu_nu)/T]`, and the relaxation coefficient is `j+chi`. A spatial characteristic attenuates with `(j+chi)/c`, even though true absorption opacity is `chi/c`. This distinction is often hidden by stimulated-absorption conventions. Gray moments require a supplied spectrum/closure, not a unique coefficient of matter alone.

Each run records immutable state units, energy zero, EOS family/parameters, species, process switches, physical correction set, angular/energy grids, geometry, boundary conditions, fixed versus evolved fields, reference version, numerical tolerances and code revision. Every budget includes externally supplied/removed energy and lepton number.

## A. Prescribed expanding/compressing fluid element

Hold `n(t)` and any irradiation history fixed. Evolve `T,Ye` and optionally `Ymu` using the equations above; evolve entropy only as a derived diagnostic. Start without radiation trapping: emitted neutrinos escape and their energy/lepton fluxes accumulate in an external reservoir. An irradiated variant supplies `f_nu(E,Omega,t)` explicitly and includes absorption, with the boundary radiation source debited in its ledger.

Use density histories constant, exponential expansion, and `n=n0[1+A sin(omega t)]`. At fixed density with an external thermostat, solve the transparent kinetic rate-balance composition from actual forward-minus-backward rates. Without a thermostat, transparent matter generally continues cooling even after its composition source vanishes; do not call this a stationary thermal equilibrium. At finite T the transparent kinetic root need not obey `mu_n-mu_p-mu_e=0`. [Alford et al.](https://arxiv.org/abs/2108.03324)

Observables: `T(t),Ye(t),Ymu(t)`, spectra, gross and net number rates, emitted energy, relaxation eigenvalues, pressure phase lag and cycle work. For driven compression, the external mechanical work must appear in the energy budget. Nonzero cooling and density-dependent heating can also contribute a cycle phase lag; a bulk-viscosity measurement must isolate weak relaxation against an otherwise identical frozen-composition/reversible baseline with a named thermal closure. Avoid applying an effective bulk-viscous heat source on top of explicitly evolved reactions.

Acceptance: baryon/charge constraints at roundoff from variable definitions; energy and lepton ledgers better than `1e-6` relative to the larger of initial thermal energy and integrated gross exchange, with the constant rest-energy offset removed from the error scale; numerical solutions change by less than `1e-3` in scaled `T,Ye` under step/tolerance refinement. Equilibrium is accepted only if the net rate residual is below `1e-6` of gross rates, with an absolute floor for frozen channels. For linear damping, amplitude-halving must reduce nonlinear contamination and cycle work must match the same-model Jacobian prediction within 1% in its resolved regime.

## B. Conservative matter-neutrino thermal box

Hold volume and baryon number fixed, with no boundary flux. Evolve matter energy/composition and energy-resolved isotropic neutrino/antineutrino occupations. Isotropy is the declared closure. Use the identical discrete collision event/moment to update both sectors; otherwise independently integrated source terms can create apparent heating or lepton leakage. Evolve at least electron neutrinos and antineutrinos; muon channels require their own lepton ledger. Pair kernels couple two energy variables and must not be replaced by an imposed zero neutrino chemical potential unless that is the chosen reservoir approximation.

A first absorption/emission-only box can establish conservation and detailed balance. It must not claim the thermalization properties of full inelastic/pair physics. The eventual equilibrium follows conserved total energy, baryons, charge and available lepton invariants, with `mu_n+mu_nue=mu_p+mu_e`. Antineutrino chemical potentials have opposite sign where pair equilibrium is established. Independent finite-volume quadrature gives the final equilibrium root used for comparison.

Acceptance: `0<=f<=1`; algebraic source cancellation to roundoff when the same moments are shared; global energy and lepton drift below `1e-8` relative to their declared thermal/exchanged-energy and gross-lepton scales over at least ten resolved relaxation times (do not hide drift behind the much larger constant baryon rest energy); final thermal occupations and matter equilibrium within `1e-4` on refined energy grids. On a coarse truncated grid, compare to that grid's equilibrium before assessing continuum tail error. Entropy nondecrease is a strong diagnostic only after specifying a collision operator with the required microscopic reversibility and a thermodynamically consistent EOS.

## C. Energy-dependent plane-parallel slab

Hold geometry `0<x<L` and initial matter profiles fixed for transport-only experiments. Supply incident occupations at `x=0,mu>0` and `x=L,mu<0`. Evolve or solve steady

`(1/c) partial_t f + mu partial_x f = C[f]/c`.

Use discrete ordinates and energy groups, retaining full angular source terms when scattering is introduced. Begin with homogeneous emission/absorption; for path length `s`, exact solution is

`f(s)=f_eq+(f_in-f_eq) exp[-(j+chi)s/c]`, `f_eq=j/(j+chi)`.

Then add elastic scattering and an optically thick diffusion comparison with an explicitly defined transport opacity including angular weighting. Inelastic scattering/pair redistribution comes after kernel reciprocity tests. A fixed background acts as a thermostat/composition reservoir: compute its energy and lepton exchange, but do not assert closed-system conservation for radiation alone. The coupled variant evolves matter in each zone and tests radiation plus matter plus boundary flux budgets.

Observables: emergent number/energy spectra, angular moments, deposition profiles, albedo, optical depth by energy, and diffusion rate. Acceptance: homogeneous analytic characteristics within `1e-8` for an analytic evaluator; a discretized solver below `1e-3` in boundary moments after spatial/angular/energy refinement; source-volume versus boundary-flux budget below `1e-6` with converged quadrature; thick diffusion flux within 1% only after the resolved-domain/closure asymptotic regime is demonstrated. Test optically thin streaming separately; a diffusion-only solver is not a benchmark there.

`slab_benchmark.py` is a standard-library exact oracle delivered with this review. It tests occupation bounds, detailed-balance fixed point, streaming, thick equilibration, source/flux convergence and an opacity derivative. It does not implement scattering or physical weak rates. Its trapezoid source integral uses finer grids at large optical depth because unresolved boundary layers can falsely fail a conservation test.

## D. Spherical cooling on a fixed background

Hold `n(r)`, composition profile, metric if used, gaps, envelope law and radius fixed. First use the Newtonian local model

`cV partial_t T = -div(F)-Qnu+Qheat`, `F=-kappa grad(T)`.

Insulating center and a prescribed outer flux or envelope relation close the model. The time derivative and `cV` must use the same volume and thermal-energy convention. Fixed composition is appropriate to an initial regression; chemical evolution and latent heat are named extensions. For fixed GR geometry derive redshifted temperature/flux equations consistently rather than adding factors by inspection.

Observables: total thermal energy, neutrino luminosity, surface photon luminosity, relaxation eigenmodes, and sensitivity to gaps/conductivity. Gate Newtonian global `delta E + integral(Lnu+Lgamma-H)dt` below `1e-6` of the larger thermal/exchanged-energy scale. Validate a constant-coefficient diffusion eigenmode before matched NMB/NSCool cooling. A fixed-background comparison qualifies these background-conditional predictions; it does not establish a new EOS-to-star derivative pipeline.

## E. Prescribed disk annulus or trajectory

Specify `r(t),n(t),H(t)`, irradiation and external heating rather than solving disk momentum/angular momentum equations. A first annulus is a slab with two radiating faces. Determine whether `H` is prescribed or obtained from a named vertical hydrostatic ansatz; changing it changes the model. Optical depth is computed from spectral opacity through this supplied geometry. Use explicit transport or an openly stated escape/leakage approximation, compared on identical fixed states against the slab. An escaping luminosity is not local emissivity at arbitrary optical depth.

Evolve energy and `Ye`, using low-density composition/NSE only within its qualification. Expose neutrino cooling, absorption, advection and expansion times separately. Compare equilibrated-composition and advected/frozen trajectories; do not impose weak equilibrium when the weak timescale exceeds inflow. A disk-annulus model can test local thermal stability only under its specified held-fixed variables (`Sigma`, density, external heating law, etc.); it does not by itself predict global disk fragmentation.

Acceptance: the fluid-element budgets above plus two-face boundary luminosity balance; transparent limit agrees with volume emissivity within 1%; opaque slab-based case converges to its transport solution. A leakage fit is accepted only on a declared optical-depth/spectrum family with less than 5% error in the chosen luminosity/heating/Ye observables. State networks and energy conventions at any Genesis handoff.

## Specific corrections and source qualifications

1. Han et al.'s gamma-th fits concern uniform QMF matter over their stated hot domain; their equilibrium curves impose thermodynamic beta equilibrium. They are not demonstrations of hot transparent kinetic balance, nor of opacities. The published composition feature `(1-beta)^(1/3)` can have singular derivatives at `beta=1`; check cancellations and endpoint behavior rather than inferring derivative accuracy from value residuals. Their quoted fit accuracy is a value claim. [Paper, Sections II and V](https://arxiv.org/html/2610.18103v1)
2. The Raithel–Ozel–Psaltis erratum DOI in the original plan is correct. It replaces the erroneous half-effective-mass at half-density prescription by symmetric-matter effective mass evaluated at total density. The authors also correct thermal expressions and entropy accounting; importing old equations while updating only parameters is insufficient. The institutional author record reproduces the erratum abstract, but its missing rendered equations mean the actual erratum PDF remains required before implementation. [Author institutional record](https://experts.arizona.edu/en/publications/erratum-finite-temperature-extension-for-cold-neutron-star-equati/), [published erratum](https://doi.org/10.3847/1538-4357/ac0630)
3. EOS-consistent quasiparticle spectra/mean fields are necessary for charged-current comparisons; a cold barotrope or thermal index does not specify them. Validate a common free-gas/mean-field approximation before interpreting code differences as nuclear-physics differences. [Roberts and Reddy](https://arxiv.org/abs/1612.02764)
4. Modified Urca and nucleon-width inclusive rates require a stated matching policy. Do not sum them by default. [Alford, Haber and Zhang](https://arxiv.org/abs/2406.13717)
5. Replace universal derivative-relative-error requirements near zeros, onset boundaries, phase contacts and suppressed tails by scaled absolute errors and explicit one-sided limits. Rates fit to 1% can still have unusable derivatives, and an accurate fitted gamma-th alone does not construct a consistent free-energy EOS. Fit detailed balance and positivity structurally before optimizing coefficient count.

## Realistic next 24 hours

Hours 0–4: freeze source/state contracts and species scope; verify accessible independent references; finish analytical slab/free-gas oracles and run manifests. Hours 4–10: implement or stage a small finite-T lepton/free-gas and one-family state vertical slice only where repository access, policies and reference data permit; keep unsupported state physics explicit. Hours 10–16: compare fixed-state number/energy source conventions and start a fluid-element or conservative-box demonstration against already available qualified rates. Hours 16–22: refine one benchmark, record failed gates, and assess one fit's values/derivatives on blocked holdouts. Hours 22–24: integrate evidence into the research note with completed versus proposed work separated and a prioritized next implementation queue.

Do not promise a qualified arbitrary-degeneracy RMF plus finite-mass muon charged-current response, warm modified Urca, paired transport and NSE in one day. The useful one-day product is executable contract validation plus a narrowly working vertical slice or explicit blockers, not blanket production certification. Apply quota guardrails before each batch and stop at the parent program's bound; elapsed scheduling time is not compute authorization beyond that bound.
