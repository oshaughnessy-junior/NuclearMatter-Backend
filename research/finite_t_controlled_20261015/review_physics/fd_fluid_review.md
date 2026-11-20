# Independent physics review: local FD material feedback

15 October 2026. Reviewed frozen `fd_fluid_element/element.py`, `short_evolution.py`, research note, 24 initial qualification checks and 30 evolution checks. Pinned element/driver hashes are recorded in `fd_fluid_probe_results.json` and checked before/after probes. The complete upstream manifest matches current frozen bytes. Reviewer changed only new `review_physics/fd_fluid_*` files, installed nothing and replayed no physical source grid. Account/window matched at start (7% used).

**Accept the short no-photon, fixed-density FD energy/composition feedback demonstration.** It evaluates changing physical rates at distinct states and uses a genuine FD EOS inversion. It does **not** establish a converged long-time cooling trajectory, an error of order machine precision in the time solution, or astrophysical validity of the inherited free-NR/transparent ansatz.

## Matched first law and EOS derivatives

The matter EOS contains free NR FD neutrons/protons and relativistic pair-equilibrium leptons, with fixed common kinetic mass and neutron/proton offsets `(Q,0)`. Subtracting the common rest energy is harmless at fixed nB. Retaining Q and lepton rest/composition energy is essential. No separate Q heat term appears.

For fixed density, `u=(e_n+e_p+e_e)/nB`, `uT=(Cv_n+Cv_p+Cv_e)/nB`, and `uY=Ap−An+Ae`, with `Ai=mu_i−T mu_i,T|ni`. Electron derivatives hold **net charge** fixed while pairs equilibrate. These are the actual FD derivatives in code, not classical `3/2` heat capacity or a chemical-potential-difference substitute for uY. The optional photon term has the correct energy/Cv derivative, but the reviewed evolution explicitly chooses **photons=False**.

Empty-radiation sources are the complete **named** electron-current inventory: nue electron capture, antinue positron capture plus crossed neutron decay. Nuclear/electron blocking remains inside j; neutrino occupations are zero. The one-helicity DOS and signs give `F=(R_antinue−R_nue)/nB`, `q=(Q_nue+Q_antinue)/nB`, `dot(u)=−q`, and `dot(T)=(−q−uY F)/uT`. No extra c or spin factor belongs in those occupation-to-density moments.

The initially positive temperature derivative is consistent with loss of total matter energy: Ye decreases and composition energy is released. The previously independent local energy-moment diagnostic predicted the same sign/magnitude. A temperature rise is not evidence of neutrino absorption or an external thermostat here.

## Conservative variables, inversion and changing rates

The driver evolves **u and Ye**, adds the opposite escaped energy/lepton counters, then solves the full FD EOS for positive T. Positive uT makes that local inversion monotone on the tested branch. The algebraic updates enforce discrete energy/lepton bookkeeping; the independent reconstructed EOS energy additionally tests inversion and state assembly.

The initial source payload is shared only for the identical checked state and controls. Each distinct intermediate state is passed to the actual optical energy integrator; the exact-state memoization is not interpolation or a frozen initial-rate approximation. Recorded new-state source values change, and endpoint source order/tail checks are performed at the actual new endpoint. The inherited strict ordinary-coefficient transfer wrapper remains the physical source interface, with its already documented domain limitations; a separate emission-only heavy-M benchmark does not expand this trajectory's source domain.

The current frozen manifest supports provenance of the accepted initial payload. The standalone driver checks its state/control contents rather than enforcing all source hashes itself, so changed-upstream reuse should not be inferred safe: regenerate/revalidate qualification or enforce a full manifest contract before future runs with changed physics. No current mismatch was found.

## Independent endpoint and derivative probes

Executed `fd_fluid_probes.py`; **10/10 checks pass**:

- Independently compute baryon and lepton **momentum-space energy integrals** at the initial state and both final states, rather than using the energy-space EOS energy output. Endpoint energy agreement is within **6.67e-16 relative**; independently assembled matter-plus-escaped-energy ledgers close within **6.27e-16** on the stated scale.
- Escaped electron-lepton endpoints close within **5.56e-17** absolute.
- Finite differences of that independent momentum-energy assembly at the actual four-step endpoint reproduce genuine frozen-composition uT and composition derivative uY within **9.12e-11 / 4.96e-11 relative**.
- No-photon selection and distinct changing source-state records are explicitly checked. These record checks complement the independently assigned numerical source review; they do not claim a second full evaluation of every rate grid.

Species chemical-potential inversion and physical constants remain shared with the qualified free-FD reference. The new integrations independently test energy assembly, conservation and derivatives, not a different nuclear interaction theory.

## Temporal accuracy and accepted interpretation

The four-step endpoint after 2e-6 s is approximately T=2.004381407217 MeV, Ye=.294891996756, u=8.416314695636 MeV/baryon. Escaped energy is .085606567923 MeV/baryon. These are a useful short feedback calculation, not a stationary equilibrium or a full cooling curve.

The nearly exact energy/lepton ledgers primarily reflect conservative variable updates and accurate EOS inversion. They do **not** measure explicit Euler truncation error. Two versus four steps differ in Ye by about 2.78e-5 and u by 5.06e-4 MeV/baryon, around .6% of their changes over this interval. The much smaller T difference reflects compensation in the coupled EOS and is insufficient by itself to qualify the full solution. Two resolutions do not establish an observed asymptotic convergence rate or long-time stability.

The note keeps these distinctions explicit. Endpoint quadrature checks and the narrow T/Ye path support the local experiment; they are not a uniform source-domain certificate. A longer/higher-accuracy trajectory needs a further step sequence or adaptive integration with independently resolved source errors.

## Remaining physics limits

Neutrino transparency, instantaneous local lepton/pair thermalization, homogeneous matter and free nucleon species are imposed ansatz. No parcel size/escape time, photon energy, hydro/compression work, nuclear interactions, clusters, pairing, muons, scattering or reabsorption is solved. The external finite-T GM1 forward gate remains open. Retain those conditions when publishing this first FD material feedback example; do not label it a realistic neutron-star cooling model or a production neutrino backend.
