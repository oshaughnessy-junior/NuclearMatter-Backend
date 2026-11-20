# Transparent isochoric FD element: thermodynamic closure and short feedback test

Research prototype, 2026-10-15. This is a spatially homogeneous, fixed-volume/fixed-baryon-density experiment with the previously reviewed complete charged-current source inventory. It supplies a genuine FD energy/composition closure, not full hydrodynamics or a surrogate. Independent review is required before accepting the new element result.

## State and physical source

The element contains free NR neutrons/protons and ideal relativistic pair-equilibrium electrons/positrons, with local charge neutrality np=ne,net=Ye nB and nn=(1−Ye)nB. The common nucleon rest energy is subtracted, while the neutron offset Q and total electron/positron energy remain in the EOS. Offsets and kinetic mass are fixed as composition and temperature vary. Muons, nuclei, pairing, Coulomb/nuclear interactions, scattering and spatial transport are absent.

Use the **complete** reviewed allowed-current NR inventory: electron capture/inverse for νe, and positron capture plus crossed neutron decay/inverses for anti-νe. The mapped-capture collapsed-panel guard and crossed channel are explicitly inherited through the reviewed `quantum_slab` coefficient factory. Transparent radiation means fν=fanti=0 at every energy, so only spontaneous j contributes to the emitted sources. Target occupations, final nuclear/electron blocking and exact quadratic recoil still enter j. This is an escaping-radiation ansatz, not trapped/inverse-channel beta equilibrium.

With D=1/[2π²(ℏc)³] for one helicity, define the physical volume moments

Rα=∫₀∞ D E² jα(E)dE [fm⁻³ s⁻¹],
Qα=∫₀∞ D E³ jα(E)dE [MeV fm⁻³ s⁻¹].

There is no extra c or spin factor. The material lepton and energy equations at fixed nB are

F ≡ dYe/dt=(Ranti−Rν)/nB,
q ≡ (Qν+Qanti)/nB ≥0,
du/dt=−q,

where u is total material energy per baryon in the stated subtracted-rest convention. Common rest subtraction is harmless because nB is fixed; discarding Q or electron rest/composition energy would not be harmless. An electron-capture event lowers Ye, while an antineutrino emission event raises Ye.

## Genuine FD temperature/composition equation

Let Ai=(∂εi/∂ni)T=μi−T(∂μi/∂T)ni. This follows from f_i=ε_i−Ts_i, f_i,n=μ_i and the Maxwell identity s_i,n=−μ_i,T. For electrons ni is net conserved electron charge, with positrons included in ε, s and Cv. The derivatives are taken at fixed kinetic mass and offsets.

Then the matched EOS returns

uT ≡ (∂u/∂T)nB,Ye=(Cv,n+Cv,p+Cv,e)/nB,
uY ≡ (∂u/∂Ye)nB,T=Ap−An+Ae,
dT/dt=(−q−uY F)/uT.

Every Cv and μT is evaluated from the full FD moment derivatives. Neither uT=3/2 nor uY=Q is substituted. The composition term is essential even in a transparent cooling calculation. The corresponding first-law entropy relation is T ds/dt=−q−(μp+μe−μn)F, with s per baryon; inverse-channel KMS does not set that finite-temperature affinity to zero.

Photons are an explicit optional EOS term εgamma=π²T⁴/[15(ℏc)³], adding 4εgamma/T to Cv and nothing to uY. The short evolution below selects **photons=False**. Both photon choices are independently checked for EOS derivatives. No radiation momentum, pressure work, expansion or density evolution is included.

## EOS and physical-source qualification

`qualify_source.py` independently finite-differences both uT and uY at decreasing steps for (T,Ye)=(2,0.3),(2.2,0.28), and at the first state with photons. It also checks inversion of the full EOS energy back to T. The initial physical source is independently integrated at outer energy orders6 and10, inner energy/angle48, internal tail40T, with explicit low-energy/static-scale/bulk panels to100 MeV. Both number and energy moments for both species are compared with the previously independently accepted320 MeV integrals; this checks the shorter tail and new panel pattern together. Representative full internal quadrature refinements are retained from the accepted moment study and repeated at the new endpoint below.

All **24/24** initial EOS/source gates pass in 37.9 s. At nB=10⁻⁴ fm⁻³, Ye=0.3, T=2 MeV, the closure gives dYe/dt=−2596.36090219 s⁻¹ and dT/dt=**+2263.85977760 MeV/s**, matching the independent moment-physics diagnostic. The positive temperature derivative is physically consistent within this ansatz: composition-energy release exceeds the escaping neutrino energy at that instant. It is not a sign repair or an imposed heating function.

## Short-time integration method and work limit

`short_evolution.py` evolves u and Ye directly by explicit Euler updates, then solves the full matched FD EOS for the new T. Thus u_next=u−q dt and Ye_next=Ye+F dt. An independently reevaluated EOS energy at the recovered T checks the radiated-energy ledger; escaped electron-lepton number checks Ye plus radiation. Neutrality is checked again through the electron density root.

The selected total duration is 2×10⁻⁶ s, compared with the initial Ye-change timescale about1.16×10⁻⁴ s. Two versus four steps supply a local step-refinement comparison. The identical initial physical source can be shared between resolutions, with exact state and controls checked against the passing qualification payload. **Every distinct new state evaluates the actual physical energy integrals again.** No old-state rate is reused as feedback, fitted, extrapolated or interpolated.

The numerical budget estimate is about100 s for the short experiment: four distinct intermediate-source evaluations, a new final-state order6/order10 comparison, and representative final-state internal48→72/tail40→50 comparisons. Each source uses the same named physical kernel, not a surrogate. This is a bounded local experiment; two resolutions cannot certify long-time stability or an asymptotic error law.

## Restrictions inherited from the reviewed moments

The accepted finite-M full-energy source is local to the named free-NR ansatz and sampled quadrature domain. The heavy-static complete-inventory energy-integrated gate remains excluded because its strict positive-coefficient wrapper underflows in a nearly forbidden band; it is not repaired here. Dense interacting EOS validation, finite-T GM1 external validation, neutron-current corrections, clusters and general cooling trajectories remain open.

Source files, raw source/panel values, root/derivative checks, step ledgers, work cost and upstream hashes are saved beside this note. No production, dependency, git, upstream physics or review file is modified by this leaf.


## Completed local feedback result

All **30/30** short-evolution gates pass in **110.27 s**, within the separate120 s numerical experiment bound. Five new-state physical source calls are recorded: four distinct intermediate states across the two resolutions, and the final endpoint for qualification. The separate initial-source/EOS qualification cost37.9 s. No state-dependent source was replaced by the initial source.

At 2×10⁻⁶ s, the four-step solution is T=**2.004381407217 MeV**, Ye=**0.294891996756**, u=**8.416314695636 MeV per baryon**. Escaped radiation carries0.085606567923 MeV and0.005108003244 electron-lepton number per baryon. The initial-to-endpoint magnitude of dYe/dt changes from2596.36 to2485.04 s⁻¹; the escaping-energy rate changes from43576.25 to41546.54 MeV per baryon per second. This is actual recomputed physical feedback.

Four-step minus two-step differences are ΔT=−8.73×10⁻⁹ MeV, ΔYe=+2.77586×10⁻⁵ and Δu=+5.06315×10⁻⁴ MeV per baryon. The small temperature difference alone would not qualify the trajectory: the differences in escaped energy and composition remain measurable (about0.6% of their short-interval changes). Two resolutions show a local step sensitivity, not a long-time/asymptotic convergence certificate.

The largest EOS-reconstructed material-plus-escaped-energy ledger residual is4.18×10⁻¹⁶ on the recorded scale; electron-lepton residual is5.56×10⁻¹⁷ and net-electron neutrality residual5.56×10⁻¹⁶. These are conservation/root checks, not a discretization-error estimate. The genuine EOS derivative finite-difference comparisons differ by at most4.42×10⁻⁹.

At the actual new endpoint, outer6→10 physical source differences are at most8.36×10⁻⁷ relative (the tiny anti-νe source), within the unchanged3×10⁻⁶ gate. At E=4 and16 MeV for both species, internal48→72 with tail40→50 changes j orχ by at most3.56×10⁻¹⁵. The initial outer6→10 discrepancy is8.46×10⁻⁷, while the initial refined100 MeV integration agrees with the accepted320 MeV reference to1.86×10⁻⁸. The intermediate states use the same controls and stay in the narrow interval T=[2,2.00439], Ye=[0.29486,0.3]; no global domain claim is made from endpoint checks.

No new failed EOS/source/step gate occurred in this bounded leaf. The existing moderate-state coarse energy grids and heavy-static integrated failure remain preserved upstream and are not altered or reclassified. Independent review must still check source-state provenance, temperature/composition derivatives, integration controls and the local step interpretation before this prototype is accepted.

Reproduce `qualify_source.py` first, then `short_evolution.py`, using Python from the workspace root. The latter deliberately consumes the passing initial physical-source payload only after checking its exact state and quadrature controls; each different state is freshly integrated.
