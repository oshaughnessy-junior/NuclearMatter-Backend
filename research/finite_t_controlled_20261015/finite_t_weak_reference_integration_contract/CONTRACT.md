# Portable finite-temperature free charged-current CPU reference: integration contract

Status: proposed staged design, NOT an implemented or qualified package API. This audit reads frozen accepted research results; it runs no new weak/EOS calculations. Independent design review is required before implementation. Default detailed physical quadrature serves controlled fluid elements, thermal boxes and prescribed irradiation/slabs, rather than production GRMHD performance. Species inventory is three explicitly named free electron charged-current reactions, not every Urca process or every neutrino interaction.

## First deliverable and bounded packaging

Implement first a standalone, separately imported CPU reference accepting explicit chemical baths. Do not wrap campaign files at runtime. The first candidate has eleven changed paths (names proposed, package-root spelling must follow the actual repository):

1. `weak_reference.py`: immutable public Bath, Current, Radiation, Controls and result records, validation and named-channel dispatch.
2. `_weak_current.py`: source-labelled full Dirac vertex, reciprocal adjoint and scalar invariant contractions; explicit immutable constants, no module reconfiguration.
3. `_weak_two_body.py`: physical EC and PC CM events, original moving-target support and separately positive forward/inverse integrations.
4. `_weak_decay.py`: exact moving-neutron three-body decay/inverse, proper spin measure, finite Dalitz phase and lab boosts.
5. `_weak_quadrature.py`: bounded positive/log accumulation, endpoint maps, convergence controls, diagnostic termination and representation records.
6–7. `tests/test_weak_reference_current.py`, `tests/test_weak_reference_moments.py`.
8. `tests/data/free_weak_reference.json`: provenance-pinned selected forward values and controls, no executable campaign loader.
9. `examples/free_weak_reference.py`: explicit-chemical T2 bath, direct F/B and energy/lepton ledger.
10. `docs/weak_reference.md`: equations, conventions, omissions and selected qualification domain.
11. README: short additive discovery link; preserve all existing cold APIs/exports.

This first PR supplies TOTAL event-number and radiation/material energy moments for all three named channels. It must not advertise spectral opacity until a second independently reviewed slice adds `_weak_spectral_capture.py`, `_weak_spectral_decay.py`, separate spectral tests/fixtures and documentation (five changed paths, with public dispatch additions). Stage B uses exact q-axis particle-hole support for capture and conditional moving-parent decay phase; it reconstructs Stage A totals independently. This staged split avoids hiding an extensive phase-space port in one nominal wrapper. Stage A physical totals default to detailed quadrature; a requested unavailable spectrum raises explicitly.

A later state binder may depend on independently merged particle-baryon and ideal-pair CPU utilities. The first explicit-chemical kernel has no dependency on those unmerged candidates. No new dependencies beyond declared NumPy/SciPy are proposed; audit actual repository minimum versions before using special functions. Use stable `-logaddexp(0,-x)` rather than assuming SciPy `log_expit` exists at the minimum. Do not export through existing cold/JAX package imports implicitly. Actual filenames/scope remain subject to independent design review.

## Material and radiation API

Proposed pseudocode signatures (not code to import):

- `FreeElectronBath(T, mu_n, mu_p, mu_e, model=PhysicalFreeNucleons())`: finite non-Boolean real total chemical potentials and positive T; occupations are per spin. Positron chemical potential is exactly `-mu_e`. Baryons are particles only; no hidden antiparticle, neutrality or beta-equilibrium assumption. A deliberately nonneutral chemical bath is allowed only with that label.
- `Model`: immutable MN=939.56542052, MP=938.27208816, ME=.51099895069 MeV; HBARC=197.3269804 MeV fm, HBAR=6.582119569e-22 MeV s, GF=1.1663785e-11 MeV^-2, Vud=.974; explicit source current parameters below. First qualification fixes this physical model; arbitrary masses/couplings are not automatically admitted.
- `Radiation.empty()` or `Radiation.fd(Tnu,mu_nue,mu_antinue)`: independent flavors, positive radiation T, explicit actual occupations/holes, massless one active helicity per flavor. LTE requires Tnu=T and separately specified equilibrium chemical potentials; empty is exactly zero occupation, not an extreme finite chemical potential. Arbitrary user functions are deferred until boundedness/metadata validation is designed. First port requires Tnu=T; different-T FD is deferred until independently qualified.
- `channel_moments(channel, bath, radiation, controls)`: channel enum EC/PC/NEUTRON_DECAY; returns separate positive F/B event densities and positive neutrino/species energy moments plus separately accumulated signed net moments, units, source/model identity, actual bounds/nodes and convergence diagnostics. Channel-local radiation selects nue for EC and antinue for PC/decay; no silent flavor substitution.
- Stage B `spectral_coefficients(channel,bath,E,controls)` returns direct positive/log j and true chi, and log-safe relaxation; no external neutrino occupation is included. `differential_coefficients(...,El,cos)` only applies to capture; a decay conditional-energy/Dalitz API is separately named.
- Later `matched_free_npe(T,nB,Ye,...)` explicitly solves nn=(1−Ye)nB, np=Ye nB, electron NET=Ye nB using independent EOS utility APIs and exact shared masses/HC. It returns particle n_e− AND n_e+, root residuals and baryon anti-equilibrium diagnostics. Muons are ABSENT, rather than calling a zero-net muon EOS and retaining pairs. No hidden warm beta solve; cold beta fractions are not warm beta equilibrium.

Number densities are fm^-3, NOT mass density or net baryon antiparticle density. E, mu and all species energies include rest mass; T is MeV, not Kelvin. Controls declare axes/orders/maps/tails and a wall-time limit, do not alter physical widths. Bath/model/control objects cannot be mutated between phases. Preserve exact source versions as build/provenance metadata, not filesystem reachbacks to campaign absolute paths or mutable JSON qualification files.

## Reactions, occupancies and conservation

Let fn,fp,fminus,fplus denote FD per-spin occupations; neutrino occupations are excluded from j and chi. Forward always means emitting the named neutrino.

| Channel | Forward reaction | j material occupation product | chi material occupation product | equilibrium radiation mu |
|---|---|---|---|---|
| EC | p+e− -> n+nue | fp(1−fn)fminus | fn(1−fp)(1−fminus) | zeta=mu_p+mu_e−mu_n |
| PC | n+e+ -> p+antinue | fn(1−fp)fplus | fp(1−fn)(1−fplus) | −zeta |
| neutron decay | n -> p+e−+antinue | fn(1−fp)(1−fminus) | (1−fn)fp fminus | −zeta |

All factors are evaluated on the actual on-shell event after boosts; initial and final baryon energies differ between channel parameterizations. Integrate both material products independently before testing log(j/chi)=−(E−mu_eq)/T. Never derive one branch by KMS to manufacture agreement. Forward/inverse rates use j(1−fnu) and chi fnu. At LTE both positive branches remain nonzero even when their net cancels; a stationary residual alone cannot establish normalization.

Per net forward event, changes (Nn,Np,Ne_net,Nnue,Nantinue) are EC(+1,−1,−1,+1,0), PC(−1,+1,+1,0,+1), decay(−1,+1,+1,0,+1). PC destroys one positron, decay creates one electron: identical NET charge changes do not make their particle populations interchangeable. Rmatter=(Rbeta+RPC−REC) where every R is a net event-density rate. dYe/dt=Rmatter/nB. Material Le plus emitted nue minus emitted antinue conserves; all three conserve charge and baryon number. Rest-inclusive material energy loses the net emitted neutrino energy, no extra Q or neutron-proton mass-difference heating term. Species-energy moments must independently reconstruct this identity, especially against large baryon rest-energy cancellation.

For common-T FD radiation, per-channel entropy production is the phase integral of (F−B)log(F/B), or netR*(mu_eq−mu_rad)/T when chemical affinities are constant. Check both ways, not only a rearranged identity. Empty radiation has no finite initial radiation-entropy derivative. Transparent material entropy may decrease: d(s/n)/dt=[−Q−zeta Rmatter]/(nT) after prescribed compression work cancels. A total second-law claim cannot be reconstructed from escaping number and energy moments alone.

## Absolute invariant normalization and current

Metric (+,−,−,−), gamma5=i gamma0 gamma1 gamma2 gamma3, sigma^{mu nu}=i[gamma^mu,gamma^nu]/2. For the n->p baryon current define the ORIENTED four-transfer q_Gamma=p_p−p_n. This is a current argument, distinct from a positive spatial transfer magnitude or a future-directed pair momentum. Use the frozen source convention

    Gamma^mu = GV gamma^mu − GA gamma^mu gamma5
               + i F2 sigma^{mu nu} (q_Gamma)_nu/(2 Mweak)
               − GP q_Gamma^mu gamma5/Mweak.
    Mweak=(MN+MP)/2, physical free q_Gamma²; amplitude GF Vud/sqrt(2) times currents.

Use Guo Eqs26–29 and47a–h with GV(0)=1, GA(0)=1.2723, F2(0)=3.706, MV=840, MA=1000, mpi=139.57 MeV and GP=2 Mweak² GA(q²)/(mpi²−q²). Full q² parameterization is pinned in `_weak_current`; preserve the denominator `(1−q²/(4Mweak²))*(1−q²/MV²)^2` and vector numerator `1−(1+3.706)q²/(4Mweak²)`. Define the reciprocal p->n vertex by Gamma_pn(q_pn)=DiracAdjoint[Gamma_np(−q_pn)], where q_pn=p_n−p_p=−q_Gamma; do not blindly reuse matrices at the same oriented transfer. Form factors use the common invariant q_Gamma²=q_pn², but equal squares do NOT identify the linear F2/GP transfer terms. In these source form-factor expressions, q² always denotes q_Gamma². Antineutrino/positron v-spinors and VA/AF crossing signs require literal-spinor versus invariant-polynomial forward gates. Recoil is exact phase space; a static point current or F2=0 is a distinct model switch, not a quadrature approximation.

Current orientation by reaction (the four-vectors p_n,p_p always label the physical neutron/proton, not generic initial/final slots):

| Process | Baryon current used | Transfer identities |
|---|---|---|
| nue+n -> p+e− (EC inverse absorption) | Gamma_np(q_Gamma) | q_Gamma=k_nue−p_e−=Q_abs |
| p+e− -> n+nue (EC emission) | Gamma_pn(q_pn) | q_pn=−q_Gamma=p_e−−k_nue |
| n+e+ -> p+antinue (PC emission) | Gamma_np(q_Gamma) | q_Gamma=p_e+−k_antinue=−Q_abs |
| antinue+p -> n+e+ (PC inverse absorption) | Gamma_pn(q_pn) | q_pn=k_antinue−p_e+=Q_abs |
| n -> p+e−+antinue (decay emission) | Gamma_np(q_Gamma) | q_Gamma=−(p_e−+k_antinue)=−Q_leptons |
| p+e−+antinue -> n (inverse decay) | Gamma_pn(q_pn) | q_pn=p_e−+k_antinue=Q_leptons |

These are labels for the already pinned amplitudes and on-shell conservation equations, not new current prescriptions. Literal reciprocal spinor/trace gates remain necessary in every channel; a scalar invariant formula must carry its own source crossing signs. Free physical and kinetic four-momenta coincide; that fact does not turn q_Gamma into its negative. The later mean-field adapter must preserve the same orientation before distinguishing physical from kinetic transfers.

Master event measure is the covariant `(2pi)^4 delta4 * sum_spins|M|² * product[d³p/((2pi)^3 2E)]`, with all initial occupations/final holes. No spin averages in this master sum; no extra helicity average. A coefficient at fixed incoming neutrino momentum has its `1/(2Enu)` flux leg, and conversion to physical rates supplies 1/HBAR and the needed HBARC^-3 for volume moments. Keep one convention from amplitude through phase, not ad hoc dimension restoration. Existing EC/PC scalar amplitudes average FOUR initial baryon/charged-lepton spins; their equivalent unaveraged amplitude is four times that value. Decay averages the neutron spin TWO and combines a spin-two parent density measure: no additional p/e spin or anti-helicity factor.

For the capture response geometry define Q_abs=k_nu−p_lepton=(omega,vector_Q_abs), with omega=Enu−El and positive spatial magnitude q=|vector_Q_abs|=|k−pl|. For nue absorption n->p, Q_abs=q_Gamma; for antinue absorption p->n, Q_abs=q_pn=−q_Gamma. Thus q in the following phase prefactor is a MAGNITUDE, not the oriented current four-vector. With El TOTAL and pe=sqrt(El²−me²), the accepted differential prefactor for the avg-four scalar amplitude A is

    d(j or chi)/(dEl dcos) = pe/[32pi³ q Enu HBAR]
                             * integral dEa <A>phi * appropriate occupations.

The invariant baryon shell phase is dEa dphi/(16pi²q); outgoing lepton phase pe dEl dcos/(8pi²); fixed nu leg1/(2Enu). No extra target degeneracy, phi2pi or neutrino spin factor. At q=0 the coordinate reduction is singular and must use a mathematically derived limit/alternate coordinates or raise unresolved-coordinate error, not a zero-opacity claim.

One-helicity DOS D(E)=E²/[2pi² HBARC³] fm^-3 MeV^-1: R=integral D[j(1−f)−chi f]dE; Q adds E. j/chi units s^-1 per occupation; differential units s^-1 MeV^-1 per cosine. True inverse absorption length chi/c in cm^-1 (c=2.99792458e10 cm/s); km^-1 is cm^-1 times1e5. Relaxation length uses (j+chi)/c and must be explicitly named. Transport opacity is NOT automatically either coefficient, nor can an angular transport weight be inferred from these isotropic total moments. cgs energy multiplies MeV by1.602176634e-6 erg and fm^-3 by1e39 cm^-3. No arbitrary km label.

## Support, endpoints and numerical admission

Capture support solves the exact unequal-mass positive-energy shell before FD quadrature, retaining spacelike, applicable timelike and lightlike limiting branches. PC physical free n/e+ inverse support is strictly spacelike; do not transplant the EC n->p support ordering. A stationary proton anti-capture threshold1.8060662080187608 MeV is a vacuum target-rest witness, NOT a finite-T cutoff: moving targets permit below-threshold response. EC initial incoming p/e threshold depends on proton momentum and incident angle, not one fixed lepton cutoff.

For decay, define the FUTURE-directed lepton-pair momentum Q_leptons=kanti+pe=p_n−p_p=−q_Gamma. It is Q_leptons that is future timelike, with me²<=Q_leptons²=q_Gamma²<=(MN−MP)². The pinned decay current still takes q_Gamma=p_p−p_n, which is past-directed on this decay support. Substituting Q_leptons into the same n->p vertex solely because the squares agree would reverse linear transfer terms and is forbidden. Neutron-rest anti endpoint E0=.7820077025790402MeV is not a lab cutoff. With parent rapidity eta, lab Eanti=gamma E*(1+beta cos*); for E>E0 parent pmin=MN sinh(log(E/E0)). Conditional phase must resolve the moving lower parent bound/cone. The frozen alternative reduction has rest spin-averaged dGamma*/(dEanti* dEe*)=Aavg2/(64pi³ MN HBAR), parent spin-two p²dp/(pi² HBARC³) and MN/En time dilation. Use actual electron/proton blocking after boosts. Inverse is the same covariant measure with swapped occupancies, not a newly time-dilated inverse proper lifetime.

Endpoint maps, Fermi-edge panels, log occupations, separately nonnegative integration and exact angular support precede refinement. Number/energy must refine independently (selected target3e-6 relative when resolved, absolute gross-scaled cancellation gates preregistered). Compare different phase coordinates and literal traces, not just increased orders within one shared formula. Tail and angular/refinement controls have separate metadata; a combined global coarse/fine test is not an isolated tail bound. Numerical limits do not authorize nuclear collision widths, arbitrary smearing or triangle-threshold fits.

Representation is a tagged positive quantity: finite log with ordinary value=None for genuine floating underflow/overflow; exactzero only with explicit physical empty-support or absent-occupation reason; invalid/nonfinite input and unresolved positive integration raise. A finite-T complete positive channel cannot become exactzero just because all ordinary nodes underflow. Retain extreme-soft finite logs without promising their relative quadrature accuracy. Reject false-zero integrated branches, NaN, Boolean numerics, mismatched radiation temperatures, stale result/source/control hashes and incomplete chunks before aggregation. No cache in first public reference; any future cache fingerprints full state/model/radiation/control/code and actual result bytes, not flavor/order alone.

Initial candidate qualification should be deliberately narrow: actual tested free-bath T1–3MeV,n1e−7–1e−6fm^-3, compositions chosen within tested source guards (including actual electron |mu/T|<=3 and dilute baryon eta), plus explicit-chemical vacuum/dilute anchors. Numerical request bounds must be preregistered BEFORE new calls; the rectangle is not a uniform accuracy certificate. Spectral initial qualification is the exact T2/n1e−7/Ye.3 bath and selected energies/total reconstruction, not all admitted positive energies. Broader degeneracy/lowT/cold support gets separate evidence and cannot inherit this gate.

## Audit: reuse mathematics, refactor execution

`SOURCE_INVENTORY.json` records literal imports/signatures/top-level calls and exact source hashes without importing any scientific module. It intentionally distinguishes a static inventory from an executed transitive-dependency proof. Main findings:

- EC total owns scalar and literal-spinor two-body event references, fresh `matched` roots and source checks; importing executes verification/dynamic file loads. Reuse amplitudes/phase formulas, move constants/bath/current into immutable package records; replace ROOT-relative loaders.
- PC independent total reference currently lives under `review_pc_moments`, a reviewer-owned location. Its vectorized Guo Eq47 polynomial is a valuable independent test oracle; production should derive/port the author event method and preserve this separate reference in tests, rather than make a review folder a runtime dependency.
- Hot beta imports generic `beta`/`kernel` after sys.path changes. It inherits current, tensor, baryon and lepton modules transitively. Conditional spectra separately restore generic aliases, but unique helper modules and `sys.dont_write_bytecode` still modify process state. Package-relative explicit imports must replace this mechanism; do not execute any study/collector driver.
- Fixed-bath hot EC/PC spectral modules duplicate geometry but have physically different masses/ordering/occupancy and crossed currents. Their `matched()` exact equality guards are useful evidence for that bath, not a general-state EOS API. Avoid patching masses/globals or replacing one scalar and assuming the rest matches.
- Current module imports `vacuum_cc` through sys.path and vacuum wrappers derive weak M from their channel masses. Separate free bare current normalization from on-shell mass records now, so a future Hartree adapter cannot accidentally feed mD into Mweak.
- Source coupling's `three_source_cooling_element/worker.py` hardcodes NB=1e−7; do not expose it as a generic density API. Process-isolated source prototypes repair and pin admission but are not a security sandbox or portable installed backend. A package client should have no mutable alias/global cfg to isolate in the first place.

Later GM1 binding must explicitly provide common939 bare-Hamiltonian fields mD,U and total/kinetic chemical potentials, consistently ORIENTED physical versus kinetic current transfers q_Gamma,physical and q_Gamma,star (and their negatives for the reciprocal channel), and independently matched thermodynamics. This is a DIFFERENT model: cannot pass free physical unequal masses or reuse the bare weak mass by accident; no in-medium CVC reduction without a stated prescription. Require the accepted Hartree forward/stability gates and maintain external native GM1 reference OPEN. No density-dependent rearrangement or arbitrary effective-mass substitution into fixed-mass CPU partials.

## Required gates before port acceptance

1. Exact package origin/minimum dependency API audit, no campaign directories, file IO at import, sys.path/module aliases, mutable mass/current settings or scientific driver execution. Real package import and existing cold regression tests; minimum runtime/CI remain separately open until executed.
2. Full-current literal Dirac traces versus source invariant VV/VA/AA/VF/AF/FF/AP/PP including reciprocal and anti crossing; point-VA limit, F2/GP switches, hermiticity/positive contractions and on-shell residuals. Physical weak mass and Q provenance explicit.
3. Vacuum two-body target-rest threshold/absolute sigma, independent dilute moving-target n*c<sigma v> (not a target-rest surrogate in a thermal bath); heavy-mass point-current limit uses consistently deformed EOS and phase. Beta proper lifetime/time-dilation and Doppler number/energy, not measured-lifetime calibration.
4. Fresh density/chemical/root forward values BEFORE source responses, particle e± versus net charge, paired electron normalization; no hidden beta/neutrality. Explicit-chemical and matched factory outputs must agree at the same full state, not nearby mu.
5. Saved three-channel R/Q and spectra below plus independent fresh holdouts preregistered before implementation validation. Each F/B number and energy refines at unchanged gates, isolated tail tests, independent coordinates; LTE cancellation alone insufficient. Cross-check Stage B DOS reconstruction against Stage A actual event totals, including beta above E0.
6. Signed species/rest-inclusive energy/Le/charge/baryon ledgers, both-affinity direct entropy identity. Normalize energy against thermal/escaping/species source scales, never939MeV rest energy; root/cancelled sources use gross rates and absolute Ye tolerances.
7. Negative input/domain/branch/timeout tests and proven underflow-versus-empty behavior, finite-T integrated false-zero adversaries, no incomplete result admission. A failed quadrature returns explicit incomplete diagnostic, not a success flag or physicalzero.
8. Different independent physics/numerical review and separate packaging/runtime scope. No AD/Hessian, full neutrino inventory, correlated/paired/modified Urca, muon weak sector, nuclear pasta, radiative/Coulomb or physical-uncertainty claim from shared-model method agreement.

## Fits and preserved failures

Default detailed quadrature remains the reference. Optional positive EC/PC number and energy fits are a later separate convenience layer: frozen coefficients/basis, actual physical source identity and complete passing heldout bytes must be verified; no arbitrary client JSON `qualified=True`. Only fixed-model empty-radiation T1–3,n1e−7–1e−6,Y.2–.5 intersect actual |mu_e/T|<=3 and baryon eta<=−5 has selected evidence. Positive fit accuracy does not certify net signs, composition roots, spectra, conditional FD inverse moments, derivatives or EOS-family uncertainty. Near cancellation use actual detailed quadrature; it must visibly compute, not reuse an unrelated saved bath. Keep fixed-state scattering fits entirely separate from CC fits.

Retain the original cold T.01 capture quadrature failures and mapped-edge repair as numerical evidence, not thermal smoothing; original strictly-positive wrapper underflow failure in forbidden bands; PC ten coarse spectral reconstruction failures and EC total31 coarse failures; original PC121PASS/1false-zero FAIL plus separate568-pass repair; fit initial191PASS/2admission FAIL plus separate160-pass repair; stationary-allowed20 MODELFAIL (not numerical error); fit signed scattering failures remain a different product. Counts are related assertions, not independent physical benchmarks. No tolerance/source/fit retuning is authorized by this contract.

`SAVED_REGRESSIONS.json` copies actual immutable JSON paths/rows and full source identities, not recomputed physics. The audit command initially assumed beta `results.json`; that filename does not exist. It stopped without writes, then read actual `pilot_results.json` and `total_results.json`. This is a retained audit-path correction, not a scientific calculation or failed physics gate. The initial audit helper also stopped before output generation on two incorrect guessed dependency names; exact initial helper and diagnostic are preserved beside the corrected read-only audit.

Primary conventions: [Guo et al. Eqs26–29,47](https://arxiv.org/html/2006.12051v2); [Strumia–Vissani full vacuum current/crossing](https://arxiv.org/html/astro-ph/0302055). These are the already web-verified source records pinned in `muon_full_current/SOURCES.json`; no new external-code or experimental comparison was run. See the pinned independent reviews for exact accepted numerical scope rather than stale author-note review-pending wording.
