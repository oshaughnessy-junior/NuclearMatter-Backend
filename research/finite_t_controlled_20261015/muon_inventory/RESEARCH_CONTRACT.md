# Finite-temperature muon inventory and implementation contract

15 October 2026. Research design and cheap EOS/stoichiometry probes; no new muon weak kernel, prefactor, production qualification or astrophysical simulation. This leaf owns only `muon_inventory/`. Frozen electron kernels and existing cooling prescriptions remain intact. Independent review is required before implementing or publishing the proposed weak extension.

## Primary-source audit and why the inventory must expand

| Primary source, verified online | Specific use and limitation |
| --- | --- |
| [Roberts & Reddy 2017, PRC95 045807](https://arxiv.org/html/1612.02764v2) | Relativistic charged-current response with mean fields and weak magnetism; Fermi-golden-rule/thermal-field-theory comparison. Use for normalization and response conventions. Its electron calculation alone does not establish massive-muon corrections. |
| [Guo et al. 2020, Charged-Current Muonic Reactions](https://arxiv.org/html/2006.12051v2) | Full muonic kinematics, weak magnetism, pseudoscalar and momentum-dependent form factors. Inverse muon decay and electron-target reactions matter at low energies in their PNS states. Explicitly linked [author code](https://git.gsi.de/nucastro_archive/supernovamuonicrates) offers independent 2D and 4D formulations. Landing page accessible; code not retrieved or executed here. |
| [Sugiura et al. 2022, PTEP113E01](https://academic.oup.com/ptep/article/2022/11/113E01/6680255) | Table1 and appendices distinguish flavor exchange, decay/inverse decay, mixed-lepton annihilation and semileptonic channels. Low-energy opacity can be dominated by leptonic channels in their late-PNS conditions. This motivates a coupled flavor kernel, with no universal claim of dominance. |
| [Ng et al., Weakhub](https://arxiv.org/html/2309.03526v2) | Useful comparison of full kinematics, corrections and kernel-based interactions; distinguish raw absorption from stimulated-absorption coefficients. EOS potentials/effective masses must match before comparing results. |
| [Gieg et al., first muonic BNS simulations](https://arxiv.org/abs/2409.04420) | Independent evolving net muon abundance with leakage and elastic-approximation rates; application motivation, not our spectral reference. |
| [Ng et al., five-species transport](https://arxiv.org/html/2411.19178v2) | Demonstrates coupled matter composition, muonic beta processes and trapped flavor hierarchy in BNS calculations. Their cooling/ejecta findings depend on the selected setup. |
| [Gieg et al. 2026, consistent treatment](https://arxiv.org/html/2604.14225v1) | Independent electron/muon fractions, full-kinematics beta reactions and decay pair kernels. Their revised energy-integrated pair treatment and equilibration prescription motivate testing kernel closure before gray fits. Application differences from earlier studies discourage adopting a universal muon cooling correction. |
| [PDG2025 muon-decay review](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-muon-decay-params.pdf) | Vacuum V−A decay spectrum, electron endpoint and massless-electron total decay normalization, Eqs57.1–57.3. Use bare tree-level predictions before radiative/finite-mass corrections or measured-lifetime comparison. |

The equations and contract below are our proposed bookkeeping and validation design, derived from conservation and occupation factors. They are not a transcription of any transport code.

## State, units and equilibrium

Use independent net fractions `Ye=(ne−−ne+)/nB`, `Ymu=(nmu−−nmu+)/nB`. For uniform nucleonic matter only, `Yp=Ye+Ymu`, `nn=nB(1−Yp)`, `np=nB Yp`. Enforce `0<=Yp<=1`; net signed lepton fractions need not each be positive mathematically. Bound nuclei, hyperons or charged condensates require explicit general charge accounting.

Each charged lepton has its own immutable mass: `me=0.51099895069`, `mmu=105.6583755 MeV`. With total chemical potential including rest mass,

`f_l−(p)=[exp((sqrt(p²+ml²)−mu_l)/T)+1]^-1`,
`f_l+(p)=[exp((sqrt(p²+ml²)+mu_l)/T)+1]^-1`.

Spin degeneracy is two. Existing `thermodynamics/lepton_reference.py` supplies these gases and charge inversion with an explicit mass argument. Pair equilibrium is an assumption; it permits nonzero pairs at zero net fraction. There is no sharp finite-T muon appearance threshold. The cold onset `mu_l=ml` applies to a T→0 equilibrium construction, not arbitrary prescribed Ye/Ymu.

Return `(T,nB,Ye,Ymu,Yp)`, species densities, total chemical potentials, effective masses/potentials, pressure, entropy, total energy, susceptibilities and domain diagnostics. Never infer muon abundance from `mu_e` unless the requested equilibrium closure actually equates the lepton chemical potentials. Preserve the baryon offset convention in both energy and response. Common NR kinetic mass plus independent rest offsets is a controlled free-gas model; it is not full RMF kinematics.

Define the effective equilibrium radiation chemical potentials `zeta_e=mu_e+mu_p−mu_n`, `zeta_mu=mu_mu+mu_p−mu_n`. Trapped beta equilibrium is `mu_n+mu_nu_l=mu_p+mu_l`, so `mu_nu_l=zeta_l`. With neutrino pair equilibrium, `mu_antinu_l=−zeta_l`. Decay equilibrium additionally gives `mu_mu=mu_e+mu_numu−mu_nue`, consistent with both beta equations.

Evolve or constrain independently `YLe=Ye+(n_nue−n_anue)/nB` and `YLmu=Ymu+(n_numu−n_anumu)/nB`, absent oscillations. Four explicit electron/muon neutrino species are mandatory. A tau-pair aggregate may remain only with declared degeneracy two; do not retain a four-species heavy-flavor multiplicity after splitting out muons.

Transparent kinetic steady state means solving `dYe/dt=dYmu/dt=0` with the actual empty or irradiated radiation distributions and selected complete inventory. It generally cannot be replaced by `zeta_e=zeta_mu=0`. Pointwise inverse KMS is a relation between forward and inverse coefficients for an equilibrium bath, not a proof of transparent kinetic equilibrium.

Units: T, momenta, masses and total chemical potentials in MeV, n in fm^-3, energy/pressure in MeV fm^-3, kB=1, time in seconds. Define j and chi as s^-1 in `df/dt=j(1−f)−chi f`. If returning cm^-1 opacities divide by c at a named boundary. One-helicity neutrino density of states is `D(E)=E²/[2 pi²(hbar c)³]` fm^-3 MeV^-1. Integrated number source `∫D C dE` is fm^-3 s^-1; energy source `∫E D C dE` is MeV fm^-3 s^-1. CGS conversion multiplies by `10^39` and additionally `1.602176634e−6 erg/MeV` for energy. Never apply charged-lepton spin two to neutrino DOS.

## Explicit reaction ledger

All arrows below include the inverse; store a unique process ID so crossing is not double counted. For `l=e,mu`:

| Process | Forward change of net material fractions per event /nB |
| --- | --- |
| `l−+p → n+nu_l` | `ΔYl=−1/nB`, `ΔYp=−1/nB` |
| `l++n → p+antinu_l` | `ΔYl=+1/nB`, `ΔYp=+1/nB` |
| `n → p+l−+antinu_l` | `ΔYl=+1/nB`, `ΔYp=+1/nB` |
| `p → n+l++nu_l` | `ΔYl=−1/nB`, `ΔYp=−1/nB` |

The last crossed proton branch must have an explicit support decision; do not assume it is active in vacuum. Vacuum neutron muon beta decay is forbidden because `mn−mp<mmu`. A dense, occupied medium with specified dispersions must be tested from its own energy/momentum support, not a vacuum switch. Modified Urca spectators, pairing and strong correlations are separate extensions.

The purely leptonic CC inventory is:

- `mu− ↔ e−+antinue+numu` and `mu+ ↔ e++nue+antinumu`, including 3→1 inverse decay.
- `nue+mu− ↔ numu+e−`; `antinue+mu+ ↔ antinumu+e+`.
- `antinumu+mu− ↔ antinue+e−`; `numu+mu+ ↔ nue+e+`.
- `e−+mu+ ↔ nue+antinumu`; `e++mu− ↔ antinue+numu`.

These are the physical crossings of the same four-fermion flavor interaction, with their own phase spaces and blocking. For mu− decay, `ΔYe=+1/nB`, `ΔYmu=−1/nB`, `ΔYp=0`; the emitted antinue and numu conserve the two *total* family lepton numbers separately. Implementing decay as a flavor-blind cooling scalar breaks this ledger.

Also track the separate NC/pair inventory: neutrino scattering on e± and mu±, same-species `l−+l+↔nu_a+antinu_a`, baryon bremsstrahlung, and neutrino flavor-pair conversion where selected. Same-flavor charged-lepton annihilation may have interfering CC and NC amplitudes; do not add two scalar rates without the interference derivation. Pair creation of charged leptons in the EOS assumes fast electromagnetic equilibration; it does not substitute for the weak flavor kernels.

## Kernel and energy contracts

For semileptonic absorption `nu_l+n→l−+p`, response transfer into baryons is `omega=E_nu−E_l`, `q=|k_nu−p_l|`. For inverse crossed neutron decay `antinu_l+l−+p→n`, it is `omega=E_nu+E_l`, `q=|k_nu+p_l|`. Declare transition direction, lepton occupancy versus vacancy and total energy offsets explicitly. Use exact `p_l=sqrt(E_l²−ml²)`; a mass substitution inside an electron rate fit is not sufficient.

For a thermal matter bath with fixed composition, each semileptonic flavor obeys `j/chi=exp[−(E−zeta_l)/T]` for neutrinos and `exp[−(E+zeta_l)/T]` for antineutrinos. Verify in log space before integrating tails. For mixed-lepton interactions, the reverse kernel depends on the other neutrino distribution; a standalone scalar KMS opacity is valid only after explicitly supplying that partner bath.

General event collision factors are `forward=product(f_initial) product(1−f_final)`, and reverse with initial/final exchanged. Their ratio in a common thermal state is `exp[(sum mu_initial−sum mu_final)/T]` on the energy delta surface. This provides a process-independent sign and equilibrium test.

Use the same event quadrature for signed species and energy moments. At fixed nB, `nB du/dt=−sum_a Qrad_a` with total material energy including changing lepton rest energy. Composition terms belong in the EOS inversion, not a second manually added muon Q heating term. With prescribed density, `du/dt=P/nB² dnB/dt−sum Qrad/nB`.

For the existing free mixture, define `Ai=mu_i−T (dmu_i/dT)_ni`. Then `uT=sum_i Cv_i/nB` (plus photons if selected), `uYe=Ap−An+Ae`, `uYmu=Ap−An+Amu`. These are energy derivatives, whereas chemical affinities use chemical potentials directly. An interacting baryon EOS must supply its own composition derivatives; do not transplant the ideal-gas A formula without its derivation.

## Reuse and implementation sequence

1. **Two-lepton state leaf first.** Reuse exact charged-lepton quadrature and separate baryon species roots. The existing `nr_baryon_eos.matter_state(T,nn,np)` hardwires electron neutrality `ne=np`; it cannot represent a muon-bearing state unchanged. Build an additive factory returning both gases and an independently checked thermodynamic Jacobian.
2. **Explicit species API.** Proposed immutable `LeptonSpec(name,mass,flavor,charge)` and `MatterState(T,nB,Ye,Ymu,baryon_model,energy_zero)`; kernels require a spec and explicit radiation species. Never monkeypatch global `ME`: allowed capture, endpoint mapping and crossed decay currently import electron-mass constants. Reuse their numerical architecture only after making a new fully explicit mass-aware leaf with electron regression.
3. **Semileptonic muon reference.** First a declared LO static-current/free-NR recoil benchmark, then independent relativistic response/current contraction using Roberts/Reddy and Guo conventions. Muon pseudoscalar, weak magnetism and q² form-factor effects require derivation and distinct switches; NR response support alone does not qualify the current at muon energies. Compare matched forward values with author code before sensitivities.
4. **Leptonic flavor kernels in parallel priority.** Start vacuum muon decay/boost and inverse kernel, then the crossed 2↔2 processes; add thermal blocking and non-LTE partner distributions. Couple in a homogeneous spectral box before gray integration. Neither scalar muon cooling nor forced FD partner radiation is a complete box model.
5. **Moments, then controlled applications and fits.** Tabulate each species' number/energy moments and coarse-graining error separately; test a prescribed fluid element and slab before a disk trajectory or fixed spherical cooling background. Fit accepted detailed outputs over a declared domain with held-out spectra, number and energy errors, not just luminosity.

## Independent release gates

- EOS: direct momentum quadrature, CP symmetry, net-zero pair plasma, T→0 analytic muon Fermi gas, fixed-charge heat capacity, and a two-composition finite-difference Jacobian. Check no `Ye=Yp` assumption survives. Include trapped states with unequal `mu_nue` and `mu_numu`.
- Support: at a free neutron at rest, `E_nu,thr=[(mp+mmu)²−mn²]/(2mn)`; antineutrino/proton target uses `[(mn+mmu)²−mp²]/(2mp)`. These include recoil and differ from static `mmu−(mn−mp)`. Reconstruct support independently from four-momentum conservation and compare near-threshold roots. Dense FD support needs its own calculation.
- Electron target: inverse muon decay `numu+e−→mu−+nue` on a rest electron has threshold `(mmu²−me²)/(2me)`; this large vacuum threshold is not a dense electron-plasma threshold.
- Vacuum decay: PDG Eq57.3 tree-level massless-electron total `GF² mmu^5/(192 pi³ hbar)` and normalized electron spectrum `2x²(3−2x)`, `x=2Ee/mmu`; finite electron mass endpoint `(mmu²+me²)/(2mmu)`. Test energy-sharing sum and moving-muon time dilation. Do not tune bare GF to the measured lifetime while omitting radiative corrections.
- KMS and inverse channels: random allowed phase-space points, both signs of radiation chemical potential, explicit zero/finite occupation partners, log-tail representability and degenerate lepton blocking. Compare all named channels separately before totals.
- Conservation: every reaction conserves baryon number, charge and both family lepton numbers. In a closed spectral box test total energy, both trapped lepton ledgers and entropy production; in an empty-radiation element test signed escaping moments and independent final EOS inversion. Exact bookkeeping is not a time-discretization convergence test.
- Cold cooling: retain the existing source-labelled electron/muon DU emissivity and pairing qualifications. The initial campaign audit pins `core_nu.py` to NMB snapshot25e68182; the available earlier native staging file confirms T^6 scalar amplitudes with triangle flags, not arbitrary-degeneracy spectra. Its muon flag requires the electron triangle as well. Do not extend that ordering assumption to independent prescribed Ye/Ymu. No fresh current-source regression was run here; raw-GitHub fetch of the private pinned file was unavailable. A matched degenerate equilibrium comparison must reproduce constants, effective masses, pairing, threshold convention and neutrino-free assumptions before claiming parity.

`contract_probes.py`/`contract_results.json` provide cheap conservation and EOS examples, plus vacuum support numbers. These validate bookkeeping and existing lepton reuse only. They do not qualify any new weak rate or astrophysical equilibrium.

Measured examples at prescribed `nB=.16 fm^-3, Ye=.06, Ymu=.02, Yp=.08`:

| T [MeV] | mu_e [MeV] | mu_mu [MeV] | n_antimu [fm^-3] | n_mu− at zero net muon density [fm^-3] |
| --- | --- | --- | --- | --- |
| 5 | 129.110075 | 137.747018 | 1.57844e−25 | 1.45473e−13 |
| 30 | 107.197517 | 99.731160 | 4.99440e−6 | 1.37645e−4 |

The unequal chemical potentials are allowed because these are prescribed-composition states. At T30, `mu_mu<mmu` still gives a substantial specified net muon abundance. The zero-net column counts particles of one sign; the antiparticle density equals it. None of these states was solved for beta balance.

Additional current-contraction gate: [Roberts/Reddy §III](https://arxiv.org/html/1612.02764v2) warns that unequal neutron/proton masses and self-energies invalidate a naive charged-current vector-current-conservation reduction. Keep all required response components. [Guo et al. §IV](https://arxiv.org/html/2006.12051v2) finds pseudoscalar interference proportional to charged-lepton mass squared in the stated approximation; its electron suppression cannot justify dropping it for muons. These are physical extension gates beyond replacing a phase-space mass.

Recommended return contract is `ProcessResult(process_id, species_stoichiometry, number_forward, number_reverse, number_net, radiation_number_by_species, radiation_energy_by_species, matter_energy_source, log_spectra, representability, quadrature_error, model_switches, provenance)`. Distinguish event rates from positive linear transfer pieces; the latter are not necessarily gross `j(1−f)` interaction counts. Number/energy units and source signs must be machine-readable. Omitted channels are returned in an explicit inventory manifest, with a justification, rather than hidden in a total.
