# Full-energy charged-current moments in a fixed FD bath

Research checkpoint, 2026-10-15. Independent review is required. This leaf integrates the reviewed `quantum_slab` coefficients over neutrino energy, retaining their exact numerical kernels; it introduces no emulator and evolves no fluid. The dilute study passes 602 author gates; the initial denser grid fails 14 of 602 gates, and targeted refinement supplies 70 passing replacement/remainder checks while preserving those failures. See the qualified results below.

## Physics and source contract

Use the matched free nonrelativistic FD nucleon EOS and relativistic ideal electron/positron neutrality solve already supplied by `nr_baryon_eos`. The charged-current inventory is electron capture on protons and its inverse (νe), positron capture on neutrons and its inverse (anti-νe), and crossed neutron decay/inverse (anti-νe). `quantum_slab.fixed_bath.coefficients` explicitly composes mapped capture, its reviewed leaf-local collapsed-panel guard, and crossed decay; the stale combined antineutrino helper is not used. Frozen upstream SHA256 hashes are recorded in the checkpoint.

This is the same leading allowed-current free-NR ansatz reviewed in the upstream notes. Quadratic-dispersion recoil and target FD blocking are included. Interactions, nuclei, muons, forbidden currents, weak magnetism, relativistic nucleon/current corrections, neutral-current scattering and pair processes are absent. Results are not a complete astrophysical neutrino opacity or a calibration of the free-gas ansatz.

For each species the occupation collision term is C=j(1−f)−χf, with j and χ in s⁻¹. Nuclear/electron occupations are already included in those coefficients. The affinity is μνe=μp+μe−μn and μanti=−μνe. KMS gives log(j/χ)=−(E−μν)/T, and feq=j/(j+χ). Chemical potentials include the rest-offset convention consistently with the matched EOS. Neither beta equilibrium nor μν=0 is imposed.

The one-helicity density of states is D E²dE, D=1/[2π²(ℏc)³] in fm⁻³ MeV⁻³. The empty-radiation volume number emissivity is ∫D E²j dE, and energy emissivity is ∫D E³j dE. Multiply by 10³⁹ for cm⁻³ and by 1.602176634×10⁻⁶ to convert MeV to erg. These are source emissivities into fν=0, not the net exchange of an irradiated or trapped radiation field.

## Slab quantities and signs

A prescribed 1 km steady slab has left incoming occupation FD(Tν=4 MeV, μν=+3 MeV), right incoming occupation 0.35 FD(Tν=6 MeV, μν=−2 MeV), for each flavor. Each boundary specifies an isotropic incoming hemisphere; outgoing flux is a positive outward magnitude at each face. A one-face incident spectral number flux is c D E² f/4, with the fm→cm density conversion above. The upstream reviewed analytic angle solution uses relaxation optical depth τ=(j+χ)L/c; χ/c alone is the true absorption coefficient and is not interchangeable with τ.

Let a=1/2−E3(τ), t=E3(τ), A=c D E²×10³⁹. Outgoing-left/right spectra are A[feq a+f_right/left t]/2. Two positive integrated exchange terms are retained separately: slab emission A feq a, and absorbed boundary irradiation A(f_left+f_right)a/2. Their difference is the radiation number gain, equal to outgoing minus incoming summed over both faces. Multiplying each integrand by E gives the energy ledger. The material energy source is the negative radiation gain; positive energy transfer into the material has a positive sign. This is total exchanged neutrino energy, not a temperature-rise rate; composition/chemical energy must be accounted for before interpreting it as thermal heating. Integrating emission, absorption, incoming and outgoing components separately avoids using cancellation of large boundary fluxes as a quadrature error estimate.

For νe electron-lepton number is +1 and for anti-νe it is −1. Material electron-lepton exchange per unit area is −(radiation νe number gain − radiation anti-νe number gain). Total energy exchange sums both species with the same energy sign. These are fixed-bath counters, not a self-consistent changing Ye/T or hydro evolution.

## Numerical design

`integrated.py` evaluates physical coefficients at every Gauss node, retaining raw node and panel contributions. `run_study.py` independently refines outer energy orders 8, 12 and 20 while holding the internal energy/angle orders at 48 and internal tail at 40T. Explicit panels resolve E→0, the nominal 0.7823 and 1.2933 MeV static scales, the emissivity bulk, and tails to 320 MeV. Finite-mass thermal reactions need not retain the sharp static thresholds; the panels resolve those scales without imposing an artificial support cut.

Outer high-energy cutoffs 120, 180 and 240 MeV are each compared with the 320 MeV integral, independently of the energy-node refinement. Contributions below 0.1 MeV remain included with positive interior nodes; the E²/E³ measure vanishes at the endpoint. Each energy panel supplies an emissivity-weighted representative node for internal order 48→72 and tail 40→50 refinement, including low energies, threshold regions and high-energy tails. This is a sampled internal-grid qualification, not a global error proof. No small-affinity or chemical-potential-sign shortcut is used. Signed-μ incoming FD integrals are also checked against independent adaptive integration.

Independent face-by-face number/energy accounting is compared with positive emission-minus-absorption exchange, using a gross exchange scale. Invalid finite/positive quadrature controls, invalid lengths, flavors and energy-panel definitions are rejected before expensive work; the separate control suite records these regressions.

The source uses existing Python/NumPy/SciPy only. Run `python outputs/nmb-finite-t-20261015/quantum_energy_moments/run_study.py dilute` (or `moderate`). Every completed integration writes a partial checkpoint, so failed or interrupted grids remain inspectable. The author grid is deliberately modest and CPU bounded.

## Remaining physical and numerical gates

Independent review must verify the integrated normalization, flavor signs, error scales and held-out quadratures. A heavy-static **energy-integrated complete-inventory** comparison was attempted and failed the existing wrapper’s strict positive-coefficient/underflow contract; it remains open. `static_inventory_failure.json` preserves the blocker. Earlier selected-energy static checks do not close it. There is no interacting-EOS comparison, no angular transport closure, no neutrino scattering redistribution or fluid feedback. Do not promote these source terms to material evolution until the integrated moments are independently accepted and the chosen controlled ansatz is stated.


## Qualified numerical results

The two T=2 MeV baths have (nn,np)=(7×10⁻⁵,3×10⁻⁵) and (0.002,0.001) fm⁻³. Their electron chemical potentials are 18.28198 and 60.83299 MeV; νe affinities are +15.27304 and +57.64326 MeV, with the opposite anti-νe signs. Thus the experiment explicitly exercises strongly positive and negative affinities.

| Bath | Species | Empty-radiation number emission [cm⁻³ s⁻¹] | Energy emission [erg cm⁻³ s⁻¹] | Material energy exchange [erg cm⁻² s⁻¹] | Material electron-lepton exchange [cm⁻² s⁻¹] |
|---|---|---:|---:|---:|---:|
| dilute | nue | 2.59636231e+38 | 6.98168201e+33 | -4.10680611e+38 | -1.71739599e+43 |
| dilute | antinue | 1.40753537e+32 | 2.40534402e+27 | 6.31150930e+37 | -1.66890675e+42 |
| moderate | nue | 1.95016854e+42 | 1.49640273e+38 | -4.46534431e+41 | -6.33838654e+45 |
| moderate | antinue | 1.90962449e+24 | 3.26957759e+19 | 3.02519814e+38 | -1.04202829e+43 |

The area exchanges above use the prescribed two-face irradiation and 1 km thickness. They are not the volume empty-radiation emissivities multiplied by thickness. The complete positive incoming/outgoing/emission/absorbed pieces are in `summary.json` and raw result files. Divide an area material source by 10⁵ cm only when a slab-averaged volume source is desired.

For the dilute bath, all 602 gates pass. Maximum outer8→20 and outer12→20 discrepancies are 1.06×10⁻⁸ and 2.66×10⁻¹¹. Cutoffs120/180/240→320 MeV change the largest component by 7.72×10⁻⁶ / 1.14×10⁻⁹ / 1.20×10⁻¹³. Weighted-node internal refinement changes any component by at most 3.73×10⁻¹⁴. The two channel-resolved antineutrino emission moments sum to the directly integrated complete inventory; `dilute_channel_audit.json` records the capture and crossed-decay pieces separately.

The moderate bath’s initial coarse orders fail **14** νe emission/face-flux gates: the worst relative difference is 1.76×10⁻⁴. These are retained in `moderate_results.json`, whose `passed` field remains false. No gate tolerance was loosened. `refine_moderate.py` independently evaluates orders32 and48 on [24,40], [40,70] and [70,120] MeV, around the degenerate spectral edge. All 42 local component comparisons pass, worst 5.01×10⁻¹². The other energy panels independently agree across the original8/12→20 ladders, supplying 28 further passing checks. The final correction to the original20-node integrated components is at most 5.14×10⁻⁸. The final moderate νe values in the table use these refined panels; anti-νe retains its converged20-node result. The original failed coarse grids are not reclassified as passing.

Actual numerical cost was approximately 132.9 s for the dilute full-energy ladders plus 13.3 s for its separate channel audit; the moderate ladders plus targeted refinement cost 167.8 s. Eleven invalid-input control checks pass. An early author reporting error accessed `Matter.T` rather than `Matter.neutron.T` after the first three completed integrals. Its partial grids and failure are preserved; those unchanged completed quadratures were explicitly resumed. Fresh reproduction runs do not silently cache old partial files; `--resume` is an explicit option for an interrupted run with unchanged physics and controls. The script now returns a failing exit status when numerical gates fail.

The heavy-M complete-inventory experiment did not produce an accepted number or energy result: `static_inventory.py` reproduces the positive-coefficient underflow failure, and the failure diagnostic records its location and log-channel coefficients. No floor, omitted channel or ad hoc interpolation was introduced. A log-domain or exact-zero-safe emission-only interface needs separate review before that gate can close.
