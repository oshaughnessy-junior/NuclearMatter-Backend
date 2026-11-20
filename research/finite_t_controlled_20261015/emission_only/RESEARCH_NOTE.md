# Log-domain complete antineutrino emission-only reference

This selected numerical reference addresses the heavy-common-mass integrated-inventory blocker without requiring ordinarily representable absorption/emission coefficients. It changes no upstream kernel. The named antineutrino inventory is **positron capture on neutrons plus crossed neutron decay**; no other astrophysical channels are implied by “complete.” Physical quadrature evaluates both channels at every positive energy node and combines them with logsumexp. There are no coefficient floors, invented support holes, or discarded channels.

## Physical contract and normalization

Use the reviewed leading allowed-current free nonrelativistic common-kinetic-mass response with FD initial/final nucleon occupations. The neutron offset exceeds the proton offset by physical Q; the electron and positron occupations are relativistic FD, with prescribed *total* electron chemical potential. Electron chemical potential is a diagnostic control here: charge neutrality is not imposed. Density roots retain quantum FD statistics; classical behavior is an independently checked limiting comparison, not a substitution in the kernel.

For an empty neutrino field, with one neutrino helicity,

    R_N = [2 pi² (hbar c)³]^-1 integral_0^infinity E² [j_pc(E)+j_nd(E)] dE
    R_E = [2 pi² (hbar c)³]^-1 integral_0^infinity E³ [j_pc(E)+j_nd(E)] dE.

Here j is the occupation source coefficient in s^-1, so R_N is fm^-3 s^-1 and R_E is MeV fm^-3 s^-1. The positive quadrature measure contributes log(DOS weight)+(2+k)log(E); channel/node/panel accumulation is entirely logarithmic. Optional ordinary results are generated only after integration: underflow/overflow gives `value:null` with the retained finite logarithm and an explicit flag. Rounded positive IEEE subnormals are marked representable and subnormal. This is emission into f_nu=0, not opacity or transfer through matter, and does not solve the old positive-j/chi wrapper's transfer contract.

`log_spectra` calls guarded mapped capture directly with reversed nucleon pair and -mu_e for positron capture, and crossed decay directly for neutron decay. Both inverse logarithms may be evaluated by the frozen kernels but are not exponentiated or required to be positive ordinary floats by this interface. The old stale combined helper is not called.

## Single qualified experiment

T=0.3 MeV, M=10^7 MeV, (nn,np)=(7e-5,3e-5) fm^-3, mu_e=0. This extreme M is a mathematical static-target normalization limit, not a physical baryon mass or astrophysical EOS. It makes both neutron and proton gases dilute: their fugacities are 8.15109e-7 and 3.49332e-7. Their FD density roots are independently checked against the convergent eight-term fugacity expansion. Final-state blocking vanishes toward the classical limit with corrections of this order. Static neutron-decay comparison retains finite-temperature electron blocking; this is not a vacuum-lifetime calibration.

| Named contribution | Number [fm^-3 s^-1] | Energy [MeV fm^-3 s^-1] |
|---|---:|---:|
| positron capture | 2.52214257e-7 | 6.74086376e-7 |
| neutron decay | 6.87430550e-8 | 3.25831816e-8 |
| combined | 3.20957312e-7 | 7.06669558e-7 |

Every channel's number and energy moments are compared separately with nn times the frozen static per-neutron moments, alongside the combined inventory. Largest discrepancy is 6.27e-7; the combined number/energy discrepancies are 4.14e-7 and 5.89e-7. Neutron decay supplies approximately21% of the number source, so the combined result does not conceal a vanishingly small channel. This is agreement with an independently formulated static integral within the same allowed-current approximation, not external finite-mass validation.

Explicit energy panels include zero, both static endpoints Q-me and Q+me, small offsets proportional to sqrt(T/M) on both sides of each endpoint, the spectral bulk and tails to30MeV. Exact square-root variable maps resolve static endpoint behavior; they never remove finite-mass support. Outer orders8/12/20 are compared, with a maximum8→20 discrepancy8.69e-7. Cutoff12/18→30 comparisons are recorded separately; they provide sampled convergence evidence rather than a mathematical bound beyond30. At five allowed-region peak energies, internal24→36 and tail40→60 checks pass. Internal refinement is sampled, not a uniform accuracy certificate over arbitrary energies/baths.

At E=0.8313293913MeV both channels have strictly finite logarithmic tails in this bath (roughly -1.24894e7 and -55933.95). Neither ordinary coefficient is representable; the returned combined logarithm remains available with an explicit underflow flag. This demonstrates preservation of the named inventory, not accurate extreme-tail relative rates: the upstream response's asymptotic log-tail accuracy has not independently been qualified there.

## Evidence, failures and limits

Run `python3 outputs/nmb-finite-t-20261015/emission_only/run_reference.py` for39 numerical gates, and `check_controls.py` for16 control/representation gates. `results.json` and `partial.json` retain all panel log contributions, controls, density comparisons and source hashes. The initial39-pass result before a representation-edge correction is preserved separately. Initial import wiring and a numpy-bool JSON reporting error are retained in named failure files; neither was a physics gate failure. The latter completed grid is preserved in `initial_partial_before_flag_cast.json`.

The original `quantum_energy_moments/static_inventory_failure.json` is unchanged and its SHA is checked. That experiment used a different, neutral T=2MeV bath and failed the strict positive transfer wrapper; it remains excluded. This new selected emission-only limit does not retroactively qualify that wrapper, generic opacities, or a density/temperature rectangle. Interactions, muons, nuclei, relativistic/weak-magnetism current corrections and other neutrino-production channels remain absent. An independently reviewed log-domain interface for other baths/flavors and for nonempty radiation exchange remains future work.
