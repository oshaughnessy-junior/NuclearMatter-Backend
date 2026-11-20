# Pure muon decay, inverse decay and crossed flavor inventory

15 October2026. **158/158 selected author checks pass; independent review required.** This leaf implements an exact-electron-mass, unpolarized **vacuum** mu− decay benchmark and derives the finite-T inventory/occupation contract. No medium collision kernel, coupled thermal evolution or production API is qualified.

## Source and normalization convention

The primary [PDG Electroweak review, Sec.10.3.2, Eqs.10.22–10.24](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-standard-model.pdf) specifies the electron-mass phase factor and distinguishes the QED corrections used in extracting GF. The [PDG Muon Decay Parameters review, Eqs.57.1–57.4](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-muon-decay-params.pdf) specifies the unpolarized and Michel distributions. Both were read for this contract. We use fixed GF=1.1663785e−11MeV^-2, m_mu=105.6583755MeV, m_e=.51099895069MeV, hbar=6.582119569e−22MeVs. GF is a declared input with its experimental extraction convention, not a strength refitted here. **There is no CKM Vud factor in a purely leptonic decay.** QED/radiative corrections, finite-W-propagator effects, neutrino masses, BSM couplings and polarization are omitted. The tree calculation is consequently not a measured-lifetime reproduction.

## Currents, spin and exact vacuum width

For `mu−(P) -> e−(p) + antiνe(ka) + νmu(kn)`,

`M=GF/sqrt(2) [ubar_numu(kn) gamma^alpha(1−gamma5) u_mu(P)] [ubar_e(p) gamma_alpha(1−gamma5) v_antinue(ka)]`.

The squared amplitude averaged over initial muon spin only is

`Mavg²=64 GF² (P dot ka)(p dot kn)`.

`decay.py` constructs explicit external spinors and both chiral currents as one method, and separately codes this scalar invariant. The full spin sum is divided by2 for an unpolarized initial muon; neither final neutrino is spin averaged. Inactive massless spin states are annihilated by the chiral current. The two final neutrinos have different flavors/particle charges, so there is **no identical-particle1/2 phase factor**. The inverse three-body amplitude is constructed with reciprocal spinor currents and checked directly. That comparison retains the same1/2 charged-spin bookkeeping on both sides; it is not an independently averaged three-incoming-particle opacity. The thermal common full-spin-sum kernel is **2*Mavg²**, with no initial-spin average, when every material occupation is per spin. Gamma/spinor primitives are reused honestly from the frozen independently authored capture oracle; no baryon vertex or new tensor implementation is imported.

With r=m_e²/m_mu², the closed tree-level width is

`Gamma_tree=GF² m_mu^5/(192pi³ hbar) f(r)`,

`f(r)=1−8r+8r³−r⁴−12r²ln(r)`.

At r=0 use f=1. For the numerical diagnostic close to r=1, cancellation is avoided by the exactly equivalent convergent series `f(1−d)=sum_n>=5 24 d^n/[n(n−1)(n−2)]`; this is algebra, not a fitted phase factor. Only the named tested mass ratios are numerically qualified.

## Two independent three-body reductions

The invariant measure is `Gamma=(1/2m_mu) integral dPi_e dPi_a dPi_n (2pi)^4 delta4 Mavg²`. `pair_integral` uses the **neutrino-pair invariant mass** s=(ka+kn)², with0<s<(m_mu−m_e)². In the parent rest frame the electron recoils opposite the pair. In the pair rest frame both neutrinos have energy sqrt(s)/2; integrate their polar angle and then boost them back. After overall unpolarized orientation and pair azimuth integration,

`dGamma/(ds dc*)=p_e(pair) Mavg²/(256pi³ m_mu²)`.

This method uses the scalar matrix invariant. A sin² map resolves the invariant-mass endpoints. The specialized longitudinal boost is evaluated algebraically as Ea=(EQ+pQ c*)/2, En=(EQ−pQ c*)/2, without subtracting1−beta² near a nearly lightlike pair.

`dalitz_integral` instead uses parent-rest electron energy Ee and antinue energy Ea. For pe=sqrt(Ee²−me²),

`me<Ee<(m_mu²+me²)/(2m_mu)`,

`(m_mu−Ee−pe)/2 < Ea < (m_mu−Ee+pe)/2`.

The third mass shell fixes the electron/antinue angle. The independent flat Dalitz measure is

`dGamma/(dEe dEa)=Mavg²/(64pi³ m_mu)`.

This method uses the **explicit spinor** amplitude and a threshold square-root Ee map. It does not call the pair-invariant phase construction or its boost. Both methods separately accumulate electron, antinue and numu energy moments and close the total energy ledger.

## Vacuum benchmark and Michel/energy gates

Physical mass result from the48×48 Dalitz spinor calculation:

| quantity | value |
|---|---:|
| Gamma_tree |457089.0316305337s^-1|
| proper lifetime |2.1877575938166523microseconds|
| mean total Ee |36.98425877394111MeV|
| mean E_antinue |31.696030920317668MeV|
| mean E_numu |36.97808580574124MeV|

The three means sum to m_mu. Electron total energy includes its rest mass; it is retained by matter if the electron thermalizes. The electron-neutrino and muon-neutrino energy shares are distinct. The total neutrino energy is not m_mu−m_e, and no flavor-blind cooling scalar can supply both lepton-number ledgers.

Both independent phase/matrix methods are tested at orders16,32,48 for physical me, exactly massless me, and a **hypothetical daughter mass.9m_mu** used only to test exact-mass phase algebra. Their rate and all three energy moments agree, and rates match the analytic phase factor under fixed tolerances. The hypothetical mass is not a medium effective-mass prescription or a new physical species claim.

For me=0 the unpolarized Michel spectrum, with x=2Ee/m_mu, is `dGamma/dx=2 Gamma0 x²(3−2x)`. Three new spectrum points reproduce it. Integrated flavor moments separately give Ee/m_mu=.35, E_antinue/m_mu=.30, E_numu/m_mu=.35, matching their different massless distributions. Reciprocal and scalar-versus-spinor amplitudes are also checked at new physical events, outside the integration nodes.

`probes.py` has97PASS in~.20s; `check_inventory.py` has61PASS. The physical lifetime differs from the measured value because it is deliberately tree level with declared inputs. No measured lifetime was used as a calibration target or fit. Radiative spectra and inclusive radiative widths require a consistent separate correction; multiplying an arbitrary lifetime normalization into differential kernels is not justified by this benchmark.

The initial generic Lorentz-boost implementation lost four-momentum closure at~8e−12 near the pair's lightlike endpoint, although integrated rates already agreed. Its complete source and results are preserved as `initial_general_boost*`. The specialized exact longitudinal algebra repairs that numerical conditioning; final closure passes3e−13 and no physical width/current or tolerance was changed. No failed final numerical gates arose in this batch.

## Species inventory and crossed channels

The vacuum amplitude is the common starting point; a crossed physical process needs its own external u/v currents, energy signs, allowed phase space and occupation products. The representative reaction pairs below, together with their charge conjugates, cover this four-lepton operator's1↔3 and2↔2 inventory:

| pair | representative direction |
|---|---|
| decay / inverse three-body |mu− ↔ e− + antinue + numu|
| neutrino flavor exchange |mu− + nue ↔ e− + numu|
| antineutrino flavor exchange |mu− + antinumu ↔ e− + antinue|
| charged-lepton pair annihilation / production |mu− + e+ ↔ antinue + numu|

The inverse of the second row is conventional inverse muon decay `numu+e−->mu−+nue`; the inverse third row is `antinue+e−->mu−+antinumu`. The pair-production inverse is `antinue+numu->mu−+e+`. All four pairs, both directions and all charge conjugates are independently reconstructed in `check_inventory.py`:16 directed reactions conserve charge, Le and Lmu separately. Baryon number is unchanged. Kinematic permission is a separate gate; crossing an amplitude is not proof that every1→3 crossing is supported in vacuum.

For an electron **at rest**, the first two inverse-scattering electron-target thresholds are `(m_mu²−m_e²)/(2m_e)`; that~10.923GeV diagnostic must not be imposed on a dense electron gas with moving targets. For neutrino-neutrino pair production require invariant s>=(m_mu+m_e)², not the electron-target rest threshold. Neither scattering nor pair-production kernel is implemented here. Bound-muon capture on nuclei is a separate semileptonic inventory.

## Matter/radiation energy and flavor ledgers

For one forward mu− decay at fixed baryon density, net material changes are `Delta Ne,net=+1`, `Delta Nmu,net=−1`, hence `Delta Ye=+1/NB`, `Delta Ymu=−1/NB` and `Delta Yp=Delta(Ye+Ymu)=0`. The antinue carries Le=−1; numu carries Lmu=+1. Thus both total flavor lepton numbers, total electric charge and total energy are conserved. The charge-conjugate mu+ decay has the opposite material and radiation lepton-number changes. A simulation must retain both signs/species; pair-equilibrated mu± occupations have chemical potentials±mu_mu.

Use the same full physical electron/muon energies in EOS and source terms. If the charged daughter remains thermalized, matter loses exactly E_antinue+E_numu, while its composition changes as above. The source must invert or differentiate the EOS with **both** fractions; at fixed nB the composition work is `(u_Ye−u_Ymu) R/NB` for forward events, not an additional manually inserted muon-rest Q. For a closed box use the same differential collision event to update material and both spectral radiation fields; separate scalar cooling fits cannot enforce the two flavor ledgers.

## Thermal products and partner-dependent affinities: contract only

For arbitrary diagonal species occupations, the common collision event has

`F=f_mu (1−f_e)(1−f_antinue)(1−f_numu)`,

`B=(1−f_mu) f_e f_antinue f_numu`.

The net forward event is F−B with common invariant measure and full spin sum. Every material Fermi factor is per spin; do not multiply a vacuum initial-spin-averaged matrix by another spin factor without tracking the common trace convention. Neutrino phase densities are one helicity E²/(2pi²hbarc³), and the partner neutrino occupation cannot be silently omitted.

At common T with FD chemical potentials, `log(F/B)=(mu_mu−mu_e−mu_antinue−mu_numu)/T`. With pair equilibrium mu_antinue=−mu_nue, the trapped condition is `mu_mu=mu_e−mu_nue+mu_numu`. This is a four-species balance condition, not separate lepton-beta equilibrium with nucleons.

If only numu is singled out into an occupation equation, its coefficients contain the **antinue partner**:

`j_numu ~ f_mu(1−f_e)(1−f_antinue)`,

`chi_numu ~ (1−f_mu) f_e f_antinue`.

For a common-T FD partner, `j_numu/chi_numu=exp[−(Enumu−zeta_numu)/T]`, with `zeta_numu=mu_mu−mu_e−mu_antinue`. Singling out antinue instead gives `zeta_antinue=mu_mu−mu_e−mu_numu`. Both identities are independently tested from occupation products. A nonthermal or differently heated partner does not supply a single universal FD affinity or a standalone flavor-independent opacity. The coefficients are coupled functionals of the partner spectrum.

## Next gates

Implement the moving-muon decay/inverse kernel with separate positive log products and exact three-body support; check Lorentz time dilation and Doppler energy moments against this vacuum benchmark. Add actual electron and both partner-neutrino blocking before calling the result a finite-T rate. Independently check both flavor collision integrals, entropy production and energy/Le/Lmu conservation in a controlled spectral box, separating finite-grid identities from continuum rate convergence. Implement2↔2 crossings with separately checked currents and moving-target thresholds; only then assemble a named complete leptonic inventory alongside semileptonic sources. Polarized muons, magnetic fields, coherent flavor transport, QED corrections, electromagnetic thermalization choices, coupled long-time evolution and production integration remain explicit later work.
