# Free electron charged-current total moments: CPU reference

`from nmbackend.reference import weak_reference as w` explicitly imports a scalar CPU NumPy/SciPy reference. Existing cold/JAX APIs/exports are unchanged. The reference evaluates detailed deterministic physical quadrature, with no fit, saved-moment substitution, cache, campaign-directory imports, runtime source-file registry, EOS factory or scientific calculation at import. It supports exactly three named reactions and their inverses:

- `EC`: p+e− -> n+nue and nue+n -> p+e−;
- `PC`: n+e+ -> p+antinue and antinue+p -> n+e+;
- `NEUTRON_DECAY`: n -> p+e−+antinue and p+e−+antinue -> n.

This is not a full neutrino-interaction inventory or a full Urca model. Muons, mean fields, baryon antiparticles, pairing, modified Urca, correlations, Coulomb/radiative corrections, photon/electromagnetic equilibration dynamics, hydro and AD are absent. Stage A totals retain their original request domain. The Stage B opt-in below supplies selected fixed-bath j/true-absorption chi and relaxation; angular transport opacities remain unavailable. Without explicit SpectralControls, `spectral_coefficients` retains the Stage A `NotImplementedError` behavior. A total volume rate must not be labelled a per-occupation opacity or divided by an arbitrary density/length.

## Inputs and fixed model

`FreeElectronBath(T,mu_n,mu_p,mu_e)` takes MeV and **total rest-inclusive** chemical potentials. Particle baryon occupations are per spin, f=1/[1+exp((E−mu)/T)]. Positron mu is −mu_e. Explicit chemical baths need not be charge neutral or beta equilibrated; there is no implicit net-charge/density/beta root and no mass-density input. Actual electron/positron occupations are separate; net electron density is not substituted for either particle population. Muons are absent rather than a zero-net muon pair gas. Qualified examples name when their chemical state was obtained from a separate neutral material calculation.

`Current()` is an immutable fixed model identity, not a configurable effective-mass wrapper. Constants: MN939.56542052, MP938.27208816, ME.51099895069MeV; HBARC197.3269804MeVfm; HBAR6.582119569e−22MeVs; GF1.1663785e−11MeV−2,Vud.974. These campaign masses differ slightly from existing NMB default cold constants; they are explicit, not silently rounded or inherited. Mweak=(MN+MP)/2 is a bare current normalization, not an EOS effective mass.

Full tree current follows [Guo et al., Eqs26–29,47](https://arxiv.org/html/2006.12051v2), checked against the equivalent [Strumia–Vissani convention](https://arxiv.org/html/astro-ph/0302055). Metric(+−−−), sigma=i[gamma,gamma]/2, gamma5=i gamma0 gamma1 gamma2 gamma3:

```
Gamma_np^mu(qGamma) = GV gamma^mu − GA gamma^mu gamma5
                    + i F2 sigma^{mu nu} qGamma_nu/(2 Mweak)
                    − GP qGamma^mu gamma5/Mweak
qGamma=pp−pn
Gamma_pn(qpn)=DiracAdjoint[Gamma_np(−qpn)], qpn=pn−pp
```

GV(0)=1,GA(0)=1.2723,F2(0)=3.706,MV840,MA1000,mpi139.57MeV, and GP=2Mweak²GA(q²)/(mpi²−q²), with the source-defined vector/axial q² dependence. All physical free four-momenta are on shell. In decay the future-directed pair Q=pe+kanti=pn−pp=−qGamma; equal q² does not identify the signed linear F2/GP terms. PC uses actual positron/antinu v-spinors and reciprocal crossing, not electron mass rescaling. No experimental neutron-lifetime normalization or altered Q is inserted. An interacting GM1 adapter would be a different separately reviewed model, not an effective-mass substitution here.

`Radiation.empty()` has exactly zero incoming neutrino occupation. `Radiation.fd(T,mu_nue,mu_antinue)` specifies independent common-T FD flavors and holes; both radiation chemical potentials must be supplied. T must equal bath.T exactly; no near-T substitution or hidden thermostat is applied. Equilibrium mu_nue=mu_p+mu_e−mu_n and mu_antinue=−mu_nue. Radiation FD chemical potentials are not constrained to equilibrium: inverse-dominated and emission-dominated examples are physically distinct tests. No arbitrary angular/spectral radiation function is accepted in this initial isotropic total-moment API.

Request guards: T1–3MeV, particle baryon eta=(mu−mass)/T in[−100,−5], |mu_e/T|<=3 and radiation |mu/T|<=10. Integer orders4–64, tail30–100T, wall budget(0,120]s and <=250000 vectorized events per radial node bound local resource use. These are numerical request/admission restrictions, **not uniform error or physical-model accuracy certificates**. Antibaryons are explicitly omitted by the particle-Hamiltonian ansatz; in this dilute positive-mu request box the equilibrium antiparticle/particle ratio is bounded by exp(−2mu/T)*(1+exp(eta)), but no full antibaryon EOS is calculated.

## Quadrature and moment conventions

`channel_moments(channel,bath,radiation=None,controls=None)` returns frozen records. Default radiation is exactly empty and controls are the channel's named base profile. `Controls.base(channel)` and `.fine(channel)` expose declared orders/maps/tails; requesting the fine profile is not an automatic convergence certificate. Call both and examine separate F/B number and energy changes. There is no hidden fallback to stationary allowed rates or a fit. Timeout/invalid domain/unresolved positive measure raises an exception and never returns a complete physicalzero/incomplete-success record. Deterministic finite-tail results are approximations to the infinite integral; selected refinement, isolated tails and independent phase checks support the documented examples, not every admitted state.

For EC/PC, exact moving-baryon/two-body CM events resolve the incoming charged-lepton total-energy threshold, final recoil, bath-frame boosts and actual baryon/lepton holes. The amplitude averages FOUR forward initial baryon/lepton spin states; radial spin-two factors for each initial particle match it. Covariant cross-section and Moller flux cancel into the explicit event weight, with 1/HBAR and HBARC^-3 volume conversion, and no extra neutrino DOS or helicity average. The scalar polynomial is the source Eq47 reduction independently checked against literal external spin sums and Dirac traces. The independent reference PC integration remains outside package runtime/tests imports.

Decay uses two exact rest-Dalitz maps (`pair` or `electron`), actual full-current neutron-spin-averaged amplitude and spin-two parent measure p²dp/(pi²HBARC³). The proper-time factor MN/En multiplies the phase. Rest events rotate/boost before actual electron/proton blocking and neutrino holes. Parent rules are generalized Laguerre or finite direct momentum with explicit tail; changing tail on a Laguerre rule does not change that infinite-interval rule. Isolated parent-tail tests use the finite direct-p profile. The lab antinu spectrum has no neutron-rest endpoint cut; total boosted events include every allowed lab energy. The reciprocal inverse uses the same covariant final-neutron measure and separately evaluated occupations, not a manufactured inverse lifetime.

Moment tuple order is `number, neutrino_energy, charged_lepton_energy, proton_energy, neutron_energy, material_energy_loss`, also available as `result.names`. Forward/backward entries are positive/log quantities; net entries are signed/log quantities accumulated directly with stable expm1 differences, rather than subtracting completed nearly equal branch totals. Empty incoming radiation gives exactly zero backward branch with reason `absent_initial_neutrino_occupation`. A finite log with `value=None` is ordinary underflow/overflow with retained positive support; it is not zero or forbidden kinematics. Positive/log constructors reject inconsistent falsezero/value/flag records. `None` energy-ledger diagnostics mean ordinary representation is unavailable, not a passing energy gate.

Volume number: fm^-3s^-1; energy: MeVfm^-3s^-1; entropy: kBfm^-3s^-1. Number cgs multiplies1e39; energy cgs multiplies1e39*1.602176634e−6. No neutrino occupation DOS needs adding to these event totals. A later spectral API would have D(E)=E²/[2pi²HBARC³] for one active helicity and j/chi in s^-1 per occupation, independently reconstructed against these totals; Stage A totals supply no such coefficients; the separate fixed-bath spectral opt-in below does.

FD material occupation products, BEFORE radiation factors:

```
EC F=fp(1−fn)fe−       B=fn(1−fp)(1−fe−)
PC F=fn(1−fp)fe+       B=fp(1−fn)(1−fe+)
beta F=fn(1−fp)(1−fe−) B=(1−fn)fp fe−
```

Multiply F by (1−fnu) and B by fnu with the channel flavor. Both branches integrate independently; KMS/affinity is a posteriori check, not used to construct either. At LTE nonzero F and B still require absolute normalization and convergence tests.

## Species, energy and entropy

Per net forward EC event: (deltaNn,deltaNp,deltaNe_net,deltaNnue)=(+1,−1,−1,+1). PC and beta: (−1,+1,+1,+1antinue), but PC destroys one positron while beta creates one electron. The result includes integer particle-count stoichiometry, NOT abundance increments: dYe/dt=(Rbeta+RPC−REC)/nB only when binding to a separately specified material density. No EOS/pair/photon/composition evolution is included.

Rest-inclusive matter energy loss is net emitted neutrino energy, without an extra neutron–proton Q or chemical-energy heating term. `energy_ledger_scaled` independently reconstructs material energy from separate species moments and normalizes by positive gross neutrino energy F+B; `direct_energy_ledger_scaled` compares the direct per-event loss against neutrino energy. Neither divides by baryon rest energy or cancelled net. MaterialLe plus radiation(nue−antinue), charge and baryon number conserve.

For common-T FD, direct phase entropy is sum[(F−B)log(F/B)]>=0, independently checked against netR*(mu_eq−mu_rad)/T. Exactly empty radiation has no finite initial radiation entropy derivative; `entropy_production=None` in that case. A zero FD entropy is an exact numerical signedzero record. Material entropy may decrease in transparent cooling; number/energy moments alone do not define the escaping spectral entropy or a total second-law certificate.

## Evidence and remaining gates

The INTERNAL JSON fixture labels saved research source paths/hashes/rows; it is never read at package import/runtime. Forward current/trace checks precede moment interpretation. Validation separately records actual three-channel total regressions, fresh predeclared chemical baths, positive F/B moment/refinement/tail tests and independently assembled phase/current controls. Shared-model method agreement estimates numerical error, not physical uncertainty. Retained campaign coarse/cold/falsezero/fit failures are not relabelled PASS by this candidate.

No broad spectral accuracy, full flavor/inverse inventory, high degeneracy/cold generic accuracy, interacting/paired response, radiative corrections, global composition equilibrium, optional moment-fit qualification or AD/Hessian fidelity is granted. Independent numerical/physics and packaging review plus normal CI/minimum-runtime qualification are separate gates. This is a CPU research-reference candidate, not a cold API replacement or production transport backend.

## Stage B: explicit selected spectral opt-in

```
ctl = w.SpectralControls()             # declared base rule, not a convergence flag
fine = w.SpectralControls.fine()
r = w.spectral_coefficients('EC', bath, 5., controls=ctl)
d = w.capture_differential_coefficients('PC', bath, 1.7,
        .5987748936352157, -.1995027501115606, controls=fine)
rule = w.EnergyRule.gauss(2., 6., 4)
pilot = w.reconstruct_spectral_moments('EC', bath, w.Radiation.empty(), rule, ctl)
```

Only EXACT bath T2,mu_n919.8287578100662,mu_p916.8449303084382,
mu_e.17446333200826397MeV is admitted; no chemical/EOS factory is hidden.
Capture requests require0<E<=60MeV, neutron decay0<E<=1.4MeV; these are
numerical request limits rather than physical endpoints or uniform accuracy.
The current/masses/constants remain exactly those above. `controls=None`
keeps the old unavailable behavior; explicit `SpectralControls` opts in.
Total APIs/constructors and their request domain are unchanged.

`r.j` and `r.true_chi` are independently integrated material branches stripped
of external neutrino occupation. `r.relaxation` is their log-safe sum. They
have units s^-1 per one active occupation, while a capture differential has
s^-1MeV^-1 per cosine. `r.absorption_cm_inverse`=true_chi/c and
`r.absorption_km_inverse`=true_chi/c*1e5 use c=2.99792458e10cm/s. Differential
results reject length conversion. None of these is angular transport opacity.
The local source is j(1-f)-chi*f; empty radiation sets the inverse SOURCE zero,
never chi zero. KMS is checked after independent branches, not imposed.

The capture primitive uses exact positive-energy q-axis particle-hole support
with physical species ordering and the same avg4 current. The measure is
pe/(32pi³qE HBAR) integral dEa <A>phi times material FD products. No extra
spin2, neutrino average or volume HC factor is inserted. Actual lepton total
energy is integrated with El=ME+y². q_abs=k-lepton is q_Gamma for EC absorption
and -q_Gamma for PC absorption; the physical source current handles crossing.

Conditional beta uses the electron–proton invariant pair and actual moving
parent boosts. E_lab=gamma E_star(1+v cosstar), the energy-delta Jacobian is
1/[gamma(1+v cosstar)], and neutron proper time supplies1/gamma. Above the
rest endpoint E0, pmin=MN*sinh(log(E/E0)); a finite-T moving bath has positive
lab spectrum there. The future lepton-pair Q=pn-pp=-q_Gamma does not change
the oriented current definition. After the direct conditional dR/dE integral,
divide ONCE by D(E)=E²/(2pi²HC³) for one helicity to obtain per-occupation j/chi.

New `SpectralQuantity` is separate from Stage A total quantities. Positive
support requires a finite logarithm; ordinary under/overflow uses valueNone
with its representation reason. Exact zero is allowed only for an analytic
empty differential-support witness. Integrated finite-T j AND chi require
positive finite logs. Unresolved coordinates, collapsed weighted panels,
timeouts and incomplete computations raise instead of becoming physicalzero.
No geometric-width/ordinary-underflow panel skipping is implemented.

`EnergyRule` stores explicit positive interior nodes/weights and finite range.
Reconstruction computes fresh spectral nodes, applies exactly one neutrino
hole/occupation, and integrates D(E) times the source. It returns separate
positive F/B number and energy, directly accumulated signed nets, and FD
entropy from (F-B)log(F/B). Exactly empty radiation has no finite initial
radiation entropy derivative. Radiation T and both flavor chemical potentials
are explicit; PC and beta scenario labels do not imply the same mu_antinue.

`complete_finite_range=True` means all declared nodes completed; it does not
mean infinite-energy convergence. `whole_moment_qualified=False` is always
returned by this first pilot. No total species-energy moments are invented
from j/chi. Material lost energy equals the neutrino-bin net energy, with no
extraQ; materialLe plus nue-antinue conserves per named event. Quantitative
numerical scope, preserved failures and remaining gates are in
[weak spectral validation](weak_spectral_validation.md).
