# Physical free positron capture/inverse: selected incident-energy pilot

Frozen author prototype; **independent review pending**. Implements the named channel n+e+→p+anti-nu_e and its reciprocal anti-nu_e+p→n+e+. Selected actual FD bath/incident-energy coefficients pass the declared numerical gates, with an independent moving-target CM event route. This does not establish total neutrino number/energy moments, a complete antineutrino inventory, or a coupled source evolution.

## Matched material state and conventions

The selected free npe state has T=2 MeV, nB=1e−7 fm−3, Ye=Yp=.3; muons are explicitly absent. Fresh physical-mass FD roots give

| Quantity | Value |
|---|---:|
| neutron density | 7e−8 fm−3 |
| proton density | 3e−8 fm−3 |
| total mu_n | 919.8287578100662 MeV |
| total mu_p | 916.8449303084382 MeV |
| electron chemical potential mu_e | .17446333200826397 MeV |
| positron chemical potential | −mu_e |
| n_e+/n_e− | .8525281372661142 |

Charge neutrality is n_e−−n_e+=n_p; it is not n_e−=n_p. The pair-rich bath is obtained by an actual net-density root, without replacing mu_e by zero. Electron and positron FD occupations use their total positive energies, including me=.51099895069 MeV. Baryons have mn=939.56542052 and mp=938.27208816 MeV and total chemical potentials; their final-state blocking is retained even in this dilute state. The inherited free EOS checks its antinucleon diagnostic; antinucleons are not thermodynamic species. There are no photons or interactions in this source/EOS match.

`pc_kernel.matched(T,nB,Ye)` uses the frozen physical `matched_npe` factory and returns bath plus full thermodynamic/root metadata. `differential(bath,Enu,Epositron,cosine,...)` returns separate positive/log j and chi per dEpositron dcosine. `integrate(...)` integrates actual lepton phase space at fixed Enu and returns s−1 coefficients. A bare `Bath` represents prescribed chemical potentials; density matching is established by the recorded factory call, not inferred from arbitrary supplied chemical potentials.

## Massive current, crossing and spin

The physical transfer for absorption is q=k_anti−p_e+=p_n−p_p. The reciprocal baryon vertex is explicitly formed as Gamma_pn^mu(q)=Dirac-adjoint[Gamma_np^mu(−q)]. The adapter evaluates this transformation at every differential call, checks its algebraic equality to the real-form-factor direct expression, then reuses the frozen full relativistic response integration. For the declared real sub-pole vertex, this is an identity, not a numerical adjustment. No frozen kernel is modified.

Current conventions and form factors follow the previously reviewed [Guo et al. current and crossing prescription](https://arxiv.org/html/2006.12051v2), Eqs26–29, with independent invariant interference expressions in47a–h. The code retains vector, axial, weak-magnetism and induced-pseudoscalar terms, with physical q² dependence. The bare denominator mass is (mn+mp)/2=938.91875434 MeV. Parameters are gv=1,ga=1.2723,xi=3.706,MV=840,MA=1000,mpi=139.57 MeV,GF=1.1663785e−11 MeV−2,Vud=.974. There is no in-medium mass, q*, effective Q or current-conservation reduction.

The local lepton trace is written literally as

    L^{mu nu} = Tr[slash(k_anti) gamma^mu(1−gamma5)
                       (slash(p_e+)−me) Dirac-adjoint[gamma^nu(1−gamma5)]] .

The independent CM route instead sums explicit v_antinu and v_positron spinors and the reciprocal baryon u-spinor current. Neither route averages the incident antineutrino helicity. The response baryon trace sums both initial and final spins with per-spin FD occupations. The CM amplitude averages only its initial proton spin, paired with the spin2 proton density measure. No extra positron degeneracy or neutrino factor2 is inserted.

With q along z, the shared response integrates exact positive-energy particle-hole support, not a Gaussian width. Its azimuth-averaged measure is dEa/(8π|q|) times the full unaveraged baryon trace and the relevant occupations. The differential coefficient is

    dchi/(dEl dc) = GF² Vud² pl/[32π² Eanti hbar]
                   × L_munu W_abs^{mu nu} × (1−f_e+).

Emission substitutes W_emit and f_e+, independently integrated. All factors1/(2E),2π and spin conventions are inherited from the reviewed relativistic tensor and independently tested here through the CM route and vacuum cross sections. hbar=6.582119569e−22 MeV s; hbarc=197.3269804 MeV fm.

## Occupations, affinity and units

Absorption uses f_p(1−f_n)(1−f_e+); emission uses f_n f_e+(1−f_p). Both products are evaluated directly, separately in log space. At common T,

    log(j/chi) = −[Eanti−(mu_n−mu_p−mu_e)]/T.

This relation is checked after integration; it does not generate j or prove convergence. The equilibrium antineutrino chemical potential is2.8093641696198066 MeV in this state. The occupation source is df/dt=j(1−f)−chi f; true absorption is chi and the relaxation coefficient is j+chi. An inverse mean free path in cm−1 would be chi/c, not j+chi. The one-helicity antineutrino DOS is Eanti²/(2π² hbarc³) fm−3 MeV−1. It is needed when later taking neutrino number/energy moments and is not inserted again into these coefficients.

## Selected forward coefficients and numerical qualification

At Eanti=5 MeV the final refined response gives

| Coefficient | Value, s−1 |
|---|---:|
| j | .3302220190349698 |
| chi | .987407803020526 |

The independent moving-target two-body CM event integration gives chi=.9874077302846892 s−1. Separate emission also agrees; the maximum relative absolute j/chi difference is7.367e−8, without fitting a normalization. Its radial/incident/CM-outgoing/azimuth rules8/8/16/8→12/12/24/12 converge by9.76e−12. The independently accumulated actual proton density matches the FD root. The event integration includes proton initial FD, neutron final FD blocking and positron final blocking; its inverse is a separately integrated occupation product. It does not substitute a classical target density or KMS-generated emission.

The response outer energy coordinate is El=me+x². Numerical panels use the free rest-energy center and thermal/recoil scales, but every node uses exact relativistic support and occupations. The initial outer orders4/8/12 failed six unchanged3e−6 comparisons; at order12 the response was still7.664e−6 below CM. These raw failures remain in `initial_coarse_results.json` and `results.json`. They are explicitly excluded from qualification.

Outer16→20 gives1.460e−6 relative change. A final20-lepton/16-angular/28-baryon-energy/8-periodic-phi rule with baryon tail70 and outer tail32 matches CM to7.367e−8. Isolated comparisons at fixed other controls verify outer tail24→32, baryon tail50→70, inner energy20→28, angular12→16 and phi4→8. The largest isolated change is4.722e−8, from angular refinement. The guard-only source update reproduces the prior order20 coefficients bitwise. No physical width or threshold smoothing is introduced. All individual experiments take less than25s; the six isolated runs take about60s in total.

Two fresh differential points also agree with an independent transverse-momentum/original-delta shell integration using explicit spinors and its own current. This checks differential normalization and crossing separately from outer lepton integration. The literal reciprocal spinor amplitudes at24angles for incident2,5,20 MeV agree with source invariant amplitudes. Their total full-current free-proton cross sections are

| Eanti, MeV | sigma, cm² |
|---|---:|
|2|3.1775564587441345e−44|
|5|1.2308283399124576e−42|
|20|2.822889709026281e−41|

These are tree-level benchmark model predictions, not measurements or a physical uncertainty estimate. Agreement of quadrature methods measures numerical error inside the same current model.

## Exact rest threshold and actual thermal support

The proton-rest absorption threshold is [(mn+me)²−mp²]/(2mp)=1.8060662080187608 MeV. This is an invariant statement about a stationary target, not a finite-temperature spectral cutoff. A constructed proton with pz=−100 MeV and E=943.5859851757691 MeV admits an exact two-body CM event for Eanti=1.7 MeV. Its boosted positron has total energy.5987748936352157 MeV and cosine−.1995027501115606. Explicit reciprocal v-spinor amplitude is positive. The response at those same lepton kinematics gives chi=.0018607452915373494 and j=.0032402781881332265 s−1 MeV−1 per dcos, both positive with exact support. No fixed1.806MeV cut is applied in a warm bath. This differential witness is not a converged subthreshold *integrated* rate.

## Material accounting and open gates

For a net forward positron-capture event, n decreases and p increases by one; one positron is destroyed and one antineutrino emitted. Therefore net Ye and Yp each increase by R/nB, while radiation Le changes by−R and material Le by+R. Electric charge and total electron lepton number conserve. Material rest-inclusive total energy loses the emitted neutrino energy; there is no additional rest-mass Q source. At common matter/radiation T, the expected total entropy production is (mu_n−mu_p−mu_e−mu_anti)R/T. These are coupling equations, not tested integrated moment/evolution results in this leaf.

There are124 selected passing author checks plus6 preserved coarse failed checks; the count includes controls and repeated angular source comparisons, not124 independent physics benchmarks. A separate syntax-only isolation-driver failure is also preserved; it occurred before any integral ran. `CHECKPOINT.json` pins final code/raw/dependencies. Independent review of this new finite-temperature channel is still required.

No total-neutrino-energy integration, combined neutron-decay+positron-capture spectrum/moments on a common bath, full antineutrino balance, coupled material feedback, derivatives, Coulomb/radiative corrections, mean fields, correlations or modified Urca are qualified. The exactq=0 coordinate requires a separate limit. Ordinary/log coefficient flags preserve representability information without a floor; extreme cold underflow, high degeneracy and high-q² pole neighborhoods have no new domain certificate. Next work should independently review this pilot, resolve common-bath crossed-decay energy integration atT2, and only then combine the anti-nu_e inventory and take number/energy moments.
