# Transparent fixed-temperature composition balance

Frozen author prototype, independent review pending. This is a selected physical free-npe charged-current source comparison at T=2 MeV and nB=1e−7 fm−3, with equilibrium electron/positron pairs and exactly empty incoming nu_e and anti-nu_e radiation. Muons and baryon antiparticles are excluded by the declared ansatz; actual EOS antiparticle logs/diagnostics remain in every state. The fixed bath is held by an external thermostat and rapid electromagnetic pair equilibration. Pair number is not conserved; electron lepton number is the net charge density. No evolving cooling or composition ODE is calculated.

## Measured result

The actual source root is **Ye=0.556044821802**, whereas the same EOS chemical root mu_n−mu_p−mu_e=0 gives **Ye=0.614520499048**. The difference is -0.058475677246; these distinct conditions were solved separately. At kinetic balance the chemical affinity is **0.516197908460 MeV**, so this is not chemical equilibrium with a zero-chemical-potential thermal neutrino reservoir. Empty incoming radiation is not a thermal reservoir.

| Kinetic-root channel | Number / fm−3 s−1 | Escaping energy / MeV fm−3 s−1 |
|---|---:|---:|
| neutron beta decay | 2.61641963655789e-11 | 1.2426947343739e-11 |
| neutron + positron capture | 6.59450841022285e-07 | 7.07262555394997e-06 |
| proton + electron capture | 6.59477005385555e-07 | 6.30369151677084e-06 |

Net material number source Rbeta+RPC−REC=-1.6690456e-16 fm−3 s−1, or dYe/dt=-1.6690456e-09 s−1. Its normalized residual is -1.27e-10. The positive escaping energy is **1.33763294976682e-05 MeV fm−3 s−1**, or 133.763295 MeV baryon−1 s−1. At zero composition drift the thermostat must provide this energy within the declared inventory. At the separate chemical root the actual source has **dYe/dt=-1.77646339955 s−1**, rather than zero. This follows from fresh event rates at that composition, not rescaling the original Ye=.3 totals.

## Physical chain and admission

The original hot beta, positron capture and hot electron capture use physical mn=939.56542052,mp=938.27208816,me=.51099895069 MeV, the same source-labelled full treelevel V/A/weak-magnetism/pseudoscalar current with physical q²-dependent form factors, and bare weak mass (mn+mp)/2. Shared current matrices at fresh timelike/spacelike transfers, masses, GF and Vud pass explicit preflight. Every changed Ye re-solves the original physical FD EOS for nn=nB(1−Ye),np=nB Ye and net lepton density np. Actual electron/positron particle occupations, not net charge as a particle population, enter the sources. Total energies and chemical potentials include rest masses; no additional reaction Q is added.

Scientific joining remained disabled until independent hot-EC acceptance was explicitly received from the parent and reviewer. `PRE_ACCEPTANCE_PREFLIGHT.json` and `pre_acceptance_controls.json` retain the rejected pre-admission source call. `EC_REVIEW_ACCEPTANCE.json` pins the accepted report and EC source. That prior review qualifies the original Ye=.3 bath and closures, not a broad parameter rectangle. This new leaf investigates selected additional compositions and requires its own independent review. All frozen upstream sources remain unchanged.

Every source at every root trial is an actual phase integral. Beta uses full three-body rest events, moving FD neutron parents, boosts and final e−/p blocking. EC uses the newly reviewed invariant-threshold-resolved moving proton/electron CM integral. PC uses the independently accepted reviewer reference `review_pc_moments/reference.py`, an actual full-current Guo-invariant event integration with neutron tangent-momentum and positron log-momentum maps and CM directions relative to the incoming positron. The parent explicitly approved this efficient actual reference. It is not saved-moment feedback or an amplitude fitted to totals. It shares the already reviewed literal/scalar current convention and physical model, so cross-method agreement does not quantify model uncertainty. No upstream driver module is imported or executed.

## Forward and numerical gates

The original Ye=.3 rates are freshly reproduced before a root. The bracket [.3,.8] is then measured to change source sign. Nine actual trial pairs (including the repeated original endpoint and final root) contain all raw EOS roots and full per-channel coarse/refined source results. At each trial, forward EOS charge/nucleon roots and integrated incoming FD densities pass before root acceptance. Maximum integrated density relative error is 2.94e-10, against3e−8.

The coarse/refined rules are EC proton/electron/angular20/32/8,tail50T→28/48/12,tail60T; PC radial n/e/angular28/44/10→32/52/12,tail60T; beta pair-rest16,parentLaguerre20,orientation8,phi12→electron-rest24,parentdirect-p28,orientation10,phi16,tail80T. Every channel's separate number and species energy moments pass the unchanged3e−6 relative gate; maximum change 5.6e-09. This includes actual rest-event and parent-coordinate changes for beta, rather than an LTE identity as the only accuracy gate.

A separate final-root experiment holds orders fixed and extends EC/PC tails60T→80T and beta direct-p tail80T→100T. Maximum individual-moment change is 1.73e-11 at the same3e−6 gate. This is selected numerical tail evidence, not an analytic omitted-tail bound. Main run328 assertions pass in 56.676s; controls/fresh-state replay/isolated tails43 pass in 10.248s, plus5 preflight and8 pre-admission checks. Counts include repeated channel/moment gates and are not counts of independent physical benchmarks. No new physical gate failed in these experiments. Original upstream coarse failures remain untouched and excluded.

The root solver's absolute Ye tolerance2e−9 and tiny net residual concern this numerical problem; they are not a uniform composition error bound, fit certificate or physical uncertainty estimate. Derivatives and sensitivities are not qualified.

## Conservation and interpretation

Per beta/PC event, material n decreases,p and net electron number increase, and anti-nu_e increases. Per EC event the material changes reverse and nu_e increases. Thus material Le change Rbeta+RPC−REC cancels radiation Le change REC−Rbeta−RPC; baryon and electric charge also conserve event by event. At kinetic balance the nu_e and anti-nu_e number sources cancel in total electron-lepton emission while both positive energy sources remain. Individually e− and e+ are consumed as pairs: e− source Rbeta−REC=−RPC, and e+ source −RPC. Restoring the imposed equilibrium pair bath draws energy from the thermostatted material; only their net difference is conserved as material electron lepton number. Separate species energy integrals and independently accumulated direct material loss are retained. Maximum normalized species energy ledger residual is 7.93e-14; no additional neutrino DOS is introduced after a total event integral.

These algebraic ledgers do not establish source accuracy. The forward, quadrature and tail gates supply separate selected numerical evidence. Empty incoming radiation has a singular initial radiation entropy derivative, so no finite radiation entropy or global thermal equilibrium is claimed.

The inventory excludes pair annihilation energy, nucleon bremsstrahlung, modified Urca, muons, interacting matter, correlations, Coulomb/radiative corrections, rescattering/trapping and all transport/evolution. The proton free beta-plus decay branch is invariantly vacuum forbidden for physical free masses; finite translational temperature does not change its rest-mass threshold. This leaf does not claim a complete neutrino energy inventory or realistic cooling-model uncertainty.

## Reproduction and preserved operational records

Run `python outputs/nmb-finite-t-20261015/transparent_capture_balance/balance.py`, then `run_balance.py`, then `check_controls_tail.py` with the existing NumPy/SciPy interpreter. The first preparatory module-path AttributeError is recorded with the exact initial source; no source integral ran in that failure. During the completed main experiment's already-loaded code a strict literal-Boolean `refined` guard was added to the file. `experiment_balance_source.py.txt` preserves the actual main-run source; the later controls prove the final file differs solely by that guard, and freshly re-evaluate the final root with all six per-channel moments bitwise identical. No physical equation changed and no failed gate was relaxed.

`CHECKPOINT.json` pins authored records and all frozen dependencies. External changes, production integration, paper publication and broader acceptance belong to the parent campaign.
