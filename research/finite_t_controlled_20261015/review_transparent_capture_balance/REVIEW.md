# Independent review: transparent free-npe composition balance

**ACCEPTED for the selected fixed T=2 MeV, nB=1e−7 fm−3, empty incoming electron-neutrino/antineutrino radiation experiment.** There are **254 passing review checks, no new failed gate**, across 137 root/EOS/source checks, 68 alternate-EC/bracket checks, 21 alternate-beta checks and 28 admission checks. Counts include repeated numerical moments and do not represent 254 independent physical benchmarks. Frozen author and upstream bytes remain unchanged. This is a research prototype acceptance, not production integration.

The reviewed `balance.py` SHA256 is `bbcf98a41d1c5eb6907eb0611da397c175e55681435193bf3af577b7bb8419c5`; author `CHECKPOINT.json` is `7e8fb5918c41b9241f5f7b4f0c7eaf0bd7cc601203bba5947b46b16a06f3db80`. Before joining any source I verified the independently accepted EC report SHA256 `2113dca85edaba56f67af36fc745a41d5d1e4e93db3eca6117ccac2eeca97aaf` and its exact source-matched acceptance. The EC report, prior PC and hot-beta reviews, current/units conventions, author contract, note, checkpoint and primitive source were read. No frozen driver/main was executed and no upstream reports were rewritten.

## Independently solved forward state and roots

My EOS inversion uses scalar adaptive QUADPACK over **infinite original momentum**, with direct relativistic FD particle densities and e− minus e+ density. It does not call the author's finite kinetic-energy `quad_vec` EOS/root. Neutron/proton/electron chemical potentials at three selected states agree within 1.14e−13 MeV. Net charge is inverted independently, using actual electron/positron populations rather than treating net electron charge as an incoming electron density.

My TOMS748 root uses fresh actual beta, PC and EC phase integrals with my independently reconstructed EOS on every call. No saved source totals or source rescaling are used for root feedback. Fresh bracket states Ye=.54 and .58 have dYe/dt=+.4884089400 and −.7283862410 s−1, respectively. Both bracket states and the final root have two actual quadrature controls. Independently measured incoming-particle FD densities agree within 1.40e−10 relative; the unchanged gate is 3e−8.

| Quantity | Independent review |
|---|---:|
| Actual kinetic source root Ye | 0.5560448217859639 |
| Same-EOS chemical root Ye | 0.6145204990481268 |
| Kinetic-root difference from author | −1.62025e−11 |
| Chemical affinity at kinetic root | about +0.5161979086 MeV |
| Refined actual root dYe/dt | −7.88177e−10 s−1 |
| Positive escaping root energy | 1.337632949814431e−5 MeV fm−3 s−1 |
| Actual dYe/dt at chemical root | −1.7764633992304226 s−1 |

The refined root channels are Rbeta=2.616419636658863e−11, RPC=6.594508410881219e−7 and REC=6.594770053633062e−7 fm−3 s−1. Root cancellation is judged by an **absolute drift/gross-source scale**, not relative error against the near-zero net. Numerical root agreement is not a physical uncertainty estimate or proof of uniqueness throughout the admitted rectangle.

The independent source controls are EC (proton,electron,initial/outgoing/phi,tail)=(24,40,10/10/10,55T)→(30,50,12/12/12,70T); PC (n,e,angular,tail)=(30,48,10,55T)→(34,54,12,70T); beta (rest,parent,orientation,phi,tail)=(20,26,8,12,75T)→(26,30,10,16,90T), direct-parent momentum/electron rest coordinates. At the same root Ye all separate number/species-energy moments change by at most 1.20e−10 relative, against the unchanged 3e−6 gate. Two-rule checks at both fresh bracket states pass, maximum 1.21e−10. These checks share the channel functions and physical model; changing rule alone does not establish physical independence.

## Alternate channel reductions at the new states

For EC I additionally invoked the previously accepted independent **direct Dirac trace**, invariant-s incident map, linear proton momentum and boost-axis outgoing CM sphere at the new root. This differs from the author's scalar current, threshold-plus-squared-energy maps and incident-electron CM axis. Rules (radial/invariant,angular,phi,tail)=(24,10,8,70T)→(32,12,10,70T) take 7.37 and 23.89 s. All six empty-radiation moments refine within 2.23e−10 and agree with the fresh scalar result within 2.23e−10 relative. Incoming proton density errors are below 7.07e−13. A new heldout Ye=.58 trace evaluation also agrees within the same 3e−6 gate. Replacing only EC in the independently assembled final ledger with the fine trace result yields dYe/dt=−7.88183e−10 s−1.

For beta I reused the separately accepted **flat-Dalitz/scalar-current/electron-axis Haar/panelled direct-parent** reference at the new root, in a fresh child process, using rules (20,10,8,12)→(28,16,12,16). It agrees with all six freshly integrated author-spinor moments and passes the unchanged density/refinement gates. These two runs take .37 and 2.46 s. The PC path is explicitly the accepted `review_pc_moments/reference.py` actual scalar phase integral, not cached moments. There is no further independent PC reimplementation here.

All methods share physical free masses, GF/Vud, full treelevel V/A/weak-magnetism/pseudoscalar form factors and the same physical-q² current. Method agreement supplies numerical/algebraic evidence within this model; it is not an independent physical model, measured-rate benchmark, medium correction or physical error bound. The author's separate fixed-order tail extension remains separate evidence; this review's control changes do not create an analytic omitted-tail bound.

## Ledgers and physical interpretation

I independently assembled the literal channel stoichiometric matrix on (n,p,e−,e+,nu_e,anti-nu_e). Its baryon, electric charge and combined electron-lepton ledgers pass. Beta creates e− and anti-nu_e; PC consumes e+ and emits anti-nu_e; EC consumes e− and emits nu_e. At the kinetic root the two neutrino **number** sources balance while their positive escaping energies add. Material net Le source is Rbeta+RPC−REC and radiation Le source has the opposite sign.

Separate rest-inclusive species-energy moments close the energy ledger to 4.27e−14 of gross escaping energy; independently accumulated direct loss closes to 1.14e−15. No extra neutron–proton reaction Q, second neutrino DOS or pair rest-energy source is added. At zero net composition change, e− and e+ are still both consumed at approximately RPC, since de−/dt=Rbeta−REC≈−RPC and de+/dt=−RPC. Maintaining the imposed FD pair bath requires electromagnetic pair replenishment and thermostat energy. Within this declared inventory the thermostat supplies Qescape≈133.7633 MeV per baryon per second at balance. Adding 2me RPC again would double count rest-inclusive energy.

The nonzero affinity at kinetic balance is consistent with empty incoming radiation: the imposed escaping-neutrino closure does not impose a zero-chemical-potential thermal neutrino reservoir. This does **not** establish chemical/thermal equilibrium, a full neutrino energy inventory, a cooling ODE or a self-consistent evolving pair bath. Initial empty-radiation entropy is not assigned a finite radiation derivative.

## Consumer boundaries, failures and exclusions

Thirteen malformed/outside material inputs and six nonliteral refinement controls are rejected. With a source-preparation spy, missing independent acceptance, false/integer acceptance flags, wrong EC source hash and changed review hash reject before any EOS/source preparation. Current source/review/upstream acceptance succeeds. The admitted Ye endpoints .05 and .9 were checked for finite EOS only; this is explicitly not endpoint-source or broad-domain qualification. The in-memory hash adversary is an acceptance test, not a filesystem sandbox/security claim.

No new physics or numerical gate failed and no tolerance was relaxed. Existing author EC 31 coarse failures and the EC review's one coarse-density failure remain untouched/excluded, as do the previously documented PC coarse failures, stopped high-order run, and beta/author operational records. This review neither erases nor reclassifies them. Reviewer raw root history is retained as `root_source_raw.json`; the later alternate-EC script reuses the review helper and writes its two bracket replays into `raw.json`. Both records are retained. There were no new runtime/reporting failures.

Acceptance is limited to these selected smooth free-npe states, prescribed pair equilibration/thermostat and exactly empty incoming radiation. Muons, antiparticle baryon source inventory, interactions, Coulomb/radiative corrections, bremsstrahlung, modified Urca, pair-annihilation emissivity, transport/trapping, long-time evolution, a full energy inventory, global state-domain certification and all derivatives remain excluded. Forward gates were performed; no derivative was attempted.

## Exact reproduction

Run from `/Users/rossma/Genesis_LANL` with the existing interpreter, without installs:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_transparent_capture_balance/probe.py
cp outputs/nmb-finite-t-20261015/review_transparent_capture_balance/raw.json outputs/nmb-finite-t-20261015/review_transparent_capture_balance/root_source_raw.json
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_transparent_capture_balance/controls.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_transparent_capture_balance/alternate_ec.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_transparent_capture_balance/alternate_beta.py
```

Main root experiment 33.83 s, alternate EC/bracket experiment 50.14 s, alternate beta 2.84 s; every complete experiment is below 120 s. Checkpoint records source/evidence hashes. Quota account/window were freshly verified before root and alternate-EC batches at 21% used, with no reset credit consumed. No installs, heavy builds, production, paper or Git operation occurred.
