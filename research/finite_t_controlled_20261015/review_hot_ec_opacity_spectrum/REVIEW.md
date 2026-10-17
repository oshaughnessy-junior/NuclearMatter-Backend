# Independent review: hot EC spectral emissivity and true absorption

**ACCEPTED for the fixed matched free-npe bath and selected differential/integrated evidence.** 192 independent review assertions pass; no new physical gate failed. Counts include repeated metadata checks and are not counts of independent physics benchmarks. Reviewed `spectrum.py` SHA2565b837c486d96b28c35585a3be579131db951155b37280f1df07e01370a38e48a and MANIFEST8d5bc98c2c5f51130ed109babf4d9768e0e6dbb0ab1c87bd9764eadf2c1f785e. Every manifested author byte and all seven physical source pins remain unchanged.

The scope is p+e−↔n+nu_e on the actual T=2MeV,nB=1e−7fm−3,Ye=.3 free physical-mass particle-only npe/e± bath, with muons absent. It is not all-channel opacity or a uniform certificate for the convenience API's0<E<=60 admission. Actual density roots and full pair populations match the accepted event-total state, rather than net electron density being used as an electron particle population.

## Independent phase, current and absolute normalization

I reused the already independently authored original delta-shell/spinor oracle at THREE NEW physical points (E,El,cos)=(3.7,4.99,.31),(7.3,8.58,−.47),(.8,2.08,−.2). It integrates transverse momentum and the longitudinal mass-shell root, with independently explicit Dirac spinors/current matrices, instead of the author's q-axis initial-energy scalar polynomial integral. Independent24/6/tail65→36/10/tail85 rules refine before comparison. Both positive j andchi agree with the new spectral differential within1.32e-13 relative at the unchanged3e−10 gate. The original oracle integrates the two baryon occupations separately and multiplies the distinct electron/hole factors; it does not use KMS to construct emission. A separate heldout timelike point(3.7,4.99,.9999) agrees with the full16-component complex response. That latter response uses an independent trace/current contraction but shares q-axis support mathematics; it is not a second independent phase-space geometry for the timelike branch.

The current is the frozen source-labelled full treelevel [Guo et al.](https://arxiv.org/html/2006.12051v2) prescription, with physical q², bare average weak mass and explicit V/A/weak-magnetism/pseudoscalar form factors. This review does not supply radiative, Coulomb, correlation or mean-field corrections, or a physical error estimate from shared-model numerical agreement.

I independently reduce the invariant phase measure. The unaveraged matrix square is4 times the EC amplitude averaged over initial proton/electron spins. After the three-momentum delta and polar baryon energy delta,

    baryon invariant phase = (16pi² q)^−1 dEa dphi,
    outgoing lepton invariant phase = pe/(8pi²) dEl dcos,
    fixed incoming neutrino flux =1/(2Enu).

Combining the factor4 with the phi average2pi yields pe/[32pi³ q Enu hbar] times the integral of the EC-averaged amplitude, exactly the declared differential coefficient. No additional target spin or neutrino helicity factor remains. The phase-space occupations have per-spin FD convention and the unaveraged amplitude already sums active states. The lepton-energy-integrated coefficient has natural energy dimension; dividing by hbar gives s−1. Before that energy integration the differential coefficient has natural dimensionzero, giving s−1MeV−1 per cosine after division by hbar. Its density dependence is internal to the baryon phase integral. Converting to an inverse length divides by the massless neutrino speed c.

For radiation occupation f, the operator is j(1−f)−chi f. True absorption is chi; the fixed-bath relaxation eigenvalue is j+chi. This distinction is checked at all32 reconstructed energy nodes. Detailed balance follows independently from the occupation products fp(1−fn)fe andfn(1−fp)(1−fe): their ratio isexp[(mu_p+mu_e−mu_n−Enu)/T], independent of outgoing lepton geometry. Agreement of that ratio alone would not establish quadrature accuracy.

## New heldout incident-energy integral

At Eν=7.3MeV I independently reorder the outgoing integration into log electron momentum and momentum transfer q. The exact combined Jacobian is

    dEl dcos = q/(Enu El) dpe dq,
    q∈[|Enu−pe|,Enu+pe].

q panels explicitly split the lightlike boundary and the unequal-mass timelike support boundary; all branches still call exact physical support, including the nonempty timelike region. Electron energy cuts resolve the central recoil/thermal region; no width is added to the physics. The inner scalar phase primitive is shared with the author, so this is independent **outer-coordinate** verification, complemented by the independently ordered delta/spinor forward points above. It is not a wholly new current model or fully independent all-dimensional integral.

Actual outgoing-order/q-order/tail/outer-tail6/10/65T/35T→10/16/85T/45T gives j=0.0928319016264512 andchi=14.5518357504074s−1. Both refine within4.49e−8, and agree with a fresh author cosine/sqrt-lepton-energy integral within3.21e-12 at the unchanged3e−6 gate. The fresh author middle→fine change is1.35e−9. `outer_results.json` retains all four actual integrals. This bounded experiment takes8.995s, without rerunning the122s global incident-energy collection.

## Independent one-helicity total assembly and ledgers

I independently regenerate the32 Gauss nodes on0..60MeV and verify exact unique coverage, fresh matched-state equality, implementation/source hashes, positive-log identities and true-chi/relaxation at every node. `probe.py` imports no author evidence driver and does not write its files. Using stable independent `math.fsum` and **one** helicity DOS Enu²/[2pi²(hbarc)³] gives empty-radiation R=3.30886033887508e-07fm−3s−1 andQ=3.16038477780939e-06MeVfm−3s−1. Both forward/inverse number and energy moments for empty, LTE andmu_nu=mu_eq−1.2MeV agree with the independently accepted physical moving-CM event totals within1.874e−8 relative. This is an independent assembly of frozen detailed coefficients, not a fresh recalculation of all32 spectra; the new physical point and integrated-energy calculations are separate.

The physically summed phase measure already contains the emitted neutrino. The DOS enters once when converting per-occupation coefficients into a volume source, and a second helicity would incorrectly double the absolute total. The inverse has no extra spin degeneracy. Energy coverage extends to60MeV but does not impose a physical upper cutoff: measured node contributions above50MeV remain positive, R=4.46028077e-14 andQ=2.37516471e-12. They are not an analytic bound on the omitted spectrum above60.

LTE independently assembled signed residuals are actual floating-point differences, not forced zeros: R=8.20637619e-22,Q=7.93270051e-21. The nonequilibrium FD case has netR=1.48539961e-07,netQ=1.42242023e-06 and direct positive entropy production8.91239764e-08kBfm−3s−1. The directly assembled(F−B)log(F/B) agrees with the matter+radiation entropy decomposition(μeq−μnu)R/T. This demonstrates selected reciprocity and entropy signs rather than using those identities to generate a missing branch.

Per forward EC event, material(n,p,e_net) changes(+1,−1,−1), andnu_e changes+1, so baryon, charge and combined electron lepton number conserve. Rest-inclusive material loss equals emitted neutrino energy, with no extra beta-Q. The independent accepted event-total model supplies the species-energy ledger; this spectral leaf reconstructs radiation moments and does not independently provide all material-energy spectra or evolve an EOS.

## Controls, limitations and preserved records

Twenty new rejection/representation checks pass: nonfinite energies/controls, zero/out-of-range incident energy, nonliteral integer orders, mismatched fixed material state, nonfinite logs, and exact-empty versus underflow/overflow/positive representations. A finite logarithm can remain valid while the ordinary positive coefficient isNone; no floor is applied. The q=0 distributional point and arbitrarily soft energies are not newly qualified.

The author's coarse/fine global incident-energy comparison changes material quadrature, node count and upper limit together. Its agreement with the independently accepted event total supports the selected moments, but is not a separate bound for each angular, energy or tail error. The isolated author E2 tail/azimuth controls and my new E7.3 reordering are additional selected evidence. There is no general error guarantee for every energy in0..60 or every bath admitted elsewhere.

The initial reviewer manifest adapter incorrectly assumed a flat dictionary; it failed before any numerical probe. Exact script/error are preserved as `initial_manifest_adapter.py.txt` and `initial_manifest_error.txt`; the corrected adapter reads the actual `files` object. No author's gate or physical source changed. Evidence: `results.json`166 PASS, `outer_results.json`6 PASS, `controls_results.json`20 PASS. The documented author EC total's31 coarse failures and prior independent coarse-density failure remain unchanged and excluded; metadata hardening snapshots are also preserved upstream.

Accepted scope does not include arbitrary density/T/composition or radiation temperatures, cached coefficient APIs, flavor-complete inventory, beta/PC join, chemical/thermal equilibrium, transport evolution, interacting matter, physical uncertainty or AD. `CHECKPOINT.json` pins this review, every upstream author file and its source/review dependencies.
