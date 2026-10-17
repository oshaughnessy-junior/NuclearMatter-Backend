# Independent free-matter/source binder implementation review

ACCEPTED at the selected scalar CPU free-npe/pair, included-three-channel scope.
This is DIFFERENT implementation/numerical/packaging review, not the old design
review and not uniform-domain, empirical-model, minimum-runtime, CI or ODE approval.
No candidate edits are requested. The prerequisite PR51/52/54 union is explicit;
it must not be presented as a feature already available on main.

## Admission, independence and forward values

The handed-off author MANIFEST7b40760f63ce34a9633a4f6d95d65724e51d82a7ab57b6649077ea3a47586fbf,
126 frozen members and exact eight-path diff4e98dd858fc5d369afcb58fb5b78f4822780af810033c271b20a774c1d7f5d43
were verified before evaluation and again at freeze. The two new modules match
8541366391867d8c7500deea3759c1c05746c40db3642d93d3d29e85a0242d9f and
9b3130d23eb2b8a229cdd1d0dda88df3898f048b069ad8e87d172d49b0320401.
The accepted prior binder contract and candidate-blind ORACLE_PLAN/LOCK fix the
physical conventions and the two fresh heldouts, (T,nB,Ye)=(1.7,8e-8,.27) and
(2.6,1.8e-7,.41). Both actual solved chemical baths pass rate admission; a rectangle
alone is not treated as evidence. Both absent and trapped_LTE photon closures
are tested. Muons are absent, not zero-net thermal pairs.

An independently rooted original-momentum FD oracle was computed BEFORE importing
candidate code, with separate particle-only n/p and NET electron inversion. Its
lepton momentum integration includes both particle and antiparticle populations.
The baryon momentum oracle is the pinned prior independent implementation; it
contains no candidate imports or executed driver on import. These methods differ
from candidate kinetic-energy quadrature but share the declared free Hamiltonian.
No empirical or independent interaction-model uncertainty follows.

112 new EOS/response checks pass. Maximum specific-energy forward discrepancy is
1.02e-10MeV/baryon, and kinetic-chemical-potential discrepancy6.83e-13MeV; pressure,
entropy, actual pair populations and root densities also pass their locked gates.
These are not normalized by the nucleon rest energy to hide thermal errors.
Only after forward checks, relaxed-composition energy partials, fixed-net versus
fixed-particle Cv and independent re-solved FD checks are compared. Cv differs
by<=3.00e-15 relative; independent FD errors are3.98e-8 relative for T,
7.64e-8MeV for eY/n, and1.08e-7MeV for e_nB. The thermal enthalpy combination
agrees within1.03e-10MeV. This is selected first-response evidence, not AD.

Physical campaign MN939.56542052,MP938.27208816,ME0.51099895069 and
HC197.3269804 are used explicitly. They are not replaced by package defaults,
common939 or effective masses. Temperature is MeV with no rest offset. Material
energy and chemical potentials retain rest masses. The spin-two omitted-anti
Maxwell-Boltzmann bound is independently integrated in scaled MOMENTUM coordinates;
its log differs by<=2.28e-13, with both species below the1e-8 fraction threshold.
This diagnostic does not add antibaryons or qualify a high-T anti EOS.

## Fresh event sources, first law and radiation closure

Exactly FOUR fresh full-source groups, each containing THREE event-channel calls,
were used: new heldout0 empty binder; new heldout0 direct W events using the
independent momentum roots; new heldout1 common-T LTE binder; and a fresh actual
wheel replay at the author's selected initial state. They take about8.72,7.98,
10.78 and15.02 seconds respectively, each below the90s outer bound; the last
includes the cold test run. There are no additional physical source groups or
rate grids. Guard injection below uses saved records and is not counted as
physics. No source fit, saved-moment feedback or PR55 finite-band surrogate enters
any of these fresh physical evaluations.

The independently assembled direct-root event comparison passes36 positive F/B
six-moment checks, maximum log difference1.78e-15. It uses the SAME frozen current
and event kernels, so this checks the binder/chemical mapping and assembly, not
a new amplitude or independent phase model. PR54's separately reviewed current
and phase anchors remain prerequisites rather than being re-certified here.
New empty and LTE binder evaluations pass9 and11 independent ledger checks.
LTE R/gross and Q/gross are below7.13e-15 and2.21e-15, respectively.

Independent ordinary sums reproduce R=rPC+rD-rEC and Q=sum signed neutrino-energy
moments. Literal charged/p/n energy changes and separately accumulated direct
material loss agree with -Q on gross escaping-energy scales. Baryon, charge and
material-plus-radiation electron-lepton signs pass. Both density and specific-energy
forms of the imposed-density first law are reconstructed. The temperature chain
uses eY=nB(hp-hn+he),e_nB=(1-Ye)hn+Ye(hp+he),hi=mui-T*mui,T at the CORRECT fixed
particle/net number. It does not substitute chemical potentials for finite-T
composition derivatives or add a second mass-difference Q. Independent material
entropy derivatives use ds/dT=Cv/T, composition/density Maxwell relations and
actual nBdot. Material entropy may decrease in empty radiation; no moment-only
claim of total radiation entropy is made.

The source is ONLY the included EC+PC+neutron-decay/inverse closure. Photons are
an imposed trapped-LTE contribution or absent; no photon transport/full EM
thermalization, scattering, muon/nuclear source, complete weak inventory, long-time
trajectory or hydro result follows.

## Canonical admission and independent guards

26 new guard assertions pass. Independently altered anti-species, retained muons,
particle-for-net electron density, rest mass, photon identity, response fields,
radiation T, nonfinite/Boolean scalars and stale prerequisite bytes are rejected
before a physical event call. A failing actual number-root sentinel proves the
factory receipt does not bypass canonical EOS reconstruction. Full channel
controls/current/state/radiation identity, actual-branch positivity and completion
checks were inspected against the source; no persistent cache is used.

Saved actual channel records are used ONLY for synthetic admission probes:
reconstructed/altered receipts, relabelled radiation, incomplete inventory and
nonfinite nBdot reject. A coherent finite-log underflow source remains valueNone
with its logs; the RHS raises UnrepresentedSourceError retaining that exact source,
rather than treating it as physical zero. This is a scientific in-process contract,
not a sandbox or cryptographic attestation against arbitrary Python memory edits.

## Actual wheel, exact scope and preserved reviewer errors

95 packaging/source assertions pass. The eight changed paths are verified against
the explicit prerequisite union;85 baseline files remain unchanged. All seven
prerequisite modules match prior frozen bytes. Actual wheel Python files match
the staged sources. It is unpacked into the OWN review directory without installing
anything, then imported in a fresh -I process from /private/tmp. Real module
origins, a fresh complete source/RHS replay and13 unchanged cold/unit regressions
pass (5 wheel-summary assertions). Static Python3.9 grammar succeeds; no minimum
Python/NumPy/SciPy execution or clean dependency resolution is inferred. Executed
Python3.12.12/NumPy2.3.5/SciPy1.15.2 remains outside the separate NumPy<2.3
validation pin. Two pytest assertion-rewrite warnings are nonphysical.

Two reviewer harness issues are preserved: the initial scope inventory counted
known build/.pytest_cache/egg-info generated artifacts as proposed source paths
(94PASS/1fixtureFAIL), and the initial wheel script had a missing whitespace
SyntaxError before any physical call. Exact scripts/results are retained.
Corrected scope accounting hashes every excluded generated artifact in an explicit
ledger. Neither correction changes candidate/model/gates or physical evidence;
no actual candidate numerical or admission failure was found.

## Frozen disposition

294 new review assertions PASS, zero candidate failures; the13 cold test cases
are separately reported, not13 new physical models. The initial reviewer fixture
failure remains failed and retained. All126 author manifest members and all pinned
old/preparation sources remain unchanged. Source, raw results, scripts, commands
and hashes are saved in this owned review directory. Acceptance supports a scoped
reference implementation and future draft review, not a merge or CI bypass.
A real prerequisite integration base, supported-version/full CI qualification,
broader state/source accuracy, physical model uncertainty and any evolution/RMF/
AD/opacity/full-inventory extension remain separate gates. No further scientific
calls are required by this review.
