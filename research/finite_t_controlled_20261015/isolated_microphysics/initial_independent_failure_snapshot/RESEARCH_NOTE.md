# Process-isolated selected microphysics research interface

**Authored checkpoint:112/112 checks PASS, 7.68 s for the final batch. Independent review pending.** This additive research leaf contains a small actual-physics subprocess adapter. It changes no frozen kernel or production backend. Its purpose is to prevent the demonstrated generic-module collision/import-time study overwrite while making state, numerical controls and provenance explicit.

## Narrow interface and ownership

`client.py` exposes frozen `MaterialState(T,nB,Ye)`, frozen `Controls(profile='base'|'fine')`, and `Client.evaluate`. The parent client uses only the standard library; it does not import NumPy, SciPy or a physical kernel. Its protocol helper is loaded by an explicit unique file spec, bypassing an existing parent `protocol` module. The caller imports the client by its actual file/module location in the ordinary way. No client code changes the parent's sys.path or inserts campaign modules in sys.modules.

Each evaluation launches the same interpreter with `-I -B` and the absolute dedicated `worker_entry.py`. `-I` ignores PYTHONPATH/user-site injection, and `-B` suppresses bytecode writes. The child may use the frozen kernels' existing generic imports in its disposable module namespace. It imports only the pinned function modules, never `study.py`, inventory scripts, validators or generic driver modules. A new process is used for every call, including repeated identical requests. There is no rate/source cache, interpolator, emulator or saved joined source.

The worker implements one explicit convention: physical-mass free relativistic particle-only neutron/proton FD states and equilibrium electron/positron pairs, fixed neutrality `np=ne−−ne+=nB Ye`. Muons, photons, interactions and antibaryon pair EOS are absent; the existing particle-baryon antinucleon rejection remains active. Total material energy and chemical potentials include rest mass. Only neutron beta decay and positron capture plus their empty-radiation inverse zeros are returned. There is no electron capture, full neutrino inventory, transport evolution or opacity service.

The selected admission box is T∈[1.9,2.1] MeV, nB exactly10⁻⁷ fm⁻³, Ye∈[.28,.34]. These are explicit request restrictions; the **accuracy evidence is the tested initial/changed states**, not a certified rectangle. EOS density/moment controls are explicit `rtol=2e-12`, tail60T and baryon anti_max_ratio10⁻⁸. Actual state roots use those arguments directly. The base PC quadrature is28/44/10 with tail60T and beta16/12/6/8; fine is32/52/12 tail70T and beta24/20/10/12. All source model parameters, physical masses, bare weak mass938.91875434 MeV, GF, Vud, gA, vector/axial/pion masses and units appear in the request conventions and response echo. No Q is added to the rest-inclusive material energy.

Current/phase-space physics remains in the reviewed function modules. PC uses the detailed independently reviewed vectorized invariant phase integral; beta uses the actual rest three-body events and moving-parent boosts. This interface supplies isolation and a checked numerical contract, not a new physical approximation or independent absolute-rate derivation.

## Protocol and forward gates

`protocol.py` accepts one strict JSON value, rejects duplicate keys, nonfinite constants, missing/extra fields, unsupported states/controls and mismatched conventions. Parent calls are bounded by110 s; each numerical kernel has90 s controls. Worker errors are one structured JSON object with status,error_type,message and exit2. Standard output must contain only the final JSON value; nonempty stderr, invalid JSON, trailing objects, oversized responses, nonzero exit or timeout reject the call. Neither request nor response accepts an output filename or code expression.

`PINS.json` freezes an explicit13-source allowlist (including worker, client and protocol), Python/NumPy/SciPy versions and platform. Hashes are checked before launching, before physical imports in the child, and after computation. Every imported campaign Python module must belong to the allowlist. `make_pins.py` refuses to overwrite an existing pin file: a new pin requires explicit preservation/requalification. No paper or generic validation driver is in the executable import allowlist.

The worker gates actual n,p density roots and pair neutrality before accessing EOS heat capacity or computing rates. It then gates the PC and beta integrated neutron density and direct material-energy loss. The parent independently reconstructs charge/density consistency from the response, checks masses and finite thermodynamics, source channel sums and signed energy/Le ledgers, and matches the complete request, source identity and runtime.

Positive moments carry `{status:'positive',value,log,reason:null}`; exact inverse zeros carry `{status:'zero',value:0,log:null,reason:'empty_incident_radiation'}`. The representation helper also retains finite logarithms with `value:null,status:'unrepresented'` on ordinary underflow/overflow, without floors. Synthetic log−55933/−1000/+1000 tests exercise that wire contract. **Those are representation tests, not new physical underflow calculations.** The selected PC integrator evaluates ordinary FD weights; a vanished/nonrepresentable actual total here errors rather than inventing a log. The admitted source responses are required to have represented positive channel totals. This does not extend cold log-domain kernels or certify arbitrary tail occupations.

Signed number sources are n:−R,p:+R,net electron:+R,antiνe:+R; material and escaping energies are −Q,+Q; material and radiation electron lepton numbers are +R,−R. Reported R and Q are per volume per second. The worker returns separate beta/PC positive totals and named signed sources, not a single neutrino absorption coefficient. Exactly empty radiation has no finite entropy-derivative claim.

## Actual physics and isolation evidence

The final batch makes these independent fresh-process calls in order:

| State/control | R (fm⁻³ s⁻¹) | Q (MeV fm⁻³ s⁻¹) |
|---|---:|---:|
| Initial T2,Ye.3/base |1.117361868342e−6|1.199463579686e−5|
| Changed T1.995541068003,Ye.321940128611/fine |1.063897868218e−6|1.139655143358e−5|
| Repeated initial/fine |1.117361868574e−6|1.199463580068e−5|
| Reordered changed/base |1.063897867987e−6|1.139655142986e−5|

They match frozen `transparent_antinu_element` initial/endpoint actual sources and exact EOS energy, pressure, entropy and Cv within2×10⁻¹² relative, before any derivative claim. The imported kernels' existing independent source qualifications remain required.

The parent contains deliberately colliding `study,potential,current,free_baryons,hot_moments,kernel,beta,vacuum_cc,events,protocol` objects. Every one remains the identical object after all calls, sys.path remains unchanged, and after client standard-library initialization no module is added or replaced by the worker calls. Input/output sentinel files are unchanged. Before/after hashes include frozen source AND raw result bytes in12 relevant campaign leaves; all match, including the restored hot-beta original results. This test is not only a mock protocol exercise.

Malformed response tests cover wrong state, controls, units, weak mass, provenance/runtime, extra fields, negative rate, inconsistent log, NaN energy, charge/mass corruption, wrong signed ledger and nonzero inverse. Strict stdin tests cover missing fields, duplicate JSON keys, invalid state and stale identity. A wrong expected source pin is rejected by the parent. A separate **owned disposable source mirror** has an actual modified vacuum-current file; the child rejects its hash before any physical import. The frozen source files are never modified for that probe.

## Preserved failures and limits

The first real pilot returned correct physics but failed an overly exact parent dictionary equality on signed values after log→exp roundtrip. It is preserved in `initial_client_float_ledger.py.txt`, `initial_pilot_failure.txt` and the old pins. The validator now uses finite relative2×10⁻¹² consistency while retaining exact schema/name checks; deliberately corrupt ledgers fail. An initial harness measured modules before first-use standard-library metadata initialization and reported27 added modules. Its raw report/source are preserved as `initial_parent_startup_*`; the corrected per-call isolation baseline follows client initialization and passes without altering the client physical imports. These are recorded operational failures, not physical accuracy failures. Additional pin versions preserve subsequent explicit weak-parameter and EOS-control additions.

This is process/module isolation for trusted, pinned research source. It is **not an operating-system security sandbox** against malicious kernel code, concurrent filesystem attacks or a hostile Python installation. A source hash is provenance, not a mathematical proof of its physical answer. Standard-library import initialization may populate ordinary Python modules; no physical campaign kernel enters the parent. Full generic finite-T API ownership and production packaging remain open. Optional pair-annihilation totals were not added to this leaf.

Recommended NMB integration: own a small stable material/current/radiation request schema separately from research executors; keep model/units/provenance explicit; expose selected kernels first with bounded source evaluations and forward gates; retain process-isolated compatibility until legacy imports/drivers have deliberate package boundaries. Any production port, new kernel inventory or expanded domain needs its own tests and review. No frozen API is refactored here.

Run `python check_interface.py` using the pinned interpreter. Final `results.json` records all real responses,112 checks, exact source pins and before/after frozen-artifact hashes. The owned `source_change_fixture/` exists only for rejecting changed-source bytes. Independent review of this new adapter is pending.
