# Initial independent weak-reference design review

**168 checks pass; one required transfer-notation clarification remains. Do not implement from this initial contract until it is disambiguated.** This is a contract clarity finding, not evidence that any frozen scientific source amplitude/rate is wrong. No new physical-rate call or campaign scientific-module import was performed.

Reviewed CONTRACT `bb836077dbca372d0e231958d37788cc66b24266811522ee91751cae552fb370`, MANIFEST `eef214edf29d993a1b87323a1cfc402e21396b20b7f0d607639c0a7617415e6e`. All 12 original author files are preserved verbatim under `initial_author_snapshot/`; all 60 scientific/review/data source pins are unchanged. Ten saved spectral rows and three saved total F/B groups match their actual source JSON paths and hashes. Saved KMS, LTE positivity/balance, reaction species ledgers, invariant phase/spin prefactor arithmetic, target-rest threshold, moving-decay rest endpoint and units pass. Those are saved-data/algebra/source checks, not new rate benchmarks.

## Required clarification

The current section states: `For n->p baryon current with q=p_p−p_n`. Later the support section states: `Decay q=kanti+pe is future timelike`. For n→p+e−+antinu momentum conservation instead gives

`q_Gamma = p_p−p_n = −Q_leptons`,

`Q_leptons = p_n−p_p = p_e+k_antinu`, future timelike.

Equal q² does not equate these four-vectors in the F2/GP terms linear in transfer, nor does it remove the reciprocal adjoint/sign obligation. The frozen beta source explicitly uses `q=pp-pn` and the actual reciprocal adjoint; it is already consistent.

A standalone physical on-shell neutron-rest decay point in `initial_results.json` has q_Gamma[0]=−1.292831216680952 MeV and Q_leptons[0]=+1.2928312166809326 MeV, with equal positive squared norm .730994726741 MeV². This witness uses only kinematics and no amplitude/rate calculation. It demonstrates why the two text conventions must be labelled separately before a portable implementation is commissioned.

Requested narrow repair: give the baryon-current transfer a distinct q_Gamma name, use future Q_leptons for the decay pair/support variable, explicitly state their negative relation in decay and the reciprocal inverse current. Audit EC/PC/inverse and any kinetic/physical transfer language for the same distinction. Preserve every pinned source, scientific equation/model, saved number, gate and original artifact; no numerical or physical behavior changes are requested. Then freeze repaired text/provenance for an independent recheck.

## Other reviewed scope

Stage A clearly requires both independently evaluated positive F and B number/energy moments for EC, PC and neutron decay, with separately accumulated signed net values and actual radiation occupations. It is not merely emission, a whole interaction inventory or a spectral-opacity API. Common-temperature FD and exactly empty incoming radiation are distinct closures; no implicit neutrality/beta state or EOS/unmerged-port dependency is required. One-helicity spectral DOS, true chi versus j+chi relaxation, length units and the signs mu_nue=zeta,mu_antinue=−zeta are consistent. Transparent kinetic balance is not chemical/thermal equilibrium. Full current/masses/rest accounting and missing radiative/Coulomb/medium/pairing/modified-Urca physics remain explicit.

The eleven Stage A paths are a bounded plan, not an executed portable dependency proof; dynamic globals/generic import aliases and campaign file IO must be removed. Stage B needs a literal path list before its later batch: the informal five-path estimate may need to count public dispatch separately if tests, fixtures and docs are distinct paths. This is nonblocking to the Stage A physics design and is not part of the requested transfer repair.

Native external agreement, empirical/model uncertainty, implementation, independent forward/refinement/entropy/guard qualification, package/runtime/CI, transport, interacting EOS binding and AD remain open. Current author and saved evidence counts must not be presented as new independent physical calculations.
