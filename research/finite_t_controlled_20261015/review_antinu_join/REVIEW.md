# Integration-contract review: two named free antineutrino channels

**ACCEPT the final selected evidence join**, source `antinu_source_join/join.py` SHA256 `ef1f166ea0738d40c854dd65279c50f815e647d1a4ae46fe56798024d0b2546d`.47 final contract/arithmetic/adversarial checks pass. This is a review of aggregation and provenance, **not an independent physics re-review of my own hot-beta author leaf**. Physics acceptance is supplied by the separately authored `review_hot_beta/`267-check review and `review_pc_moments/`177-selected-check review. Their acceptance snapshots and source bytes are pinned in the accepted manifest.

The scope is only n→p+e−+anti-nu_e and n+e+→p+anti-nu_e, each with its reciprocal, at the specified matched free npe bath. Pair processes, a complete neutrino inventory, transparent kinetic equilibrium, evolution, derivatives and realistic medium/radiative physics are not supplied.

## Exact matching and common conventions

The actual source material dictionaries match exactly: T=2MeV,nB=1e−7fm−3,Ye=Yp=.3,muons absent, with the same neutron/proton masses, total chemical potentials, root values and full pair-electron EOS fields. Neutrality is the net density n_e−−n_e+=n_p. Actual electron and positron occupations are separate; their particle densities must not be identified with net charge. The selected gross number units arefm−3s−1 and energy unitsMeVfm−3s−1. Final build checks the explicit source unit strings before aggregation.

Both accepted channels use the same full massive-current model, physical q², bare weak mass(mn+mp)/2, physical mass values, GF,Vud and rest-energy conventions. Their different spin-averaging layouts are already reconciled by the separate normalization reviews. The join adds **event moments**, not squared amplitudes or per-occupation spectral coefficients, so it must not add another spin, volume conversion or neutrino DOS factor. The whole accepted source chain is hash-covered; no fresh physics calculation is replaced by a stale state.

Only identical radiation occupations can be joined. Empty radiation agrees with empty radiation. CommonT FD radiation must have the identical numerical anti-neutrino chemical potential, not merely the same sign of affinity. In particular the actual beta `emission_dominated` scenario hasmu_anti=mu_eq−.8T, while the PC `noneq` scenario hasmu_eq−.4T. The final join correctly rejects this pair. A scenario's display name is not a physical occupation variable. Explicit extra temperature fields are rejected, so a different or ambiguously specified radiation temperature cannot silently acquire the common-materialT label.

## Positive, signed and rest-inclusive ledgers

Separate gross forward/backward number, neutrino-energy and direct material-energy-loss moments are summed with `math.fsum`. Stable signed net moments from each accepted source are summed separately; the code does not recompute them by subtracting nearly equal gross totals. Nonfinite net values reject. Positive gross moments must be ordinarily represented. A finite positive-rate log with ordinaryvalue0 rejects as an invalid floor. Exact absent backward occupation is allowed only withvalue0,logNone and one of the two named source zero reasons (`empty_support` or `absent_initial_antinu_occupation`). Positive values require a finite log.

Both forward reactions decrease neutron number, increase proton number and increase net material electron number by one. Beta adds an electron; positron capture destroys a positron. Thus both have dn_n=−R,dn_p=dn_e,net=R, conserve electric charge, and give material/radiation electron-lepton changes+R/−R. The join's signs are correct. Material energy change is minus each independently accumulated direct loss; radiation energy is the emitted neutrino energy. Rest masses already enter both source event energies and the common EOS. Adding a new Q term would double count.

For empty radiation the exact selected joined values are

| Quantity | Sum |
|---|---:|
| Net number |1.1173618686632175e−6fm−3s−1|
| Net antineutrino energy |1.1994635801727575e−5MeVfm−3s−1|
| Direct material energy loss |1.1994635801727568e−5MeVfm−3s−1|

Independent arithmetic reproduces every forward/backward/net field and the E/Le/charge ledger. This state is a prescribed empty radiation bath, not inferred transparent equilibrium or a time evolution.

At common FD LTE, separate positive gross events remain nonzero and the summed net/gross number ratio is7.586161748333129e−16. This is ordinary integration roundoff. It must not be interpreted as a physical finite net source or an inferred relaxation time. The common affinity is(mu_n−mu_p−mu_e−mu_anti)/T; the algebraic reservoir entropy from the joined net source is that affinity timesR. No new nonthermal radiation entropy model is established by this sum.

## Provenance gate and concrete adversaries

The accepted manifest is created after both distinct channel reviews freeze. Final code requires mandatory author/raw/matched-state/review/upstream hash coverage, validates paths and64-character hexadecimal digests, then verifies all61 hashes actually present. Missing source/review coverage, changed source metadata, unsafe paths and malformed digests reject. The complete actual accepted-manifest build passes. A private verbatim code/data fixture also builds correctly, then rejects a changed unit string or a failed/coarse report even when that private fixture's hash is updated to match the mutation. No actual author, source or accepted-manifest bytes were edited by this review.

`postrepair_results.json` contains36 final checks: actual material equality; all gross/net arithmetic; E/Le/charge and LTE roundoff; different radiationT, NaN/Infnet, positive underflow metadata, actual−.8T versus−.4T affinity mismatch, altered electron root, missing required provenance, mutated source hash, unsafe path and malformed digest. `final_build_results.json` adds11 exact-zero/log, accepted-manifest and private valid-fixture unit/pass tests. All47 pass at the final source hash.

## Preserved failures and limited acceptance

Initial synthetic-only adversaries found three real contract gaps: explicit mismatched FD temperature was ignored, NaN net values were accepted, and a private accepted-status manifest with no hash coverage could build synthetic rows. Those failed gates remain in `initial_results.json`; the original root source is preserved separately. The first repaired code passed36 probes but still accepted a finite-log positive value reported as ordinaryzero; `floor_probe.json` preserves that additional failed adversary. Intermediate source versions are preserved by the root author. The zero guard was then made compatible with both frozen source encodings, and the required review-manifest filename was matched to the actual PC review artifact. All current gates were rerun against the final pinned source; no old failed result is relabeled PASS.

These aggregation repairs do not repair or reclassify any physical experiment. The62 PC author coarse failures,110s incomplete high-order run and independent PC review's excluded strict coarse-density diagnostic remain excluded. The actual qualified refined rows are selected uniquely, and the accepted source-review snapshots remain the basis for physical claims. No combined full antineutrino inventory or transparent source-balance/evolution gate follows from two correct named-channel sums.
