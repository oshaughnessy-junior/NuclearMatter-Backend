# Evidence join for two compatible antineutrino channels

This read-only integration layer joins qualified saved total-moment evidence for neutron decay/inverse and positron capture/inverse. It performs no new physical quadrature or changed-state source evaluation. The source methods, assumptions, failed coarse controls, independent reviews and actual distributions remain in the pinned leaves. It is not a production source API or a complete neutrino inventory.

Both material records are exactly identical physical free npe states at T2 MeV,nB1e-7fm^-3,Ye.3,muons absent, including their actual pair-electron populations and physical rest masses. Only common empty radiation and common-temperature LTE are joined. The saved nonequilibrium cases deliberately fail compatibility: beta muanti=mu_eq-.8T and PC muanti=mu_eq-.4T are different distributions. Equal chemical-affinity signs are insufficient.

ACCEPTED_MANIFEST.json pins61 frozen author/review/upstream files after independent source reviews accepted their selected scopes. build verifies status, required coverage, all digests and source report completion, unique selected completed controls and literal units. No partial/time-stopped rule is selected. Radiation fields are strict: an extra explicit radiation-temperature field is unsupported, rather than silently ignored. Net values must be finite, and finite positive logs cannot be represented by zero floors. The two qualified exact-zero metadata reasons remain distinct and accepted for absent incident radiation.

Actual empty-radiation sum:

- R=1.1173618686632175e-6 fm^-3s^-1;
- Qanti=1.1994635801727575e-5 MeVfm^-3s^-1;
- separately summed direct material loss=1.1994635801727568e-5 in the same energy units;
- mean emitted antineutrino energy=10.73478175524071 MeV.

LTE retains the actual numerical residual: net number/(gross forward+backward)=7.586161748333129e-16. It is not zeroed, used to infer a relaxation rate or confused with empty-radiation equilibrium. The matter/radiation electron lepton sources are opposite; n/p/net-electron sources follow the two named reaction stoichiometries. Rest-inclusive material energy loses Qanti without an extra Q term.

Original root integration-contract failures and exact source versions remain preserved. Independent adversarial review found ignored radiation-temperature fields, incomplete manifest coverage and nonfinite net acceptance; a further finite-log zero-floor case was also rejected after repair. A too-narrow exact-zero reason guard was corrected to accommodate the actual frozen beta metadata, and required review-manifest naming was corrected after a missing-file build stop. No author or physical source was changed. These are integration/control repairs, not improvements to current-model physical accuracy.

No pair production, electron-neutrino channels, matter evolution, broad parameter domain, mean fields, modified Urca or quantified interaction/radiative uncertainty is supplied by the join. Future changed-state configurations must call the detailed source methods; these fixed-bath measurements are not cached feedback rates. Independent join-contract review is in review_antinu_join/.
