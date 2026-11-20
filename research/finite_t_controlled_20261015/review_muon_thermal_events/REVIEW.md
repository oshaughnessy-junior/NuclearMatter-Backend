# Independent review: selected thermal muon decay/inverse event moments

Decision: **ACCEPT the selected unpolarized, free-particle, tree-level mu− decay and three-body inverse event moments**, including electron and both neutrino-partner FD blocking, at the stated common-temperature baths. No blocking finding. This is not qualification of a full muon weak inventory, a coupled evolution, or arbitrary cold/degenerate parameter space.

The frozen author source `muon_thermal_events/muon_events.py` has SHA256 `24caed4dcd68b5785b994a89f1a84d253b9fae3c8acf2c1eed2867ef192e4884`. All author manifest and upstream hashes were checked without changing any author file. There are **74 passing review checks**, including source pins and control checks; they are not 74 independent physical benchmarks. Raw measurements are in `results.json` and `anchors.json`; independently authored executable probes are `probe.py` and `anchors.py`.

## Physical measure and inverse

The unpolarized squared amplitude is 64 GF²(P·ka)(pe·kn), with initial muon spin averaged, all final charged spins summed, and one physical helicity for each massless neutrino. The common thermal phase-space measure is g_mu d³P/(2π hbarc)³ times (m/E) dΓ_rest with g_mu=2. Thus its spin factor restores the fully summed muon spin in both directions. Electron spin is already in the amplitude sum; adding g_e would double count. Neither neutrino receives a factor of two. There is no CKM factor.

The direct positive products are F=f_mu(1−f_e)(1−f_anti_e)(1−f_nu_mu) and B=(1−f_mu)f_e f_anti_e f_nu_mu. They share exactly the same invariant phase-space and summed matrix element. The inverse has no extra initial-state spin average, symmetry factor, or inverse proper-lifetime factor. Multiplying the common spin-averaged rate measure by g_mu=2 is the equivalent unaveraged phase-space representation. The author actually evaluates B, rather than replacing it with a detailed-balance expression.

For common T, energy conservation gives log(F/B)=(mu_mu−mu_e−mu_anti_e−mu_nu_mu)/T. The independently constructed events verify this locally to 6.16e−14, not merely after integration. Material chemical potentials include charged-lepton rest energy. Neutrino chemical potentials apply to the named anti-electron and muon neutrino species, with the displayed affinity signs.

## Independent fresh forward values

Fresh physical free npeµ state: T=3 MeV, nB=8e−5 fm−3, Ye=.25, Ymu=.015, Yp=.265. FD roots give mu_mu=92.45401004472521 MeV and mu_e=14.799143077584498 MeV. A separate direct-p FD integral recovers n_mu−=1.2000000000000016e−6 fm−3 and n_mu+=2.055084893461116e−33 fm−3. Independent e± integration verifies the net electron density and total neutral charge. These are actual physical particle occupations, not a classical replacement.

Partner bath: mu_anti_e=8 MeV, mu_nu_mu=68.4548669671407 MeV, T=3 MeV, giving A/T=.4. The independent integral singles out anti-nu_e as spectator. With a=E_anti_e*, t=m_mu²−2m_mu a, electron energy bounds are

    Ee± = [(m_mu−a)(t+me²) ± a(t−me²)]/(2t).

The exact squared amplitude reduces independently to 32 GF² m_mu a(t−me²), and dΓ/(da dEe)=|M|²/(64π³ m_mu hbar). The electron opening angle follows from (P−pe−ka)²=0. A new Haar orientation quadrature then boosts all three correlated daughters. The parent integral uses p directly and its p² dp density measure; it does not use the author's sqrt(parent kinetic energy) map or either author's rest-event factory.

| Moment | Forward | Inverse | Net |
|---|---:|---:|---:|
| Event number, fm−3 s−1 | .00674855514428 | .00452369179499 | .00222486334929 |
| Electron energy, MeV fm−3 s−1 | .176057200288 | .118014670602 | .0580425296861 |
| Anti-electron-neutrino energy | .143580097972 | .0962446178824 | .0473354800896 |
| Muon-neutrino energy | .457421992555 | .306619131107 | .150802861448 |
| Parent muon energy | .777059290816 | .520878419592 | .256180871224 |

Independent order24→32 changes all forward/inverse/net moments by at most1.688e−7. The fresh author pair-coordinate result differs from the independent order32 result by at most4.464e−9. All comparisons retain the3e−6 gate. Individual independent experiments take8.43s and32.82s; the author fresh result takes9.85s. This establishes selected quadrature accuracy inside the specified tree-level model, not its physical uncertainty.

An additional independent one-dimensional antinue marginal is

    dΓ/da = GF² a²(t−me²)²/(2π³ t hbar).

Its adaptive integral recovers the exact finite-me phase-space factor in GF² m_mu⁵[1−8r+8r³−r⁴−12r² ln r]/(192π³ hbar), r=me²/m_mu². Its electron and nu_mu energy moments use the analytic conditional mean electron energy (Ee+ + Ee−)/2. Those independent absolute rest values match the frozen pair primitive. Fresh unblocked thermal tests then verify rate=n_mu−<m/E>Γ_rest, and each daughter energy moment=n_mu− times its rest energy rate: the boost gamma cancels the proper-time1/gamma after isotropic orientation averaging. No measured lifetime is used to adjust the normalization.

## Energy, charge, flavor and entropy

Per signed net event, material gains one electron and loses one muon, while radiation gains one anti-nu_e and one nu_mu. Hence total Le and Lmu separately conserve, material electric charge and Yp remain fixed, and dYe=−dYmu=R/nB. Material total-energy change is−(Q_anti_e+Q_nu_mu), independently equal to electron gain minus parent loss. Adding a separate muon rest-mass Q would double count energy already in the EOS and event energies. Independent per-event energy residual is4.69e−16; independently accumulated integrated energy also passes the gate.

The material entropy derivative is [−Qrad+(mu_mu−mu_e)R]/T and the FD radiation-bath derivative is [Qrad−(mu_anti_e+mu_nu_mu)R]/T. Their sum is AR/T≥0. Fresh independent moments verify both the total entropy and material heat. This is a derivative of prescribed equilibrium reservoirs. It is not a finite spectral-grid entropy result or evidence for a coupled thermal relaxation solver. Transparent radiation has a singular initial entropy derivative and is correctly excluded from a finite-entropy claim.

## Tail, limitations and retained failures

For common T, F≤exp[−(E_mu−mu_mu)/T] and B≤exp[−(E_mu−mu_e−mu_anti_e−mu_nu_mu)/T]. The parent cutoff must include both chemical centers. The repaired code does so; the high-chemical-potential algebra and domain guards pass, but high-chemical-potential *rate convergence* is not established. Mixed-temperature partners are rejected. Such baths require new tail analysis, rather than applying the common-T exponent.

Author isolated tail50→70 tests qualify the published selected state; ordinary/log outputs preserve positive underflow as an explicit representability flag. This review does not certify extreme-tail signed net representability or arbitrarily near-LTE cancellation. Neither a finite log nor affinity agreement alone establishes integral convergence.

All18 original coarse FD refinement failures and their20 excluded diagnostics remain excluded and preserved. Their failure does not disappear because refined controls pass. The review is limited to the demonstrated smooth bath states and prescribed, isotropic common-T neutrino partners. Radiative corrections, finite-W effects, polarization, mu+ decay/inverse, crossed two-body leptonic reactions, mean fields and a complete muon/neutrino flavor inventory are absent. Fixed material/partner distributions are not evolved here.

The primary [PDG muon-decay review](https://pdg.lbl.gov/2025/reviews/rpp2025-rev-muon-decay-params.pdf), checked during this review, identifies the V−A interaction and the anti-nu_e/nu_mu flavor assignment and gives the unpolarized Michel normalization. The independent finite-me invariant marginal above supplies the specific numerical normalization check. Using GF extracted with radiative corrections does not mean this tree calculation includes those corrections. The PDG's discussion of inverse muon decay also includes a crossed2→2 process; this leaf implements only the named3→1 reverse of mu− decay.

Next gates: an independent evolution coupling with separate trapped Le/Lmu and rest-inclusive EOS; a rigorously qualified hotter/different-T partner tail; explicit mu+ and crossed leptonic channels before any full muon-inventory claim. No derivative qualification follows from the present forward checks.
