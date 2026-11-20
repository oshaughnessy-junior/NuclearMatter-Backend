# Independent binder expectations from actual frozen APIs

STATIC PREPARATION ONLY. No candidate contract read, physical integrations, imports of science drivers, implementation or AD qualification. SOURCE_EXPECTATIONS.json pins15 actual public/private/API metadata and genuine saved evidence paths. Wait for root's frozen contract/report/manifest before comparison. The initial metadata-path helper error is preserved; correcting MANIFEST to CANDIDATE_MANIFEST changed no scientific source or data.

## API admission

PR51 finite_t_leptons: state_mu(T,mu,mass_mev=M_E,*,rtol=2e-12,tail=60) and state_density(T,NET_number_fm3,mass_mev=M_E,...). LeptonState exposes particle, antiparticle and signed NET densities; energy/free energy/pressure/entropy, Cv at fixed NET density, susceptibility and fixed-net dmu/dT. T>0,m>=0; signed mu/net zero are valid and zero net retains pairs. Ordinary population underflow is not proven physical absence. thermal_energy_density is a separately computed same-net cold-subtracted increment; it is not the rest-inclusive total energy needed for the ordinary first law.

PR52 finite_t_baryons: state_mu(T,total_mu,mass_mev=M_N,...) and state_density(T,POSITIVE_PARTICLE_number_fm3,mass_mev=M_N,...). ParticleState has no antiparticle gas or veto, and fixed-mass responses only. T,m,n>0; zero nucleon boundary is unsupported. Kinetic energy/chemical potential are separate from total rest-inclusive fields. No m-star substitution supplies rearrangement or RMF response.

Both scalar CPU APIs use two spin states and fm^-3 number, MeV fm^-3 energy/pressure, kB fm^-3 entropy, and MeV temperature/total chemical potentials. Strict nonBoolean finite-real controls, 0<rtol<1e-4,30<=tail<=100; unresolved/nonrepresentable responses raise. Density roots enforce1e-9 reconstructed density. Request bounds are not uniform accuracy certificates. Public signatures/record fields are AST-extracted in SOURCE_EXPECTATIONS rather than guessed aliases.

PR54 channel_moments(channel,bath,radiation=None,controls=None) supports ONLY EC,PC,NEUTRON_DECAY, actual direct F/B total moments. Default radiation is exactlyempty; controls default to per-channel base, not fine. FreeElectronBath admits T1..3MeV, each baryon eta=(mu−physicalmass)/T in[-100,-5], |mu_e/T|<=3. Radiation.empty carries no T/mu; FD requires exact commonT at the call, independent |mu_nue/T|,|mu_anti/T|<=10. Arbitrary nonequilibrium flavors need not be opposites; LTE imposes mu_nue=mu_p+mu_e−mu_n and mu_anti=−mu_nue. Controls literal orders4..64/tail30..100/budget(0,120], named coordinates and bounded event arrays. Require actual full controls, not a label.

MomentResult.names fixes six entries: number, neutrino_energy, charged_lepton_energy, proton_energy, neutron_energy, material_energy_loss. Number is fm^-3s^-1; energies MeVfm^-3s^-1. Do NOT multiply already integrated totals by DOS/c or additional spin2. Positive F/B and signed net/log records remain distinct; finite log with ordinaryNone cannot silently become zero in an ordinary ODE. Default/incomplete/timeout/error must not reuse stale moments. PR54 is not an opacity or EOS factory; PR55 is unnecessary for this total-source binder.

Mass compatibility is literal: EOS defaults MN939.565421,MP938.272088,ME.510998950 differ from fixed weak MN939.56542052,MP938.27208816,ME.51099895069. Pass explicit weak-compatible masses to EOS; HBARC197.3269804 matches. Full current/weakmass938.91875434 are fixed, not inferred from EOS fields. Source identity includes actual module origin/hash and package/class identity, not import spelling or fabricated records. No implicit unmerged dependency or dynamic campaign loader is acceptable in a later portable implementation.

## Thermodynamic chain and saved-only checks

For neutral free npe± with muons ABSENT, nn=(1−Y)n,np=Yn,ne,net=Yn; electron TOTALparticle density is not Yn. Require0<Y<1 for positive baryon components. Material e sums total EOS energies; optional trapped LTE photons explicitly add eγ=pi²T⁴/(15HC³),Pγ=eγ/3,Cvγ=4eγ/T. Zero-net muon pairs must not appear under an absent-muon label. Source admission is the intersection with PR54 even if an EOS-only state has broader request bounds.

At fixed supplied masses define hi=mui−T(dmui/dT)|ni (electron derivative fixedNET). Then eT=sumCvi, eY=n(hp−hn+he), en=(1−Y)hn+Y(hp+he); photons contribute only to eT here. Equivalent specific quantities are u=e/n,uT=eT/n,uY=eY/n,un=en/n−e/n². A useful independent identity is e+P−n en=T[s+sum ni(dmui/dT)|ni]; the PLUS sign is essential. Genuine saved forward/partial records are pinned for later algebra, not new forward calculations.

Let R=Nbeta+NPC−NEC using signed net number moments, and Q=sum net neutrino-energy moments. Ydot=R/n. The temperature chain is

    eT*Tdot = [(e+P)/n−en]*ndot − Q − eY*R/n.

Equivalently uT*Tdot=(P/n²−un)*ndot−Q/n−uY*R/n. Include composition/rest energy ONCE; no extra mass-difference Q or reaction heat. At fixedn/T thermostat power is Q+uY*R, not Q+eY*R. Entropy per baryon obeys [-Q−(mup+mue−mun)R]/(nT); transparent material entropy need not increase. Total radiation entropy needs distributions and is not supplied by number/energy moments alone.

After frozen handoff, independently check these identities/units against genuine saved records, forward before responses; no new rate/EOS integration is proposed. Adversaries should mismatch masses/rest versus kinetic mu, pair versus net density, temperature/current/controls/module provenance, radiation potentials with unchanged labels, stale state moments, missing/signed-underflow values, and duplicate composition/Q work. Check exact charge/baryon/Le stoichiometry and gross thermal/work/escaping-energy scales. Transparent kineticR=0 is distinct from chemical LTE; no automatic beta-equilibrium root or rapid-relaxation claim. Numeric/ODE consistency is not an AD or physical-model uncertainty certificate.
