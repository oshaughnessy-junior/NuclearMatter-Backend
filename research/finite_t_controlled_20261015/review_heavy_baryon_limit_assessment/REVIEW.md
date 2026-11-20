# Independent review: artificial heavy-baryon point-current limit

**ACCEPTED at the declared selected limiting-family scope:136 PASS, no new failures.** This is a literal point V−A mass-family benchmark, not the full q²-dependent physical-current limit, a universal1/M law or physical uncertainty estimate. No author/upstream source or frozen evidence was changed.

Frozen worker f8e43c2918ae43fe10cefee267b09c8cfc42c8bd62a5e272334f790d218f669d; manifest236dfb8dc548219467fc1dafe3ba1e69886e262a8f478b0d942b36958f2017ab. Fresh calculations and preserved-data audits are separate in results.json and anchor_audit_results.json (79 and57 passing gates respectively).

## Forward current/normalization gate

The current is exactly F1=1,GA=1.2723,F2=GP=0 with constant couplings, not a change of one denominator in the full current. I independently construct gamma matrices in the Weyl basis and evaluate complete massive baryon/lepton Dirac traces. The original lab two-body shell is solved at new points: neutrinoE7.9MeV,cos−.37 and antineutrinoE19.6MeV,cos.41, at lambda1 and8. The original charged-lepton/baryon capture spin average is4; the trace prefactor is GF²Vud²/8, with the chiral projection selecting the active neutrino helicity. Actual ma/mb appear in the spin projectors. Neutrino versus antineutrino order and charge signs are explicit. All four absolute scalar/trace comparisons agree within2.23e-16 without rate rescaling.

The author's preserved mass-one coarse full-phase literal-spinor calculation is independently compared with its scalar counterpart for every separate species-energy/number moment. It passes3e-6; this is a saved-data audit, not another full spinor phase integration. study.py explicitly refuses limit interpretation until BASELINE_FORWARD_PASS, and the extra spinor anchor was recorded before physical source data. The mass-one current/phase normalization is established before interpreting lambda>1. No heldout result or heavy-mass normalization factor is fed back.

## Consistent mass deformation and EOS

Both masses follow mn=lambda*mbar0+Q0/2,mp=lambda*mbar0−Q0/2 at fixed physical Q0=1.2933323600000222MeV. The float64 Q differences pass the locked2e-12MeV absolute gate. The worker explicitly configures both actual phase modules' MN,MP,MW and scalar mass arguments in isolated children, while passing deformed masses to the FD density roots. CM thresholds, final shells, rest-inclusive energies and spin projectors therefore change coherently. The recorded bare weak mass is inactive for this point current. This is not an EOS effective-mass or physical nucleon-mass sensitivity.

At all four scales I independently reconstruct neutron and proton particle densities from the16-term relativistic FD fugacity/Bessel series, using the actual saved chemical potentials and explicit mass. Both densities agree within3e-10. Fixed electron mu is exactly identical across the family. Electron/positron neutrality, T3MeV,nB1e-7fm^-3,Ye=.5490065797957709 and empty incoming radiation remain prescribed. Baryon antiparticles are not included; this is the same declared particle-only gas, not a new pair-baryon EOS.

Literal rest-inclusive species energy sums close independently for EC and PC at every scale. The saved separate emitted neutrino energies, charged-lepton energies and initial/final baryon energies give the energy ledger without adding Q. The scale sequence is a mathematical comparison, not time evolution; no work term for changing masses is claimed.

## Fresh physical integrals and allowed anchor

I execute fresh actual FD phase integrals at lambda1 and8 with a NEW profile: EC24/40/10/10/10,tail65 and PC30/48/angular11,tail70. Each scale has fresh EOS roots. These share the previously accepted physical CM/event integrations and point scalar invariant formula; they are not a second independent nuclear model or wholly different phase-space derivation. The new Weyl trace separately checks the current. All eight separate number/energy moments agree with saved fine results within1.96e-10, far below3e-6, and fresh target-density integrals pass3e-8. Runtime2.52s for the combined independent probe, including both scales.

An independent charged-lepton rapidity integral uses E=me cosh(u),p=me sinh(u),Ee pe dEe=Ee pe² du and a100T ceiling. The same absolute GF²Vud²(1+3GA²)/(2pi³hbar) and target densities yield EC[4.849878328817105e-6,7.070093372165178e-5] and PC[5.530523135269572e-6,8.813485608730493e-5] in numberfm^-3s^-1/energyMeVfm^-3s^-1. They agree with the frozen stationary allowed anchor within3e-12. No asymptotic result was used to set this normalization.

Reconstructing all four absolute relative errors against this independent anchor confirms strict decrease at scales1,2,4,8, each successive ratio in the locked[.25,.75] interval, and each scale8 error below the illustrative1% gate. Scale1 EC energy is−2.2170% and PC energy−5.5738%; scale8 maximum absolute error is.73082%. These selected finite-mass distances support the intended point-current stationary limit. They are not quadrature errors, uncertainty bars, or proof of an exact/universal1/M law. Intrinsic q² form factors in a different full current would remain different physics; this test does not remove them by assertion.

## Preservation and limitations

All manifest artifacts and upstream pins verify. The full author tree is byte-identical before/after the independent physical probe. No driver is imported or author output regenerated. Phase/current runtime changes are isolated to the review process's pure module instances; source bytes and author parent-state globals remain untouched. No source correction, basis/gate retuning, dependency/install, production, paper or git/PR action occurred.

The benchmark covers EC/PC only, not beta decay or a complete neutrino inventory. It does not authorize net-source/root accuracy, cooling/transport evolution, interacting matter, arbitrary mass scales/states, derivatives or parameter inference. Existing allowed-rate assessment20 model-accuracy failures remain unchanged. Shared-method agreement does not measure a physical uncertainty.

Commands from Genesis with the existing interpreter:

```
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_heavy_baryon_limit_assessment/probe.py
/Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_heavy_baryon_limit_assessment/anchor_audit.py
```

Raw fresh source/EOS outputs, separate comparisons, Dirac-trace gates, conserved energy tests and all retained input hashes accompany this review. No failing checks arose; no scientific failure was converted to a pass by denominator changes or normalization fitting.
