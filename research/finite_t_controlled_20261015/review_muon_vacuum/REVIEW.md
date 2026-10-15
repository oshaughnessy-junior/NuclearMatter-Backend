# Independent normalization review: massive-lepton vacuum charged current

2026-10-15. **Accept the stated point-V−A vacuum normalization benchmark.** All **163/163 independent checks pass** in0.142 s. This is a review of exact vacuum two-body kinematics with constant vector/axial couplings. It does not qualify omitted current terms, a full physical muon opacity or any finite-temperature kernel.

Pinned author `muon_vacuum/vacuum_cc.py` SHA256: `7f79ba2f60f90b53cd23c7cc19e9191837b99442a5fb95f298d163d20d372dd9`. Source and both preserved author test-design failures remain unchanged. Only this new review directory was written.

## Independent primary normalization anchors

[Strumia & Vissani2003, Eqs3–6,11–13](https://arxiv.org/html/astro-ph/0302055) supplies the invariant cross section, unequal-mass threshold and polynomial matrix element. The reviewer independently implements its **full unequal-mass** expression with f2=g2=0, f1=gv,g1=−ga, retaining the mass-difference terms. For physical neutrino and antineutrino channels, the appropriate crossing and initial-target mass are applied. Its reduced matrix-element convention requires physical squared amplitude=8GF²Vud² times the reduced expression.

[Guo et al.2020, Eqs47a–c](https://arxiv.org/html/2006.12051v2) independently displays the VV,VA,AA invariant products. Their vacuum reduction agrees with the stated point-current model; the additional vertices in that paper are not included here. Both primary pages were read on2026-10-15.

Published algebra is an external normalization anchor, not an executed external software comparison or an uncertainty estimate for a complete physical current.

## Explicit spinor test and spin conventions

The reviewer separately constructs Dirac u and v spinors, forms their four-current amplitudes, and sums their squared magnitudes. It does not call the author trace or invariant helper to construct that reference. External spinors have u†u=2E and the standard massive normalization. Both initial nucleon spin states are averaged with **one factor1/2**; final nucleon and lepton spins are summed.

Two massless incident Dirac solutions are evaluated explicitly. The weak projector annihilates one, leaving a single active state. There is **no further incident-neutrino1/2 average**. Thus the use of slash k completeness in the author trace does not introduce an additional interacting helicity. In the antineutrino calculation the explicit lepton bilinear is bar(v_nu) gamma(1−gamma5) v_l, with the antilepton v spin sum. This tests the transposition/sign of the antisymmetric lepton tensor without merely reusing the author's crossing rule.

At new electron energies17.3 and77.7 MeV and muon energies137.2,247.3 and511.1 MeV, angles(−0.83,0.17,0.76) and both neutrino signs, explicit spinors and the author trace agree within8.89×10⁻¹⁶ relative. The full unequal-mass published polynomial agrees within2.67×10⁻¹⁵. These include the odd mass-difference/lepton-mass contribution absent from an equal-mass-only check. Pure-vector and pure-axial equal-mass channels independently show the expected absence of neutrino/antineutrino interference differences.

A further explicit-spinor calculation boosts a new muon point from CM to target-rest lab. It independently recovers the specified incident lab energy and stationary initial target; the spin-summed amplitude is Lorentz invariant within1.12×10⁻¹⁶ relative. This tests the explicit spinor normalization under a changed frame.

## Flux, phase space and threshold

For CM initial momentum pi and final pf, invariant flux is4pi sqrt(s), while the two-body phase-space measure is pf dOmega/[16π²sqrt(s)]. Azimuthal integration therefore gives

    dσ/dcos(thetaCM)=averaged|M|² pf/[32π s pi].

The reviewer instead starts with the primary invariant dσ/dt and uses dt/dcos=2pi pf. It agrees with the author CM differential cross section within1.03×10⁻¹⁴. An adaptive integral over the invariant t interval agrees with the author angular total within9.55×10⁻¹⁵. This independently checks the CM flux/phase normalization and the azimuth factor. Conversion from MeV⁻² to cm² uses(hbarc)²×10⁻²⁶; no neutrino DOS, extra spin degeneracy or speed-of-light factor belongs in a vacuum cross section.

Thresholds are independently evaluated with60-digit decimal arithmetic from the declared masses. Agreement with the factored float64 threshold is within5.69×10⁻¹⁴ MeV. Positive-threshold channels have zero cross section below threshold and positive cross section at the next representable energy above it. The exothermic electron-neutrino/neutron case correctly has a clamped zero threshold; this does not imply equality of the incoming and final rest-mass invariant at E=0. The preserved exothermic-threshold test failure remains a test-design failure, not a repaired physical width.

## Electron, heavy-mass and representation limits

New electron and muon heavy-target tests use nondefault(gv,ga)=(0.9,1.1), E=(31.7,183.4) MeV and commonM=(2×10⁶,2×10⁷) MeV. The independently imposed static total is GF²Vud²(gv²+3ga²)El pl/π, including the conversion to cm². Both neutrino signs converge toward it as M increases; the largest finite-M error at the lower mass is3.73×10⁻⁴ and decreases by the expected recoil scaling. This tests the absolute factor that an incorrect target or incident spin average would change by two. These are mathematical normalization limits, not physical baryon masses.

A separate per-channel lepton-mass sequence at217.4 MeV verifies the massless limit without changing a global electron/muon constant; mass0.001 versus0 MeV differs by at most1.49×10⁻¹¹. The massive electron and muon channels remain independently parameterized.

No new failed check occurred. The original exothermic-threshold and finite-distance threshold-asymptotic failures are preserved with identical hashes. The author's narrowed threshold-asymptotic interval is compatible with taking an actual limit and did not change the kernel; this review does not turn finite-distance asymptotics into an arbitrary-mass error certificate.

## Scope of acceptance

Weak magnetism, pseudoscalar and their interference terms, q²-dependent form factors, radiative/Coulomb corrections, nuclear targets, interactions and finite-temperature blocking/inverse/crossed medium channels remain absent. Exact relativistic kinematics does not supply those current terms. Pseudoscalar omissions cannot be justified for muons by electron-mass suppression arguments. Numerical algebra/phase-space agreement near roundoff must not be quoted as the physical uncertainty of a muon cross section or opacity.

Reproduce with `python outputs/nmb-finite-t-20261015/review_muon_vacuum/probe_vacuum.py`. Raw explicit-spinor contributions, new point amplitudes, heavy-limit comparisons, tolerances and before/after hashes are in `results.json`.
