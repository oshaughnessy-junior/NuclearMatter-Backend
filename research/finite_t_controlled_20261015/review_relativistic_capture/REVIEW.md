# Independent review: integrated free relativistic capture and CM event oracle

**Scoped acceptance at the two frozen pilots and one new held-out muon-neutrino point.** All82 new review checks pass:36 frozen-forward/EOS/provenance checks,20 new boosted-event normalization/blocking checks and26 fresh integrated comparisons. The original lepton-integral3 coarse failures and event-oracle2 coarse failures remain unchanged. All original budget stops, reporting failures and completed-checkpoint records retain their hashes. No author edits occurred.

Sources reviewed:

- `relativistic_capture_integrals/capture.py`: `f511e3c6cfe7c95188877203d901e8b99f97536e89e1df168ea6e17c834f8ffe`.
- `relativistic_capture_oracle/events.py`: `c2c4afc2797b812c0585cdf09c598553e45c275ff362b64277b31350d49c4a76`.

## Physical inventory, normalization and independence

These are capture and inverse-capture coefficients for a fixed incident neutrino occupation, in s⁻¹. The accepted total thermal pilots are **nu_mu+n→p+mu−** only. Crossed neutron decay, other crossed inventories, total antineutrino/electron capture, incident-neutrino energy moments and evolved material are not qualified. GM1 is absent. No incident-neutrino DOS belongs inside these fixed-Enu coefficients; a future volume source requires the one-helicity Enu² dEnu/[2pi²(hbarc)³] measure.

The tensor implementation integrates the actual charged-lepton energy and angle, with an independent initial-baryon energy/azimuth response integral. The event implementation instead integrates target momentum/direction and the two outgoing CM angles, constructing explicit external spinors and source-defined vertices. It imports neither the new tensor nor its final-lepton integration. It boosts final states to the bath before evaluating their actual FD blocking. Inverse fb fl(1−fa) weights are accumulated separately, rather than obtained from forward fa(1−fb)(1−fl) by a KMS multiplier. Both algorithms implement the same physical positive-energy capture phase space.

The event amplitude has GF²Vud²/4 times the complete external-spin-current square: the target-spin average1/2 is explicit and no incident active-neutrino average is present. Its spin-two target measure after target azimuth is p²dp dc/(2pi²). The Møller factor is1−p c/Ea. With dσ/dΩCM=Mavg² pf/[64pi²s pi], the resulting natural-unit rate requires1/hbar and no additional c or hbarc factor. The tensor route's independently reviewed coefficient GF²Vud² pl/[32pi²hbar Enu] has the same absolute normalization.

Two new moving-target CM events, one neutrino and one antineutrino, compare the oracle's spinor amplitude directly with the independently implemented published-source scalar invariants and a separate full-current trace. They agree to1.45e−15 relative. At the target-rest incident energy Erest=(k·pa)/ma, the event differential CM cross section agrees with the published-source vacuum expression to2.45e−15, without a fitted scale. Original four-momentum conservation, mass shells, invariant Møller flux and lab-energy occupation KMS pass. The mass-shell residual is below4.98e−16 scaled and the blocking log-ratio residual below1.57e−13. These local antineutrino checks do not qualify a total antineutrino integral.

The current prescription is the previously reviewed physical-mass vacuum [Guo current and invariant rate](https://arxiv.org/html/2006.12051v2), with bare Mweak=(mn+mp)/2=938.91875434MeV. Effective masses or GM1 potentials are not substituted. These numerical agreements measure selected phase-space/current implementation accuracy; they are not a physical-model uncertainty estimate or an external interacting opacity benchmark.

## Frozen forward and EOS gates

Fresh free-EOS solves reproduce both pilots' physical neutron/proton/muon masses and total chemical potentials exactly at represented precision. Ye,Ymu remain independent with Yp=Ye+Ymu; the dilute zero-net-muon gas retains pair equilibrium. The event target-density integrals agree with the fresh particle-baryon density within3.00e−14 relative. State identity is checked before forward-rate comparison.

The frozen warm tensor/lepton rate is compared with the completed fine event checkpoint; the frozen dilute order20 rate is compared with the event14 result. Both j and chi pass the explicitly selected3e−6 pilot gate. The maximum log-relative rate discrepancy is6.89e−9, in the dilute pilot. Warm discrepancy is~7.3e−12. The cold inverse log near−767.96347 is finite and agrees between independent positive integrations even though the ordinary j is unavailable. Neither implementation calls that unrepresented ordinary value a physical zero.

The recovered fine warm checkpoint explicitly has panel=panels−1=4 and its saved chi,j logs exactly equal the recovered report. Its unavailable fine shell-residual/event-count metadata remain absent and **are not inferred** from earlier runs. The recovery validates completed positive sums only; it does not manufacture missing work. The tensor controls batch's interrupted work is also excluded; the actual later completed cold20 record is used. The recovered fine values and source/checkpoint provenance remain restricted to those recorded outputs.

## New held-out full integration

A new state/energy, not one of either author's pilots, uses

    T=2.7MeV, nB=1.2e−4fm⁻³, Ye=.26, Ymu=.035, Yp=.295,
    Enu=157MeV, nu_mu+n→p+mu−.

Each physical run freshly solves its full EOS state. Exact full-state and all captured source/data hashes agree across the runs; no fixed or substituted rates are reused. Lepton-integral controls are12/12/24/phi4,tail60,outertail40 and16/16/32/phi4,tail80,outertail56. Event controls are radial/incident/CMpolar/azimuth10/10/16/8,tail60 and14/14/24/12,tail80.

| Method | chi [s⁻¹] | j [s⁻¹] |
|---|---:|---:|
| Lepton integral16 |3357064.412711678|.000244208794288640|
| Moving-target event14 |3357064.412170105|.000244208794249242|

The two finest independent results differ by1.62e−10 relative. Lepton12→16 changes1.26e−11; event10→14 changes2.25e−7. Fresh independent event densities agree within7.25e−13 and retained four-momentum residuals within7.26e−16 scaled. Forward and inverse integrals each satisfy the total-chemical-potential KMS relation after integration. The pilot3e−6 tolerance passes unchanged, but remains a **selected numerical gate**, not a universal accuracy certificate for defaults, thresholds, stronger degeneracy, other spectra or other flavors.

The four new completed physical runs took18.52,43.69,1.49 and5.68 seconds, each under120 seconds. Event partial checkpoints are retained separately from completed results. No unfinished numerical work enters the comparisons. These experiments test joint order/tail changes; they do not prove a rigorous infinite-tail bound or independently isolate every error source at this new point.

## Preserved failures and release boundary

The original three coarse cold lepton rules and two coarse warm event comparisons remain failed. Their failures are not erased by the passing refined records. The oracle's earlier square-root endpoint quadrature failure and subsequent k=x² substitution, JSON serializer failures and completed-panel recovery remain recorded. This review does not reinterpret a budget stop or absent metadata as success.

Two new review-only helper scripts initially failed to parse due to missing whitespace in ternary expressions before numeric literals. Both failures occurred before physics work; their original drivers/errors are retained, and corrected scripts ran the comparisons. No physical source, numerical result, prefactor or author criterion changed.

The EOS is a qualified free particle-baryon approximation with a negligible-antibaryon diagnostic, plus exact pair leptons. Nuclear interactions/correlations, baryon pair cuts, radiative corrections, external finite-T references, complete antineutrino inventory, incident-energy-integrated emission/source moments and material/radiation evolution remain outside this acceptance. No GM1 claim follows from these free capture integrals.

Reproduction: `frozen.py`, `check_cm_event.py`, four separate `heldout.py` modes (`capture12`, `capture16`, `events10`, `events14`), then `compare_heldout.py`. `MANIFEST.json` pins the review and all captured immutable author files, including original failures and checkpoint-recovery artifacts.
