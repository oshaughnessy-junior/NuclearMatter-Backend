# Independent review: relativistic free tensor and delta-shell oracle

**Accepted at the selected differential/tensor points and representation tests below.** All208 new review checks pass:192 new forward/component/support/KMS checks,11 representation/scaling checks and5 matched NR-anchor checks. Author sources, original results, and all preserved failures are unchanged. No finite-temperature EOS coupling, charged-lepton phase-space integrated rate, total source or GM1 current is qualified here.

Pinned sources:

- `relativistic_tensor/tensor.py`: `6fd5feeb66eae2c2685b0803eb27b2496a6f035f570c83d4f4689429e307d975`.
- `relativistic_delta_oracle/oracle.py`: `c5bb43e7945347bd6e30b76a18290a3cdfe4d8cd4b9f562e1401c5bd502b163e`.

## Independence and normalization audit

The tensor integrates initial baryon energy with the q-axis angular delta root, evaluates full Dirac traces and reuses the separately reviewed vacuum vertex building block. The oracle independently constructs its gamma matrices, external u/v spinors, current form factors and four explicit baryon spin amplitudes. It integrates transverse momentum with a longitudinal original-delta root. It imports neither tensor support/vertices/traces nor its quadrature. The earlier author comparison used frozen data; reading both implementations during this review does not retroactively provide implementation independence, but the distinct source and coordinates support the claimed independent forward gate.

Both definitions implement the full unaveraged target-spin sum. There is no target1/2 in W and no extra target2 in its contraction. The one inactive incident neutrino spinor is annihilated by V−A, so no incident-helicity average belongs in the differential prefactor. Both retain all16 complex W and L components and the covariant metric factors; no scalar/spin-diagonal replacement or generic charged-current conservation relation removes components.

The reviewed invariant contract gives

    W = integral d³pa/(2pi)^3 [2pi delta(Ea+omega−Eb)/(4Ea Eb)] H fa(1−fb),
    dchi_s/(dEl dc) = GF²Vud² pl/(32pi² hbar Enu) (1−fl) L_munu W^munu.

The q-axis coefficient is1/(16pi²q) before baryon azimuth, or1/(8piq) with an azimuth-averaged trace. The cylindrical oracle measure is r dr dphi/[16pi² Ea Eb |(pz+q)/Eb−pz/Ea|]. These are the same invariant measure in different coordinates. W has MeV² and no weak constant; differential coefficients have s⁻¹ MeV⁻¹ per unit cosine. The separate contract review supplied an independent lab/CM phase-volume and dilute c-sigma absolute anchor. New matched NR comparisons below add a fresh absolute coefficient check here. Neither code contains a fitted multiplicative scale.

The physical current convention follows the already checked [Guo et al., Eqs26–28](https://arxiv.org/html/2006.12051v2): physical-q² form factors and a separately specified bare weak normalization mass. This review is free U=0, so physical and kinetic q coincide. The timelike synthetic masses below deliberately differ greatly from Mweak, testing that the vertex does not silently use a changed spinor mass.

## Fresh spacelike all-component gate

Two new physical-mass points prescribe total chemical potentials, not EOS densities:

| Pair / lepton | T;mu_a,mu_b [MeV] | Enu,El;cosine;mu_l [MeV] |
|---|---|---|
| n→p / electron |7.3;948.2,940.1|97,82;−.17;31.2|
| p→n / antimuon |16;943,954|191,142;.31;−24|

Energy/transverse order40,16 azimuths and80T tails are used. Every absorption/emission W component, all L components and both differential j,chi agree with the independent oracle. The largest normalized forward discrepancy is1.56e−14. Both occupations are accumulated separately before checking KMS; no reverse rate is imposed by an exponential shortcut.

## New timelike and exact-null references

A third independent coordinate construction extends the evidence beyond the spacelike oracle's API. For a **synthetic** unequal-mass transition ma810,mb700MeV, T20,mu_a845,mu_b740,omega−105,q20MeV,Mweak938.91875434MeV, the transfer is timelike. This is a kinematic/model witness, not the physical neutron-proton channel.

Let B=omega²−q²>0, eta=sign(omega), D=mb²−ma²−B. In the transfer rest frame the initial baryon energy is E0=D/(2eta sqrt(B)) and its fixed momentum radius R=sqrt(E0²−ma²). Integrating the original radial delta gives the invariant measure

    R dcos(theta') dphi/(16pi² sqrt(B)).

Each external spinor is boosted back to the bath frame before evaluating H and FD occupations. This sphere includes both longitudinal delta roots, instead of borrowing the author's energy-support polynomial. Independent original energy-delta and complex-step Jacobian checks also pass. The48×16 sphere integral reproduces every absorption/emission tensor component within5.56e−15 normalized error.

For exact lightlike omega=−q=−.7MeV, physical neutron/proton masses, T30,mu_n960,mu_p945, define A=(mb²−ma²)/(2omega)>0. The **original** delta fixes

    Ea=(A²+ma²+r²)/(2A),
    pz=sign(omega)(ma²+r²−A²)/(2A).

A cylindrical transverse integral uses the original Jacobian and explicit external spinors. It never displaces omega or q to manufacture a spacelike limit. Its48×16 result reproduces every W component to2.29e−13 normalized error. The largest original-shell residual is7.28e−13MeV and original-Jacobian discrepancy2.44e−10 relative. Independent forward/inverse occupation KMS agrees to1.12e−14 tensor-norm scale. These selected new witnesses address the formerly open timelike/null numerical gate; they do not certify arbitrary near-boundary floating-point inputs or timelike pair-creation cuts.

## Log representation and matched absolute NR anchor

The response's scaled tensor plus finite log scale is authoritative. Optional ordinary tensors correctly distinguish component underflow from exact empty support. A new actual low-occupation coefficient has log chi−1020.51377053 and log j−2056.38041773: ordinary values are unavailable with explicit underflow flags, rather than floors or physical zeros. Shifting only the already dilute initial mu by−1000T changes log chi by−1000 while leaving its scaled W unchanged within the gate. The final-state distribution and forward current remain fixed. This tests representation and analytic dilute scaling, not independent extreme-tail relative accuracy. The ordinary-positive oracle explicitly rejects such cases and is not presented as a deep-log reference.

A new common-mass point-V/A sequence uses T3,mu_a=M−8,mu_b=M−9,Enu16MeV,c=.37 and omega=.2 q0 sqrt(T/M)+q0²/(2M),q0=Enu sqrt[2(1−c)]. Actual El=Enu−omega and the massive electron beta factor are retained. Both implementations are matched to the frozen spin-two NR density response and its absolute GF²Vud²(hbarc)³/(4pi²hbar) coefficient. Relative finite-M differences decrease1.64367e−5→8.15845e−6 for M1e6→2e6MeV, passing the unchanged3e−5 gate. Author/oracle differences remain below1.47e−10. This is a controlled mathematical NR limit, not physical nuclear matter or a calibration of full form-factor physics to a contact current.

## Preserved failures and limits

The tensor's original fixed-phase-velocity NR discrepancy−.00664062 remains in its failed report. Its repaired shrinking thermal-velocity comparison path tests a different, controlled limit; the old discrepancy is not erased. The oracle's initial reference omitted two axial spin components, and its subsequent full-current/contact mismatch remains in separate failed records. Point-V/A matching corrects the model comparison without changing the three physical full-current results. Original timelike-test mislabeling and JSON failures also remain frozen. This review verifies their hashes and does not recast them as passes.

One new review-only NR summary initially failed numpy.bool JSON serialization after computing its values. The original driver/error is retained; explicit numpy-scalar serialization replayed the identical numerical comparison. No author edits occurred.

All new experiments are below one second. The agreement assesses selected integration, spin and normalization accuracy under a common declared current, not physical uncertainty. There is no charge-neutral EOS gate, anti-baryon thermodynamic qualification, charged-lepton energy/angle integrated opacity, crossed-decay inventory, global source conservation, interacting-medium prescription, GM1 field mapping, or production promotion in this review.
