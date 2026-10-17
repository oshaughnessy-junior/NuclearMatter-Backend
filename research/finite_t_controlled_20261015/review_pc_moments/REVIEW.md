# Independent positron-capture total-moment review

**Accept the selected completed total event moments for free n+e+↔p+antinue at the matched T2 bath.** Independent source-invariant/phase-space totals agree with the author within4.58e−11 relative. There are114 main,37 conditional/control and26 strict-density/species-ledger passing checks. One additional **own coarse-density diagnostic fails and is excluded**. All62 author coarse failures and the preserved110s runtime stop remain excluded.

The frozen total source is `moments.py` SHA256 `376c05dfe75f9788a9d9ab7559db899de9c6c559e48c9a8989dcc45ae8276a12`. Every author file/report, failure record and shared current dependency is hashed before/after; no bytes changed. No author validator or stopped high-order rule was rerun.

## Forward bath and actual pair neutrality

Before integrating reactions, independent direct-momentum FD quadrature reconstructs the neutron/proton particle densities and **both** electron/positron densities at the stored chemical potentials. It verifies nn=7e−8,np=3e−8fm^-3 and nminus−nplus=np, not nminus=np. The equilibrium pair density ratio is.8525281373, so the positron population cannot be dropped or confused with net charge.

The physical masses and total chemical potentials match the frozen free npe EOS: T=2MeV, nB=1e−7fm^-3, Ye=.3, (mun,mup,mue)=(919.8287578100662,916.8449303084382,.17446333200826397)MeV. The positron potential is−mue. Muons, photons, interactions and thermal antibaryons are absent from the reaction inventory; the stored antibaryon diagnostics remain valid and unchanged.

## Independent amplitude, coordinates and absolute measure

The new reference implements vectorized **published invariant polynomials** for full V/A, weak magnetism, pseudoscalar and physical-q² form factors, using [Guo et al., Eqs.27 and47a–h](https://arxiv.org/html/2006.12051v2). It does not call the author's external-spinor amplitude for its total integrals. Three fresh reciprocal vacuum events agree with the spinor primitive within4.45e−16 relative before thermal moments are compared. Constants are fixed physical bare masses, GF and Vud, without a normalization fit.

The independent integration uses neutron **momentum with a tangent map**, positron **momentum with a logarithmic map**, and outgoing CM polar angle relative to the **incident positron CM direction**. The author uses square-root kinetic energies and outgoing direction relative to the bath boost. A separately implemented Lorentz boost constructs all lab energies. Actual initial/final occupations remain FD; no Maxwell replacement is made. Incoming radial/angular coordinates and the outgoing physical axes therefore differ, alongside the amplitude method.

For two incoming spin-two particles, the phase densities are pn²dpn/pi² and pe²dpe dc/(2pi²). The inverse antinue+p spinor/invariant matrix averages proton spin2 and does not average the active incoming neutrino. For original n+e+ the initial spin average is4, so its reciprocal squared amplitude is divided by2. This is paired with the two spin-two incoming measures, with no extra neutrino degeneracy.

The Møller factor is sqrt[(pn·pe)²−mn²me²]/(EnEe)=pi sqrt(s)/(EnEe). Together with dσ/dOmega=Mavg4² pf/(64pi²s pi), pi cancels, giving the independently used kernel pf/(64pi² sqrt(s) EnEe) times the squared amplitude. There is no stationary-neutron approximation or lab-frame outgoing-energy cutoff.

The two incoming measures give MeV6, cross section MeV^-2 and velocity dimensionless. Dividing by hbar gives MeV3/s, and by(hbarc)^3 gives fm^-3s^-1. Energy moments add oneMeV. A neutrino DOS is **not** multiplied into an already complete event integral. Vacuum cross-section reciprocity requires sigma_PC=(1/2)(pf/pi)² sigma_antip, consistent with these initial-spin dimensions and the frozen absolute anchors.

## Independent completed totals

The three independent rules use neutron/positron/angular orders20/32/8,28/44/10,32/52/12 and tails45,60,60. They take.15,.54,1.43s, respectively; total main review2.13s. The last two rules meet the stricter3e−8 integrated-neutron-density gate, with fine error4.67e−11. Their separate F/B number and all five energy moments differ by at most3.74e−10 relative.

The **own coarse20 rule** has integrated neutron density error2.31057e−7: it fails the stricter3e−8 gate even though its looser main diagnostic and moment comparisons pass3e−6. `qualification.json` explicitly records that failed coarse diagnostic rather than claiming it qualified because KMS holds. Only the completed fine rules underpin the acceptance.

Against the author's completed28/56/angular12 result, the independent fine rule differs by at most4.58e−11 across all positive forward/backward number, neutrino, positron, proton, neutron and direct material-loss moments for empty, LTE and nonequilibrium radiation. For empty radiation the independent values are:

| Quantity | Independent result |
|---|---:|
| Number |1.1173192812170766e−6fm^-3s^-1|
| Antineutrino energy |1.199461556346285e−5MeVfm^-3s^-1|
| Positron energy |1.0561635984540592e−5MeVfm^-3s^-1|
| Proton energy |.0010517448496192169MeVfm^-3s^-1|
| Neutron energy |.0010531778291981398MeVfm^-3s^-1|
| Direct material energy loss |1.1994615563462857e−5MeVfm^-3s^-1|

The inverse empty-radiation rate is exactly absent. LTE separately computed positive F/B moments cancel within their roundoff-scaled gate; cancellation is not used as a quadrature-accuracy claim. At muanti=mueq−.4T, the independent net number is3.514064535969846e−7fm^-3s^-1, antineutrino energy3.848695914401492e−6MeVfm^-3s^-1, and common-T entropy production1.4056258143879375e−7kBfm^-3s^-1.

Separate products use fn fe+(1−fp)(1−fnu) and(1−fn)(1−fe+)fp fnu. B is not generated from a KMS ratio. The review entropy uses the common thermodynamic affinity times independently integrated net events; the author additionally evaluates actual pointwise log-product differences. These checks support the common-T identity, not arbitrary nonthermal radiation entropy.

Independent stoichiometry verifies baryon, electric-charge and electron-Le conservation and matches actual source magnitudes: dn=−R,dp=+R,dpositron=−R,d(electron,net)=+R,dantinue=+R. Radiation Le is−R. The separately accumulated signed material energy Ep−En−Ee equals−Eanti, with no manually addedQ. Direct material-loss and separate rest-inclusive moment ledgers both pass.

## Conditional Eanti5 normalization and guards

The frozen cone-delta phase ordering was also replayed with the **independent invariant amplitude**, using a function clone and private primitive namespace; no author module/file was changed. This shares the author's cone quadrature and is explicitly **not** an independent derivation of that quadrature. The total-moment reference above has independent physical axes/maps.

At a fresh boost configuration, finite differentiation independently verifies dEanti/dcosCM=gamma beta Eanti*, hence the delta Jacobian1/(gamma beta Eanti*). The one-helicity DOS is Eanti²/(2pi²hbarc³). Dividing dR/dEanti by that DOS reproduces the frozen conditional j/chi within7.4e−15. The accepted q-axis response result is pinned to the exact same bath and Eanti=5MeV, not merely a copied number found at a different state. Conditional versus response j/chi differ by7.36635e−8, passing the unchanged3e−6 gate.

The independent replay confirms empty-scenario, nonfinite radiation/tail, malformed order/budget and conditional-energy guards. General cold/degenerate/high-affinity domains are not extended. Full source/conditional logs and ordinary representation flags remain those of the frozen implementation; no positive floors are introduced.

## Preserved exclusions and scope

The original author coarse report remains102checks with62failures. The110s stopped fourth rule contributes no completed moment result. Its partial report and timeout record remain unchanged. The JSON-only guard-reporting failure is preserved separately and is not evidence of numerical convergence. The later empty-scenario guard's saved bitwise identity record remains intact. No new review runtime stop or reporting exception arose.

This acceptance establishes the **selected actual total event moments**, independently normalized within the free full-current model. It does not establish a comparison of the entire neutrino-energy-integrated tensor response, every conditional energy, a generic state rectangle, or a complete antineutrino inventory. Coulomb/radiative/medium corrections and physical uncertainties remain absent.

Adding a beta-decay channel requires the SAME bath, physical mass conventions and radiation scenario. Empty/LTE cases can be compared only after those identities are checked; the present nonequilibrium offset.4T cannot be combined with a separate channel using.8T and called one common radiation bath. Older low-temperature beta results are not imported into this T2 inventory.

Reproduce from repository root:

```sh
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_pc_moments/probe.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_pc_moments/conditional_review.py
/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_pc_moments/qualification.py
```
