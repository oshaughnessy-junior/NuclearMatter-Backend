# Source-defined full-current vacuum muon benchmark

15 October 2026. **Author-frozen; independent review required.** Extend the frozen `muon_vacuum/` reference only in this new directory. Exact vacuum two-body kinematics, initial-target spin average1/2 and one incident active neutrino helicity are retained. No matter response, thermal occupation, RMF dispersion or warm muon rate is supplied.

## Sources and exact convention map

[Guo et al.2020, Eqs26–27 and47a–h](https://arxiv.org/html/2006.12051v2) defines the vertex, vacuum form factors and eight nonzero interference contributions used for independent scalar-product checks. We set all medium potentials to zero, so q*=q. Numerical pole masses follow its declared convention: vector840MeV, axial1000MeV, pion139.57MeV. No strong-interaction environment is inferred from its PNS examples.

[Strumia & Vissani2003, Eqs4–7](https://arxiv.org/html/astro-ph/0302055) supplies a separate complete real-form-factor A,B,C polynomial including unequal bare nucleon masses. Its form-factor notation maps as `f1=GV`, `g1=−GA`, `f2=F2`, `g2=−GP`; the reduced squared-amplitude polynomial is multiplied by `8GF²Vud²` to match our physical amplitude convention. We insert our **same Guo form-factor values** into that polynomial. This tests its algebra, not equality of the papers' numerical parameter sets: their vector-pole choices840MeV versus sqrt(.71GeV²) differ.

The benchmark coupling values remain the frozen point reference's `gv=1`, `ga=1.2723`, `Vud=.974`, `GF=1.1663785e−11MeV^-2`, not a new precision fit. Bare neutron/proton masses are unchanged. All F2 and GP mass normalizations use **M=(mn+mp)/2**, never a lepton mass, effective mass or arbitrary source-dependent replacement.

## Vertex and crossing derivation

Using metric+---, sigma^(mu nu)=i[gamma^mu,gamma^nu]/2 and `q=pb−pa=k−l`, implement

`Gamma^mu=GV gamma^mu−GA gamma^mu gamma5+i F2 sigma^(mu nu)q_nu/(2M)−GP gamma5 q^mu/M`.

Thus weak magnetism can equivalently be written `−F2[gamma^mu,gamma^nu]q_nu/(4M)`. The explicit i and lowered q index matter. GP is dimensionless in this convention: `GP/M=2M GA/(mpi²−q²)`. Other conventions often absorb M or2M into a dimensionful pseudoscalar coefficient; substituting such a coefficient without conversion would be wrong.

For the charge-conjugate nucleon transition, the conjugate vertex is `barGamma(−q)`. Dirac adjunction flips the explicit i in the sigma vertex and flips gamma5 in the pseudoscalar vertex; reversing q compensates each. The functional vertex therefore has the same form for p→n antineutrino absorption, with the appropriate initial/final masses. The leptonic tensor transposes for antineutrinos. Consequently VA and AF change sign, whereas VV,AA,VF,FF,AP,PP do not receive that helicity sign. Their numerical values still change with the different kinematics. We do not independently flip F2 or GP by hand.

The physical amplitude remains `GF Vud/sqrt(2)` times the two currents. Matrix traces retain target averaging1/2 and no incoming-neutrino average. Every diagonal and mixed current pair is contracted directly. The eight source invariant terms are independently coded; VP and FP are checked to vanish. The antisymmetry of sigma also makes `q_mu Gamma_F^mu=0`. This does **not** impose vector-current conservation on the whole unequal-mass charged current: bare neutron/proton mass differences remain present.

## Form-factor choices and switches

Let t=q², xi=3.706 and `den=(1−t/(4M²))(1−t/MV²)²`. The implemented values are

`GV=gv[1−(1+xi)t/(4M²)]/den`, `F2=xi/den`,
`GA=ga/(1−t/MA²)²`, `GP=2M² GA/(mpi²−t)`.

`weak_magnetism=False` sets F2=0; `pseudoscalar=False` sets GP=0. **`q2_dependence=False` removes the GV,GA,F2 momentum dependence only; GP retains its pion denominator**, using the selected GA. This is explicit because treating GP as an unrelated constant would change the model. Disabling all three switches reproduces the frozen point-current trace and total cross section. The interface rejects inconsistent channel/current gv,ga, unknown amplitude methods and pole-domain inputs, avoiding silent convention mixing.

These are source-defined dipole/pion-pole choices, not uniquely known physical functions. QED/electroweak radiative corrections, second-class currents, nucleon form-factor uncertainty, nuclear effects and medium corrections remain absent. The vacuum threshold/phase space depends only on external masses and is unchanged by the current extension.

## Forward evidence

`validate.py` passes **560/560 checks** in less than1s on this host. It compares each source interference at36 selected neutrino/antineutrino, electron/muon, energy/angle configurations; verifies their sum against the full trace; checks the independently coded complete Strumia polynomial; tests zero-correction recovery of the frozen point source, forbidden VP/FP, weak-magnetism transversality, massless-lepton pseudoscalar suppression, positivity and angular refinement. No empirical normalization factor was introduced.

The largest **individual Guo component** relative discrepancy is7.39e−10, dominated by cancellation in suppressed electron pseudoscalar terms. The full Strumia polynomial differs from the full trace by at most6.69e−15; component-sum agreement is below8e−16. Full-source versus trace integrated cross sections and Gauss32→64 are separately checked. These numerical comparisons share the same specified physical current and are not an external-software execution or a physical uncertainty estimate.

At incident150MeV, the muon total cross sections are:

| current selection | nu_mu+n [cm²] | antinu_mu+p [cm²] |
| --- | ---: | ---: |
| Frozen point V−A | 1.00984167787319e−39 | 8.11862155917659e−40 |
| Point+WM, no dipoles | 1.34292854466775e−39 | 6.03990391759695e−40 |
| Point+PS, no dipoles | 9.26124352732270e−40 | 7.28738622896119e−40 |
| Point+WM+PS, no dipoles | 1.25921121952683e−39 | 5.20866858738155e−40 |
| All selected form factors | 1.12558984905986e−39 | 4.86325249872854e−40 |

Relative to point, WM alone gives+32.98% for nu and−25.60% for antinu; PS alone gives−8.29% and−10.24%. The combined selected full-current changes are+11.46% and−40.10%. These are **vacuum benchmark model differences**, not reusable correction factors for dense matter or arbitrary spectra. At electron20MeV, PS-only changes are approximately−6.15e−6 and−7.40e−6, while the muon mass makes that omission consequential at the selected muon energy.

The raw angular decomposition is retained as well. At muon150MeV, cos(thetaCM)=.2, neutrino AP is−2.37111e−11 in physical amplitude squared while PP is+6.08437e−12; AF is+4.72878e−11. For antineutrinos AP is−2.42499e−11 and PP+6.01546e−12, whereas AF is−4.57296e−11. Different initial/final masses make the magnitude comparison slightly asymmetric. Raw configurations also include near-threshold and300MeV muon points and10/50/150MeV electron points, with all form factors, signed terms and point/full ratios recorded.

## Reproducibility, failures and next gates

`python outputs/nmb-finite-t-20261015/muon_full_current/validate.py` uses the existing NumPy/Python environment. Code, source links, raw values and hashes are local. The frozen point source hash is checked explicitly. The first driver invocation had a syntax error from missing whitespace in an inline conditional and failed before evaluating physics; correcting that driver syntax changed no current formula or scientific tolerance. All subsequent forward physics gates pass; no failed numerical comparison was removed or loosened.

Independent review must rederive/check the conjugation signs, M/2M factors, pseudoscalar convention, each interference and source-polynomial crossing before acceptance. A matched external implementation point is still useful. Only after this vacuum gate should the full tensor be integrated over EOS-consistent finite-T baryon occupations and coupled to independent electron/muon compositions, inverse/crossed processes and flavor-resolved energy/number ledgers. In-medium q versus q* and effective-mass prescriptions require a fresh consistent derivation; the vacuum identity q*=q cannot be transplanted silently.
