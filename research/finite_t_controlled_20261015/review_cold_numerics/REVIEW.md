# Independent numerical review: selected cold-capture quadrature

2026-10-15. **Accept the selected repaired cold-capture numerical results.** All **41/41 independent checks pass** in2.51 s (about5.08 s including the initial review run), well below the45 s numerical bound. This acceptance is pointwise and shares the frozen physical response/current. It is not a global cold-domain, physical-broadening or interacting-EOS certificate.

Pinned author `cold_capture/resolved.py` SHA256: `d988860e717ce3a08d8fc2d59a3bdc91c506be80ffb44250e6c55cd67b5354a1`. All six model/physical-source/original-failure hashes match the author freeze and remain unchanged after the review. No author/upstream file was edited. Checkpoints are saved before capture work and after forward density gates.

## Forward state first

The exact original failed point is Yp=0.1,T=0.01 MeV,Eν=0.03 MeV,nB=0.01 fm⁻³. A second, independently chosen point changes composition, temperature and Eν/T: **Yp=0.102,T=0.009 MeV,Eν=0.0234 MeV=2.6T**.

For each composition, the cold neutron offset is reconstructed directly from independently evaluated Fermi momenta and the commonM=939 MeV: Δ=(pFp²−pFn²)/(2M)+sqrt(pFp²+me²). It is held constant within that model. Values are54.4470184982 and54.8903926324 MeV. This is the declared local quasiparticle construction, not vacuum Q, GM1, or an EOS derivative across separately calibrated compositions.

Before any optical comparison, separate kinetic-energy-coordinate FD quadratures reconstruct both nucleon densities and total-electron-energy quadrature reconstructs net electron charge. Worst relative residual is6.30×10⁻¹⁴. The state roots and offset convention therefore pass before considering a tiny optical coefficient.

## Independent direct j and χ integration

The reviewer integrates y=(Ee−μe)/T adaptively and uses a **direct cosine** angular Gauss rule. This differs from both author methods: the former adaptive y/q comparator uses q as angular coordinate, while the new root-panel method uses fixed direct-energy Gauss panels. The reviewer does not reuse the author energy panels or root finder.

Both j and χ are independently assembled from their respective lepton occupation/vacancy and reversed/forward nuclear responses, with separate logarithmic scales. KMS is checked afterwards and is not used to construct either coefficient. Angular orders64→96 and upper tails100→180 are varied together; the largest change is3.56×10⁻¹⁵ relative. Estimated adaptive errors are at most3.53×10⁻¹³. The root-panel result, evaluated at order28/angle56/tail180, agrees with the independent adaptive/direct-cos reference within2.85×10⁻¹⁴ relative for both coefficients at both points.

| Yp | T [MeV] | Eν [MeV] | j [s⁻¹] | χ [s⁻¹] |
|---:|---:|---:|---:|---:|
|0.100|0.010|0.0300|1.427141166296×10⁻¹³|2.877189682039×10⁻¹²|
|0.102|0.009|0.0234|3.332126148074×10⁻¹²|4.501115833904×10⁻¹¹|

The independently integrated j/χ pair obeys the actual chemical-affinity KMS relation within3.94×10⁻¹³ logarithmically. All four ordinary coefficients are representable, and the log-domain accumulation remains explicit.

## Budget-edge scope and preserved failure

The reviewer explicitly panels the additional energy-budget location **Ee=Eν+μn−μp**, corresponding here to y≈3.00373 and2.60330. This location is not an explicit root in the author’s cold branch-panel finder. At the selected points the author’s electron-Fermi panels already resolve the feature, as confirmed by the comparisons above. This evidence does not certify arbitrary Eν/T, density, composition or a complete cold root list; generalized use must assess this energy-budget edge and root detection separately. No physical width is inserted by either numerical method.

The original frozen mapped72 calculation is evaluated again at the original point. Its j and χ each remain discrepant by approximately **−0.0003602258564**, exceeding the unchanged3×10⁻⁶ gate. The original mapped48→72 failure0.00662339 and its failed JSON remain unchanged. The new coordinate/root-panel agreement does not retroactively classify the old mapped quadrature as passing.

## Controls and disposition

Independent rejection checks cover nonfinite/zero energy, invalid boolean/noninteger/small orders, invalid tails, invalid adaptive tolerance, the nondegenerate-electron domain, a collapsed q-coordinate interval, and nonfinite log representation. Logarithmic underflow/overflow values retain their finite logs and return ordinary `None`; a rounded positive IEEE subnormal remains positive and marked representable. These checks do not establish extreme-log-tail relative physical accuracy.

No new failed check occurred. Initial and final source/failure hashes are saved in `results.json`; all model/failure bytes remain untouched. Acceptance applies to the original selected repair and one new local model/point. Full Urca inventory, neutrino-energy moments, arbitrary cold-domain extrapolation, nonempty radiation, scattering, finite quasiparticle widths and interacting/relativistic currents remain outside this review.

Reproduce with `python outputs/nmb-finite-t-20261015/review_cold_numerics/probe_cold.py` from the workspace root.
