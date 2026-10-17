# Independent physics review: combined-log NC gain repair

15 October2026. **Accept the new combined-log interface at the selected fine cold controls, with the warm/near-LTE regressions.** The original API remains excluded at T=.08MeV, and the two new coarse-grid failures remain failed. No author source, response, physical support, numerical acceptance gate or archived failure was changed.

Reviewed `nc_log_gain/log_gain.py` SHA256 `09f5f0437535f247a4bf0ca2a22298c7da96c80b7727986b98055ccd823ee4ec`, author study/refinement/control reports, preserved overflow evidence, and frozen underlying NC/response equations. This review contributes **55/55 checks in10.75s**, using a fresh q-outer/outgoing-energy-inner physical integral, actual reverse kernels and selected occupation/representation probes. Raw inputs, moments, failures and hashes are saved separately here.

## Reverse response, DOS and blocking

The bare outgoing kernel contains Eout² and has units s^-1MeV^-1 per incident occupied neutrino before final blocking. For the same neutron NC response, `S(q,−omega)=exp(−omega/T)S(q,omega)` and omega=E−Eout. Thus

`K(Eout→E)/K(E→Eout)=(E/Eout)² exp(−omega/T)`.

To compute gain in **occupation** at fixed E, multiply the reverse kernel by the density-of-states ratio `g(Eout)/g(E)=(Eout/E)²`. These factors cancel, leaving the author's KMS gain kernel. Omitting the DOS ratio would break reciprocity and FD stationarity. There is no extra neutrino spin factor or target density factor: the one-helicity convention and spin-two neutron response are already present in their respective measures.

Correct gain/loss are `[(Eout/E)² K_reverse] fo(1−fi)` and `K_forward fi(1−fo)`. For FD radiation their ratio has logarithm `omega(1/Tnu−1/T)`, independent of radiation chemical potential. NC conserves this neutrino species' particle number, so **arbitrary** common radiation mu is compatible with thermal equilibrium; setting it to zero is unnecessary. Both FD and inherited packet occupations are correctly combined in log space. The packet is a bounded .15 FD plus .45 Gaussian mixture, with its own vacancy rather than an assumed FD vacancy.

The new interface exponentiates the complete response/DOS/occupation logarithm. It does not exponentiate `exp(−omega/T)` separately and does not modify the physical kernel. Stable signed differences use expm1 relative to the larger log. For FD, the analytic occupation ratio stabilizes very small temperature departures; the independently evaluated reverse path verifies rather than imposes the underlying KMS relation.

## Exact original overflow and representation meaning

The review freshly reproduces the original `OverflowError` at T=.08,n=.0042,E=19.6,Eout=77.2MeV, cosine−.3. The exponent is720. Independently evaluated forward and reverse physical kernels have finite logs; after the DOS ratio and occupations, **log gain=−2443.55803566139**. It is an extremely small nonzero gain in this ansatz, below ordinary float64 representation.

The new return has a finite log, `value=None`, `reason=underflow`, and a nonzero sign. This is correctly distinguished from a structural/algebraic zero and from overflow. It must not be described as vanishing physical support. Likewise `representable=True` for a positive subnormal says a nonzero float exists, not that its relative accuracy is high. Equal computed gain/loss logs give zero at working precision; their flag cannot resolve an arbitrarily tiny physical non-equilibrium perturbation. The author note states that qualification.

There is no response floor or fitted thermal width. Panel widths only divide the integration domain. The old API still fails, the old overflow artifact remains unchanged, and no blanket extension of that interface's domain follows from the new log wrapper.

## Fresh independent integrated cold witness

The selected bath is T=.08MeV,n=.0042fm^-3,M939MeV,offset0; incident E19.6MeV; radiation FD(Tnu=.4MeV,mu5MeV). Our new integration **reverses the outer/inner ordering** relative to the author: integrate momentum transfer q first, then Eout over `|E−q|<=Eout<=min(cut,E+q)`. Its exact angular Jacobian is `q/(E Eout)`. It evaluates both physical kernels separately and forms complete gain/loss factors before exponentiation. No author combined-log angular/integral helper constructs these independent moments.

At q192/Eout96, independent results are:

| per-incident spectral quantity | fresh independent value |
| --- | ---: |
| Blocked out event rate [s^-1] | 179082.060723 |
| Blocked angular transport [s^-1] | 211504.585012 |
| Signed energy into bath [MeV s^-1] | 328400.438165 |
| Absolute energy transfer [MeV s^-1] | 328594.136128 |
| Gain [s^-1] | 1.84258053666e−12 |
| Loss [s^-1] | 2.51943694617e−11 |
| Net C(E) [s^-1] | −2.33517889250e−11 |

All seven moments agree with the author's selected energy192/angular-q640 result within **6.55e−8 relative**, passing the unchanged3e−5 gate. Independent q160/Eout80 agrees with that author reference within5.59e−7; the independent160→192 comparison is below4.94e−7. These are numerical consistency bounds within the shared free-NR model, not physical uncertainty bounds.

Extending the **actual absolute cutoff** by1.5 in the independent q160/Eout80 calculation changes no returned moment at the displayed float64 precision, passing1e−8. This is a finite tail-extension check, not a proof bounding the infinite tail. The nominal tail parameter alone is insufficient because the cutoff is width-dominated at this state.

The out/transport/energy moments are conditional outgoing moments per incident neutrino; gain/loss/net include incoming/outgoing occupations. They are not total fluid energy sources. A global material heating calculation needs integration over incident E with its DOS and the consistently paired radiation exchange. No global double-energy energy/conservation claim is accepted here. The1−cos diagnostic also does not by itself define a full inelastic transport closure.

## Warm, near-LTE and preserved failures

Fresh independent q64/Eout40 integrals reproduce the author's warm T2,n.002,E10,FD(3,4) moments within1.16e−9 relative and near-LTE FD(2.00002,6) moments within2.35e−9. New point probes use actual reversed responses for arbitrary-mu LTE, a1e−7 relative radiation-temperature departure and the non-FD packet. These check the physical/log arithmetic beyond a single underflow witness.

The initial cold energy48/angular144 and energy64/angular192 gain errors are **9.14173664e−5 and3.48104797e−5**, both above3e−5. The author main report still has `passed:false`; both initial failed gates are explicitly checked as failed in this review. Higher-resolution success does not retroactively accept those grids or qualify the inexpensive defaults at the cold point.

## Disposition

Accept combined-log representation and the selected fine T=.08 result with recorded controls. Retain old overflow, coarse failures, cutoff and precision qualifications. This extends one controlled free unpolarized NR neutron calculation; relativistic/current corrections, correlations, arbitrary cold-domain coverage, global radiation/material exchange and evolving transport remain unqualified. No author fix was requested or applied.
