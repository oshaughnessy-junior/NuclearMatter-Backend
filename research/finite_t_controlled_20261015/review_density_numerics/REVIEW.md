# Independent numerical review: prescribed-density FD weak-off element

**Accepted at selected weak-off scope.** Thirteen independent numerical checks pass in17.40s. They support the genuine FD density derivative, reversible entropy closure and compression-work normalization. They do not qualify a weak-on density trajectory, hydro, a long-time integrator or a general EOS domain. No author or upstream source was edited.

At a held-out state(T,nB,Ye)=(2.15MeV,1.03e-4fm^-3,0.287), direct adaptive **momentum-coordinate** integrals reconstruct neutron/proton densities, offset-inclusive kinetic energies, relativistic electron/positron charge/energy/pressure and entropy. These reuse the reviewed chemical-potential density roots, but integrate the thermodynamic moments separately. Forward density/energy/pressure/entropy discrepancies are≤2.23e-15.

Finite density changes of these independently integrated moments validate the author'su_n formula. Relative density steps2e-4 and1e-4 giveu_n errors6.31e-9 and1.57e-9, and entropy-density Maxwell errors1.40e-8 and3.50e-9. The decreasing errors are consistent with the central-difference truncation, and all pass the unchanged3e-7 gate. This is an independent check of the derivative/coordinate convention, not an external interacting-EOS comparison.

For the original compressionnB=1e-4→1.02020134003e-4fm^-3,Ye=.3, an independently bracketed Brent solve fixes the **momentum-integrated** entropy at its initial value. It givesT=2.024622724117697MeV, agreeing with the author's Newton EOS inversion to4.45e-16 relative. The independently integrated endpoint entropy agrees with its target to2.23e-16. This uses a different root algorithm and separately integrated entropy, rather than invoking the author's temperature-at-entropy helper.

Pressure work is then integrated directly in **linear density**, using composite Simpson rules with4 and8 intervals along those independent entropy roots. The result is+0.0711172401817474MeV/baryon. Refinement changes it by2.20e-11 relative; agreement with the independently reconstructed EOS energy increase is1.45e-12, and with the author's logarithmic-density Gauss8 integral is2.00e-12. No signed compression/expansion cancellation is used as an error estimate. This review calculates the compression branch only; it does not add an independent expansion trajectory.

## Scope and remaining gates

The mechanical first-law termP dnB/nB² has the correct sign for compression. Combiningdu=uT dT+un dnB+uY dYe with that work and the physical weak sources yields the stated temperature equation. With weak sources off, entropy conservation follows and agrees with the checked FD construction. Photons are absent from this independent selected experiment; the author's separate photon-option checks remain their own evidence.

The author's preserved initial budget stop remains unchanged. Its completed retry and this17s review are distinct numerical attempts. The review checkpoints after the held-out derivative, entropy-root and each completed work rule, and has a43s calculation bound; no timeout occurred. The successful weak-off calculation does not close the stated cost/accuracy gates for fresh full-energy weak moments at changing density. Those sources must still be independently converged and included in energy/lepton ledgers before claiming weak-on feedback.

`results.json` contains the13 gates, forward moments, entropy-root output and both Simpson work arrays. The selected author source/result/failure hashes match before and after the experiment; the immutable upstream source manifest is additionally checked at review completion. `probe.py` and `MANIFEST.json` retain the executable review and provenance.
