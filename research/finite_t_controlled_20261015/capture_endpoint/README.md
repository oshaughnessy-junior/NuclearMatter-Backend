# Endpoint-controlled capture quadrature

This is a numerical integration extension of the frozen `allowed_opacity/` model. It uses the same common-mass nonrelativistic response, leading allowed current, units, panel edges and chemical conventions. It changes each energy panel to `Ee=me+x*x`, with exact Jacobian `dEe=2x dx`. No physical threshold, response or normalization is modified. Default energy/angular orders are48/48. These remain selected-state quadrature settings, not a universal error bound.

Run `python capture_endpoint/check_mapped.py` from the campaign root using the existing NumPy/SciPy environment.34 checks pass: eight archived independent q/Ee references, a freshly recomputed most-sensitive threshold point, refinement, chemical detailed balance, heavy-mass static normalization and invalid controls. Maximum order96 error versus the eight independent references is2.43e-14. The references use the same physical response but different coordinates and adaptive quadrature; this tests integration accuracy, not independence of the physical ansatz.

The initial default24 map still missed a3e-8 gate at antineutrinoE1.8MeV (8.93e-8 relative error); `initial_order24_failure.json` preserves the failure. Increasing both quadratures to48 passes; a96 refinement is retained. The frozen raw-Ee quadrature at24 erred9.29e-6 at that state. Frozen code is unchanged. No full-domain or cold/extreme-tail qualification follows.

This leaf supplies capture/inverse absorption only. Use the independently reviewed crossed decay channels for a complete charged-current reaction inventory in the declared free-quasiparticle model. Independent review of this numerical extension is pending.
