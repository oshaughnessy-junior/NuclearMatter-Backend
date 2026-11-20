# Independent physics review: crossed neutron beta decay

15 October 2026. Reviewed `crossed_opacity/crossed_decay.py`, technical note, 39-check result payload, preserved failures, and frozen-capture endpoint investigation. Reviewer changed only `review_physics/`. Account/window matched campaign guards at start (4% used).

**Accept the named neutron-decay and three-body-inverse channel within the common-mass free-FD nonrelativistic response and leading allowed-current ansatz.** Crossing, occupancy signs, absolute normalization and shared antineutrino chemical parameter are consistent. This completes the three named branches of the earlier static electron-channel inventory within this ansatz, not every conceivable weak process at arbitrary offsets or a full relativistic/interaction model.

## Crossing and angular trace

For emitted e− and antineutrino, the baryons lose **sum energy and sum momentum**: `omega=−Ee−Enu`, `q²=pe²+Enu²+2pe Enu c`. The emission response must therefore be n→p at this negative transfer. Three-body absorption uses reversed p→n at the opposite transfer. The code uses exactly these directions. The class contract deliberately differs from `OpticalState`: `DecayState` orders the spontaneous direction; callers must preserve that distinction.

The beta-decay trace is `Tr[(slash(pe)+me) gamma^mu(1−gamma5) slash(k_antinu) gamma^nu(1−gamma5)]` with physical outgoing momenta. Its symmetric 00 and spatial traces give the same `gV²(1+beta c)+gA²(3−beta c)` weight verified by the earlier independent gamma/Pauli matrix probe. Its cosine is now the **outgoing electron–antineutrino** angle, while the response uses their momentum sum. Retaining the angular factor with a difference-momentum response would be a different, incorrect channel. The leading negative electron–antineutrino correlation is consistent with the primary [aCORN angular-correlation study](https://arxiv.org/abs/2012.14379); no measured lifetime calibrates the current normalization.

Final electron vacancy multiplies spontaneous beta emission; initial electron occupation multiplies inverse absorption. Neutrino blocking remains outside j/chi in `C=j(1−f)−chi*f`. The KMS ratio is

`S_np(q,−Ee−Enu)/S_pn(q,Ee+Enu)=exp[(−Ee−Enu+mu_n−mu_p)/T]`.

Multiplying by `(1−fe)/fe` cancels Ee and gives `mu_antinu=mu_n−mu_p−mu_e`, as implemented. Positron capture has the same chemical parameter, so summing its positive j and chi with decay preserves reciprocity. The logarithmic sum avoids losing a suppressed constituent to numerical underflow.

## Absolute normalization and endpoint repair

The same K, spin-two Fourier convention, one-helicity neutrino convention and charged-lepton measure apply. Distinct e− and antineutrino particles introduce no identical-particle symmetry factor. The dilute heavy-mass response enforces `Ee=Q−Enu`, giving the existing bare static spectrum for `0<Enu<Q−me`; no factor two or extra Q enters the prefactor.

Recorded spectral heavy-mass deviations are at most **2.39e-6** at the tested energies. Warm integrated number/energy moments differ by **5.53e-6** and dilute-lepton moments by **2.44e-7**. The bare approximately 947.37 s static lifetime is an appropriate internal normalization diagnostic, not the measured neutron lifetime. Radiative, Coulomb and complete current-recoil corrections remain omitted.

The replacement `Ee=me+x²`, with exact `2x` Jacobian on each panel, removes a mathematical square-root endpoint corner. It does not alter the physical response or introduce a broadening scale. This is a sound repair for the preserved degenerate-endpoint convergence failure. The separate adaptive-reference absolute-tolerance failure was also correctly diagnosed: small-tail relative discrepancies require an absolute error budget smaller than that tail. Neither repair adjusts physical normalization.

## Independent matched-EOS probes

`crossed_eos_probes.py` writes `crossed_probe_results.json`. At a finite-M state constructed from the new charge-neutral FD nucleon EOS (`T=2 MeV, nB=1e-4 fm^-3, Ye=0.3`):

- Decay and capture factories agree on the antineutrino chemical parameter within **3.56e-15 MeV**.
- Crossed j/chi and the combined positron-capture-plus-decay j/chi obey KMS at Enu=0.4, 1 and 2 MeV, including above the static endpoint. Largest logarithmic residual **1.78e-14**.

All seven checks pass without a full suite replay. This demonstrates compatibility of the independently constructed EOS chemical potentials and the optical signs; numerical KMS alone does not certify absolute normalization, which relies separately on the static limit and current/measure audit.

## Integration conditions and remaining blocker

Finite-T recoil removes exact cold/static support zeros in the quadratic-dispersion model. A tiny nonzero coefficient does not demonstrate rapid thermalization, and responses requiring nucleon p~M are outside the physical NR domain. This is thermal/recoil phase space, not collision-induced spectral widths. A complete box must still check rates/timescales and energy-grid conditioning rather than infer equilibration from formal support.

The read-only frozen-capture study finds a concrete **extension-domain numerical limitation** in the imported capture integrator: at the quoted degenerate state and antineutrino energy 1.8 MeV, default order 24 differs from the independent reference by **9.29e-6**, exceeding its earlier selected-state 3e-6 target. Refinement reduces it. This does not invalidate the crossed implementation or earlier selected capture checks, but an unqualified default-order combined scan is not accepted there. Apply a separately reviewed endpoint map to capture, use demonstrated convergence controls, or restrict the state/energy domain before wider coupled runs.

For a matched FD fluid element, use the new occupied FD baryon EOS, not classical `3T/2` baryon energy. Number and energy moments of the **same named complete channel set** must update composition, matter energy and escaped radiation. A two-state demonstration can proceed after capture endpoint convergence is established at its actual integration nodes, all channel moments converge, and NR-domain diagnostics are recorded. Full current 1/M corrections, weak magnetism, interactions, muon channels, scattering and pairs remain separate physics gates.
