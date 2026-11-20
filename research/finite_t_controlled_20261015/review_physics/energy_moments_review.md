# Independent physics review: full-energy charged-current moments

15 October 2026. Reviewed `quantum_energy_moments/integrated.py`, research note, final summary, qualified versus failed numerical evidence and reported independent numerical review. Reviewer changed only new `review_physics/energy_moment*` files; no author/upstream code changed. Campaign account/window matched at start (7% used).

**Accept the qualified dilute moments and refined moderate-bath summary as finite-M fixed-state source emissivities and irradiated-slab net energy/lepton exchanges in the named free-NR allowed-current model.** The original moderate coarse failures remain failures. The heavy-M complete-inventory integrated gate remains **open/excluded**. Publication must identify the two positive slab terms as a **linear transfer decomposition**, not gross microscopic emission/absorption event counts. No time evolution, interacting EOS validation, correlated response or complete astrophysical opacity is established.

## Inventory, phase space and units

The nue source is electron capture/inverse absorption. Antinue combines positron capture/inverse absorption with crossed neutron beta decay/three-body inverse, each using the reviewed physical transfer and blocking factors. EOS factory densities, offsets and total chemical potentials match those kernels; the stale combined helper is not used. Neutrino and antineutrino affinities are opposite, not forced to zero.

For one massless helicity, `D=1/[2pi²(hbarc)³]`. Empty-radiation number and energy sources are `integral D E²j dE` and `integral D E³j dE`. The outgoing DOS is correctly included exactly once when converting an occupation coefficient j [s^-1] into volume moments. Factors 1e39 and 1.602176634e-6 give cm^-3 and MeV→erg. These are volume sources into **f_nu=0**, not irradiated/trapped net collision sources.

Each slab boundary is an isotropic incoming hemisphere. The one-face flux factor `c D E²/4` and two-face analytic solution remain correct. Slab energy/lepton entries have **per-area** units. Dividing by L gives a column-averaged volume source; local deposition or evolving composition would need spatial radiation/collision moments. Net material energy is minus the summed radiation energy gain, while material electron-lepton source is `−Rgain_nue+Rgain_antinue`. The antinue absorption cases therefore deposit energy but decrease material electron-lepton number, with the correct signs.

## Positive slab terms: essential interpretation

Write `lambda=j+chi`, `f_eq=j/lambda`, and `a=1/2−E3(lambda L/c)`. The returned positive terms are

`A f_eq a` and `A(f_left+f_right)a/2`, where `A=c D E²*1e39`.

Their difference is the correct net radiation gain. They arise from splitting the **linear transfer solution** into internally sourced outgoing radiation and attenuation of prescribed incoming radiation. They are useful non-cancelling quadrature/error scales.

They are **not** the volume-integrated gross event integrals `j(1−f)` and `chi*f`. In particular, incoming attenuation uses lambda, which includes both true absorption and the change in Pauli-blocked emission. Thus the legacy `slab_absorbed_boundary_*` output name must not be interpreted as a count of absorbed particles or true absorption reactions; the `slab_emission_*` label is likewise an emergent transfer piece, not total microscopic emission events. Computing gross events requires the actual angular/spatial occupation f and separate integrals of the two original collision terms. Net material exchanges remain correct without those gross counts.

Use explicit prose labels such as “positive internally sourced transfer piece” and “positive incident-field attenuation piece” in paper-bound evidence, or rename future outputs. This is a publication interpretation condition, not a request to alter the reviewed physical rates or their net ledger.

## Independent integrated ledgers and thermodynamic closure

Executed `energy_moment_probes.py`; all **18 checks pass**, recorded in `energy_moment_probe_results.json` with the final summary SHA.

For all four state/flavor combinations, independently assemble outgoing-minus-incoming from the four boundary flux entries, then compare number/energy gain, MeV-to-erg material energy and electron-lepton signs. Largest scaled discrepancy is **1.80e-15**. This checks the serialized accepted summary without reusing its positive-piece difference to build the reference.

For a second check, combine the **complete empty-radiation volume moments** with the matched FD EOS at each state. For instantaneous homogeneous fixed-density n,p,e+- matter without photons,

```
F=dot(Ye)=(R_antinue−R_nue)/nB
q=(Q_nue+Q_antinue)/nB
uY=Ap−An+Ae,  Ai=mu_i−T mu_i,T|n_i
uT=(Cv_n+Cv_p+Cv_e)/nB
dot(T)=(−q−uY F)/uT.
```

A separate finite displacement of the full returned EOS energy along `(dot(T),F)` reproduces `dot(u)=−q` within **2.41e-11 scaled**. This does not integrate an ODE or establish subsequent trajectory validity; rates are held only for this local directional derivative.

A useful implication is that **initial temperature rises although total matter energy decreases** in both examples. The no-photon instantaneous derivatives are approximately +2.264e3 and +5.054e6 MeV/s, while dot(Ye) is −2.596e3 and −6.501e5 s^-1. Composition energy release exceeds the energy carried away at that instant. These are conditional model diagnostics, not claimed cooling trajectories. They demonstrate why net neutrino energy loss cannot be divided by a heat capacity alone while Ye evolves, and why positive material neutrino energy deposition is not automatically a temperature-rise rate without the composition term.

## Numerical disposition and remaining gates

The separate numerical review accepts selected default/fresh integrations and the moderate targeted refinement. The 14 failed moderate coarse gates must remain visible; final moderate values use refined panels, not a retroactive declaration that coarse grids passed. Tail/internal-node qualification is sampled rather than universal. Exact static endpoint scales in the outer panel list do not impose an artificial finite-M support cutoff.

The attempted **heavy-M full-inventory energy integration failed** the strict positive-coefficient wrapper in a forbidden/underflowed band. No accepted heavy-M total result exists. Earlier component or selected-energy limits cannot be substituted for this missing complete integrated gate, and these finite-M results should not be described as a lifetime/rate calibration.

The explicit `--resume` key based only on flavor/order is insufficient for safe future reuse: state, couplings, source hashes, boundaries, geometry, energy edges, inner controls and integration version also matter. Current accepted payloads have reviewed provenance; this does not qualify that cache for a different run. Use fresh default computation for new evidence, or independently repair/test a complete cache fingerprint before resuming altered physics or controls.

Next: preserve the positive-piece interpretation in the paper note; introduce a reviewed log-domain/exact-zero-safe integrated interface to close the heavy-M gate; then build one matched FD fluid or irradiation experiment with the same named full inventory, EOS derivatives and energy/lepton ledgers. Gross event counts, continuum transport, weak-current corrections and interacting/relativistic opacity remain separate deliverables. No result here closes the external finite-temperature GM1 forward gate.
