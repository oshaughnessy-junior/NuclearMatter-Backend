# Independent review: endpoint-mapped capture quadrature

15 October 2026. **Accept the mapped leaf as a numerical repair at its qualified selected points**, retaining the frozen leading-current and nonrelativistic-response physics. This supersedes the default-order numerical blocker at the independently reproduced antineutrino endpoint example when callers explicitly use the new mapped evaluator.

`probe_mapped_capture.py` executes **9 independently passing checks** at E_antinu=1.79,1.8,1.81 MeV in the degenerate T=3 MeV, `(nu_n,nu_p,mu_e)=(20,5,5)` MeV state. The two bracketing energies are absent from the parent's archived endpoint sample. Results and source hashes are in `mapped_capture_review.json`.

The independent reference uses adaptive **electron momentum** integration, with `E_e p_e dE_e=p_e² dp_e`, and a 128-node angular rule. It does not reuse the mapped evaluator's electron variable, energy panel generator or integration nodes. The earlier independent reference also verified angular 64→128 convergence at E=1.8 MeV to 4.4e-16 relative.

At all three points, default mapped48 agrees with the independent momentum calculation inside 3e-8, mapped48→96 refinement passes that gate, and chemical KMS holds inside 3e-11. The exact map `E_e=m_e+x²` carries the Jacobian `2x`; the code preserves the optical prefactor, angular current factor, response arguments, blocking factors and original physical panels. It removes an integrable electron-mass endpoint corner without introducing physical threshold smoothing.

The parent's original mapped24 failure remains valid evidence: an exact endpoint map alone does not guarantee adequate finite-order accuracy. Selecting default48 is supported by the selected convergence/refinement tests. No universal order48 accuracy certificate follows, especially at stronger degeneracy, narrower recoil structure or different thermal tails.

The frozen old `allowed_capture.py` remains unchanged and retains its measured default24 extension error. Callers must opt into the mapped leaf; this review does not imply the old function was repaired in place. The crossed/combined antineutrino wrapper still imports the old capture evaluator unless separately adapted and reviewed. A combined physical-channel calculation must therefore select qualified constituent evaluators explicitly.

All underlying leading allowed-current/common-mass NR limits remain. No relativistic current, interaction correction, arbitrary-energy domain, numerical rate sensitivity or fluid conservation claim is added. No author files changed.
