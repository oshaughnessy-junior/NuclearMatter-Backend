# Independent numerical review: FD fluid element

**Accepted at restricted scope:** the matched free-NR FD material energy/composition closure, correctly ordered explicit-Euler feedback, and the recorded short isochoric transparent-source demonstration. No implementation defect was found in the checked contract. This does not qualify a time-accurate cooling trajectory or establish a general EOS/rate domain.

`eos_ledger_probe.py` supplies48 passing independent checks in0.515s. Momentum-coordinate adaptive integrals independently reproduce neutron/proton densities and offset-inclusive energies, relativistic electron/positron charge and total energies at both the initial and four-step endpoint. The energy integrals use spin-two1/[pi²(hbar c)³] phase space, retain neutronQ and electron rest energy, and include both electron charge signs. Their reconstructed endpoint energy plus recorded escaped energy matches the independently integrated initial energy. All48 residuals are at most2.23e-15 on their named relative/absolute scales. Pinned author sources/results remained unchanged.

For each of the six recorded Euler steps, source-stateT/Ye equal the pre-step state; u and Ye updates have the required signs. Raw stored source-panel sums independently reproduce the reported material number/energy rates: electron-neutrino escape lowersYe, antineutrino escape raisesYe, both remove material energy. Accumulated energy and electron-lepton counters reconstruct their reported values. These checks audit arithmetic and source provenance; re-summing stored panels does not independently qualify the physical quadrature.

At held-out(T,nB,Ye)=(2.003MeV,1e-4fm^-3,0.297), `selected_source_probe.py` adds4 passing differential-source checks atE=8MeV for both species. Internal order48/tail40 versus64/tail50 agrees for j and chi within the unchanged3e-6 gate. This is a selected spectral-point check only.

## Preserved review budget stop

`probe.py` attempted independent full-energy log(1+E) integration with an extended140MeV cutoff at the actual endpoint and the held-out state. Its explicit88s runtime bound was reached during the last held-out12-node calculation. Unfortunately the completed endpoint12-node and held-out8-node arrays had not yet been serialized. **Those unsaved values are not review evidence and no new full-energy convergence claim is made.** `initial_budget_stop.txt` preserves this limitation. No expensive replay was undertaken; the accepted EOS/ledger checks were reproduced in a separate cheap probe. Approximate numerical work is88+0.515+1.117≈89.6s, within the90s allocation. The budget stop is not evidence of a failed author quadrature gate, nor evidence that it passed independently.

## Interpretation and remaining gates

The four-step endpoint isT=2.004381407217MeV,Ye=0.294891996756,u=8.416314695636MeV/baryon, with0.085606567923MeV/baryon escaped. Independent energies/charge support its thermodynamic consistency. The source is reevaluated at distinct pre-step states, so this is genuine local physical feedback.

The two/four-step differences inYe(2.77586e-5) andu(5.06315e-4MeV/baryon) remain material, despite the much smaller temperature difference. Exact ledgers follow the conservative update and cannot establish temporal accuracy. A third resolution or higher-order method, with independently converged new-state moments and source-manifest checks, is required before trajectory accuracy claims. There is no trapped radiation, inverse-channel evolution, density work or full hydro claim. Existing author's sampled endpoint/source gates and earlier reviewed kernels support the narrow prototype; this review adds no independent full-energy gate beyond them.

All author code, upstream kernels and previous review directories were read-only. Raw results, the bounded-attempt source, preserved stop, selected-source results and SHA manifest are retained here.
