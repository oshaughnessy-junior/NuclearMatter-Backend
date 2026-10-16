# Independent installed free-matter binding review

ACCEPTED at the selected controlled scope: **146/146 new checks pass**, with no new failed gates. This is a binding/EOS, weak-OFF prescribed-density path, and instantaneous three-channel source review. It does not qualify a weak-ON trajectory, an interacting EOS, a broad admission rectangle, or an AD chain.

The author froze before candidate comparisons: MANIFEST `c3db0aacf894cb688de7358d49a0931b6980473c399a50bcddc32049992e4512`, matter.py `b57e8c602d94ade33bee592c2b13ce9980915f3925ab78e2405094e34e06ab27`, weak_point.py `1cd2760691e35f893b3da538249075a6aacc978d30895f0a3c0ec7387bc3554d`, cycle.py `1e979c79aca77acc0a015c5cd5e3f792a108714f32909dac9ae7ad061306298c`. All 119 manifest-covered source/evidence/package paths were verified before comparison and remain byte-identical after this review. The exact path count is recorded by the checkpoint rather than used as a physics gate.

## Independent evidence

`VALIDATION_LOCK.json` precedes candidate imports/comparison and fixes four author-matched baths plus fresh `(T,nB,Ye)=(2.37,2.8e-7,.41)`, three entropy-root times, source controls, and gates. `probe.py` reuses the previously frozen independent original-momentum baryon oracle (not the installed implementation); scaled-Bessel fugacity sums independently check each matched baryon population, pressure and energy. The lepton oracle is a new original-momentum e−/e+ integration with independently solved signed net-charge chemical potential. Its FD susceptibility, energy derivatives and charge response use separate explicit particle/hole integrands. This tests port/binding numerics within the same free-gas physical model, not independent physical-model validation.

All unperturbed forward values precede response checks. Maximum forward relative discrepancy is `2.47e-13`, Bessel discrepancy `6.67e-16`, and analytic total Cv/e_n/e_Y discrepancy `7.24e-14`. Re-solved independent energy finite differences at the fresh state agree within `5.19e-7` at the unchanged `3e-5` derivative gate. Both public modules are loaded from the real owned package with exact accepted lepton/baryon hashes and physical campaign masses: mn=939.56542052, mp=938.27208816, me=.51099895069 MeV, hbarc=197.3269804 MeV fm. Default NMB mass rounding is not silently substituted.

Independent fixed-entropy roots at t=.125,.25,.75 s give T=2.361465533615707, 2.5297599426080444, 1.581587490249325 MeV. They agree with the saved 64-step RK4 cycle within `8.12e-9`; no candidate EOS is used in these roots. A different pressure-work integral runs over log density on the independently constructed adiabat. Gauss8/12 results are 11.387497561204574/11.387497561926008 MeV/baryon; refinement is `6.34e-11`. Comparison to the actual compression-branch energy change has relative error `1.315e-7`, normalized to the **11.3875 MeV/baryon nonzero compression work**, never rest-mass energy or net closed-cycle zero. This is a selected weak-OFF cycle accuracy check, separate from the algebraic first-law identity.

`source_probe.py` makes fresh physical EC/PC/beta event-phase calls at the locked original T2/nB1e-7/Ye.3 bath, using alternate orders EC24/40/10 tail65, PC30/48/11 tail65, beta20/24/9/14 electron/direct-p tail85. All separate number and energy moments agree with the frozen author's fine profile within `2.61e-10` at the unchanged `3e-6` gate. Reconstructed rest-inclusive n/p/lepton species energies, direct material losses, charge, baryon count and electron-family lepton number pass; instantaneous Tdot agrees within `2.21e-10`. The phase kernels and full current were previously accepted; this is fresh quadrature and independently assembled stoichiometric/EOS coupling, **not** a new independent amplitude derivation or a whole weak-ON evolution.

## Physical and bookkeeping contract

The matter consists of particle-only free n,p plus equilibrium e−/e+ pairs. Muons are absent, including muon pairs. Photon-ON means two-polarization trapped LTE radiation, zero chemical potential and no photon escape; photon-OFF means that component is absent. No antibaryon or interacting matter certificate is implied. Total chemical potentials and matter energy include physical rest masses; weak event losses already contain the neutron/proton rest-energy difference and require no extra Q term.

At fixed species mass, `h_i=mu_i−T (dmu_i/dT)|n_i`; thus `e_Y=n(h_p−h_n+h_e)` and `e_n=(1−Y)h_n+Y(h_p+h_e)`. The corrected identity is

`e+P−n e_n = T s + T Σ_i n_i (dmu_i/dT)|n_i`.

The initial author report with **48 passing and 8 wrong-sign CHECK-only failures** remains failed and byte-identical. Its reporter exception and LOCK timestamp correction are also preserved. The initial check failure is not erased or counted as a physical source failure. The currently corrected identity is independently checked using thermal normalization at all five reviewed states.

The source convention is `R=R_beta+R_PC−R_EC`, `Ydot=R/n`; the prescribed-density temperature equation is

`Cv*Tdot=[(e+P)/n−e_n]*ndot−Q−e_Y*R/n`.

Energy accounting, a thermostat-free weak-OFF adiabat, and instantaneous weak source signs are distinct tests. No total-inventory, long-time relaxation, hydro, continuum neutrino transport, uncertainty, or performance qualification follows.

## Reproduction and limits

Run the two scripts with the existing junior_tools Python and `OPENBLAS_NUM_THREADS=1`, `-I -B`. `probe.py` took 19.40 s; `source_probe.py` took 3.67 s. No dependencies, upstream files, package sources, git state, or production checkout were modified. The strict baseline quota identity was checked before each experiment (26% used, unchanged codex weekly window/reset). Final hashes and unchanged-source proof are in `CHECKPOINT.json`.
