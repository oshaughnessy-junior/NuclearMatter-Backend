# Independent weak-ON prescribed-density element review

**ACCEPTED only for the selected fine short trajectory.** The independent review has **347/347 new passing checks**: 86 in RK4-two-step, 192 in RK4-four-step and 69 in cross-method/endpoint analysis. There are no new failures. The author's global summary correctly remains **104 PASS / 3 retained coarse FAIL**. This review does not change those failures or qualify the coarse trajectory.

The frozen author MANIFEST is `5e78fc0eb5ff0df6b8274b01edb7e874c5a3ae1a98469cc8b9cffa7141f3d0d4`, element.py is `6da23ca9ce8c56e12b9e03c68e403f43c7fa049bfecb4fff05e380cffae962be`, and CHECKPOINT is `41f98048ab6c0b436a0dcfa2cb9ee41f698b1ff5b5e90c7846be963e7ee68c1b`. They were checked before candidate comparisons and remain unchanged. Source and package pins are recorded in the final review checkpoint.

## Contract and independence

The controlled element begins at T=2 MeV, nB=1e-7 fm^-3, Ye=.3, follows n=n0 exp(.25 t) for .002 s, and includes free particle-only physical n,p, equilibrium e−/e+ and trapped two-polarization LTE photons. Muons, including muon pairs, and antibaryons are absent. Incoming electron-neutrino and antineutrino occupations are zero. The only admitted weak channels are electron capture, positron capture and neutron beta decay with the accepted matching full-current/mass conventions. No full inventory or interacting EOS is implied.

The author uses explicit midpoint on specific rest-inclusive energy u=e/n and Ye. The independent code uses RK4 with two/four steps, independent assembly of source/ledger arrays, and a bracketed temperature-energy inversion at every stage. Both methods intentionally share the accepted physical event kernels and EOS. This is a different numerical time method, not an independent physical model, amplitude, or density-variable ansatz. The initial preparation anticipated a T-equation author; the immutable revised validation lock corrects that assumption **before any candidate comparison or physical calls**.

The actual physical EC/PC/beta integrals are recomputed at every stage. The old fixed-nB source convenience wrapper is never used. Each target-density check uses that stage's actual nB and Ye; maximum new density discrepancy is below 1.88e-10 at the unchanged 3e-8 source gate. Pure accepted function modules are imported with unique names; no upstream study/driver executes or writes data. Exact installed modules remain lepton `baeef78b174ecc0bcc1ab6016c839a8e8dc6a2ba915541d7fd994d994f01e719` and baryon `f4c007f400df346811d4ce759ee8d96d6b74170c7ef1d0c4f5bc063f8018f65b`. Campaign masses are explicitly mn=939.56542052, mp=938.27208816, me=.51099895069 MeV and hbarc=197.3269804 MeV fm.

## Numerical results

The independent four-step endpoint is T=1.996561935027137 MeV and Ye=.31518880207481054. Against the author's eight-step midpoint result, maximum sampled temperature and Ye differences are 7.512e-9 MeV and 1.839e-7, below the locked absolute 2e-6 gates. Escaping energy and number differ relatively by 2.367e-6 and 1.952e-6, below the locked 1e-3 escape gate. Independent RK4 two-to-four differences are 2.98e-13 MeV in T, 1.068e-10 in Ye and 3.03e-10 relatively in escaping energy. These are selected numerical comparisons, not a uniform time-order certificate or a physical uncertainty estimate.

The final material EOS is independently checked through the previously frozen original-momentum baryon oracle and explicit momentum e−/e+ integrals/density roots. Six forward/response fields agree within 1.10e-13. Initial material forward values also reproduce the accepted independent binding oracle. Fresh initial source moments reproduce the pinned accepted binding anchor before any scientific time interpretation. Fine endpoint physical source refinement is at most 5.229e-9 against the unchanged 3e-6 gate.

The integrated escaping energy is .3001818519577001 MeV/baryon and pressure work is .007214625612389206 MeV/baryon. Energy and electron-family lepton ledgers close at roundoff, but their shared RK4 weights make those algebraic identities insufficient to establish temporal accuracy. An additional independently assembled composite Simpson rule uses the actually computed RK4 node sources, with no interpolation of rates or new frozen-bath shortcut. It agrees with RK4 escaping energy, net lepton change and pressure work within 4.30e-10, 8.61e-10 and 2.50e-11 respectively, at the locked 3e-5 gates. All quantities use work/escaping/thermal/gross-number scales; no rest-mass denominator or zero closed-cycle normalization hides errors.

The matter entropy change is **−.12945760606763201 per baryon**. A negative change is allowed for transparent losses. The material identity tested is

`d(s/n)/dt = [-Q − (mu_p+mu_e−mu_n) R]/(n T)`,

with `R=R_beta+R_PC−R_EC`. Its independent Simpson integral agrees with the actual EOS entropy change within 2.157e-9 relative to the nonzero entropy change, below 1e-4. This is not a total radiation-plus-matter entropy or H-theorem certificate: no escaping radiation entropy can be recovered from number and energy moments alone.

The reconstructed temperature chain also matches

`Cv*Tdot=[(e+P)/n−e_n]*ndot−Q−e_Y*R/n`.

Physical rest masses appear once in the EOS and event energy losses; no extra Q heating is inserted. Stage-level direct and n/p/e species energy ledgers, actual density checks and the separate emitted electron-family number moments all pass.

## Preserved failures and limits

The author's two-step endpoint Simpson escape error 3.960065e-5 and net-lepton error 1.077067e-4 fail the unchanged 3e-5 gates. The two-to-four-step Ye difference 2.257e-6 fails the unchanged 2e-6 gate. All three remain in the frozen global FAIL summary and original raw files. Four/eight-step individual gates and the finer selected cross-method comparisons are accepted separately.

The model imposes rapid internal EM pair equilibrium and LTE photons without solving photon transport or plasma equilibration. Escaping neutrinos see empty incoming occupations. The duration is only .002 s and the imposed density changes by .0500%. Neither this review nor the author establishes long-time stability, full hydro, full weak/flavor inventory, continuum neutrino transport, AD, interactions, empirical uncertainty or broad parameter admission accuracy.

## Reproduction

Use existing junior_tools Python with `OPENBLAS_NUM_THREADS=1`, `-I -B`: run `rk_reference.py 2`, `rk_reference.py 4`, then `analyze.py` in fresh processes. The actual physical experiments took 27.87 and 58.73 s, each below 120 s; analysis took less than one second. No installations, source fixes, dependency changes, git/production/paper modifications or upstream driver imports occurred. Exact baseline quota identity was asserted before both event experiments: 26% used, same account/window/reset. `CHECKPOINT.json` pins every review record and unchanged author/source evidence.
