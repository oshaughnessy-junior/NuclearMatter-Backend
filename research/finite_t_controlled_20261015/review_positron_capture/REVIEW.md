# Independent numerical review: free positron capture pilot

**Accepted at selected scope: 43/43 new review checks pass.** No author source, original coarse failure, or dependency was edited. Author files were hashed before and after; hashes are unchanged. This review accepts the named finite-temperature positron-capture/inverse pilot, with the limitations below. It does not accept total neutrino-energy moments or a combined antineutrino inventory.

## New physical state and forward qualification

Fresh density-matched free npe bath: T=1.7 MeV, nB=2.3e-7 fm^-3, Yp=Ye=.27, muons absent. Total chemical potentials are mu_n=924.692044991338, mu_p=921.7112989721977 and mu_e=.49775430121148906 MeV. Positrons use mu_plus=-mu_e. Independent momentum-coordinate adaptive integrals reproduce neutron/proton densities and NET electron charge before rate comparisons. The new electron and positron densities are 1.4971707036652943e-7 and 8.761707036653201e-8 fm^-3, respectively: the pair ratio is .5852176385, not a pair-negligible state.

## Fresh differential and absolute total comparisons

Two new differential points (Eanti, Epositron, cosine)=(5.7,4.31,-.37) and (3.6,2.17,.61) were evaluated using the frozen, separately authored original p_perp/p_z delta-shell oracle and explicit antineutrino/positron v spinors. Its coordinate, spinors, vertex and contraction are independent of the response energy/q-axis trace algorithm. Order16/tail50 to order28/tail70 refinement passes; absolute j and chi agree with the response within 2.18e-13 relative error. Direct baryon shell residual and separately accumulated j/chi KMS checks pass.

The fresh incident Eanti=5.7 MeV total was then evaluated by the independent moving-target two-body CM event integral, using exact boosts, Møller flux, explicit reciprocal spinor amplitudes and all three actual FD factors. Separate inverse occupations generate j directly; KMS never generates it. Initial proton spin averaging in this event method is paired with the spin2 target density. The response uses an unaveraged baryon trace with per-spin FD occupations; agreement therefore checks the absolute spin/phase-space normalization without a fitted scale.

| Method | j (s^-1) | chi (s^-1) |
|---|---:|---:|
| CM 8/8/16/8 | .4680014625542065 | 3.1051870497288254 |
| CM 12/12/24/12 | .46800146261824543 | 3.105187050153681 |
| Response outer16 | .46800217979115954 | 3.105191808591836 |
| Response outer20 | .4680014722218064 | 3.105187113873188 |

CM refinement is 1.37e-10; response outer16→20 refinement is 1.51190e-6, against the unchanged 3e-6 gate. Finest absolute agreement is 2.05204e-8 for both j and chi. These are numerical differences inside the same physical current model, not a physical uncertainty. Response controls are angular16/inner-energy28/phi8/baryon-tail70/outer-tail32; CM tail60. The complete fresh numerical experiment took 53.40s. This is an independent total-lepton-phase-space check at ONE new incident energy/bath, not a full neutrino energy moment. It does not establish a fresh independent uniform tail certificate or arbitrary outer-node accuracy. The original author isolated-tail evidence remains useful only at its original selected state.

## Current, source and control audit

The [Guo et al. primary current prescription](https://arxiv.org/html/2006.12051v2), Eqs26–29 and Appendix B, was consulted. The free specialization retains vector/axial, weak magnetism, induced pseudoscalar and physical-q² form factors with bare average nucleon mass. Three additional transfer vectors give agreement between the literal reciprocal adapter Gamma_pn(q)=Dirac-adjoint Gamma_np(-q) and the separately authored current implementation. Positron v spinors and antineutrino ordering are retained, without an incident helicity average or an extra positron spin factor.

The direct FD ratio is exp[(mu_n-mu_p-mu_e-Eanti)/T]. This validates the anti chemical sign and is checked only after separate positive coefficients exist; it is not an accuracy or normalization proof. The occupation source is j(1-fanti)-chi fanti. Neutrino one-helicity DOS belongs in later neutrino-energy moments, not again in j/chi. A literal reaction reconstruction conserves electric charge, baryon number and electron lepton number, with material Ye increasing for net n+e+ capture and radiation Le decreasing. Rest-inclusive matter energy must lose the emitted neutrino energy without an extra Q source.

The exact rest-proton inverse threshold is 1.8060662080187608 MeV. It is not a finite-T global spectral cut; moving-target support is handled by exact invariants. The original author's sub-rest-threshold witness was source/evidence audited rather than rerun here. Twelve fresh malformed-input probes reject nonfinite/negative kinematics, bad cosine/rest lepton energy, invalid orders/tails/budget and unphysical charge fraction. No control blocker was found.

## Preserved failures and remaining gates

The original six coarse numerical failures are explicitly retained and counted as failures in the pinned author evidence. They are not reclassified by this review. This review adds no failed physical/numerical gates or incomplete runs. The previously preserved syntax-only isolation-driver failure is also unchanged.

Acceptance is limited to the author's selected bath/energy evidence plus this new bath, two differential points and one integrated incident energy. Full Eanti number/energy integration, complete neutron-decay+positron-capture inventory on one bath, material feedback, derivative accuracy, extreme cold/high degeneracy, exact q=0 limits, radiative/Coulomb physics, mean fields and external physical calibration remain OPEN. Source methods share the declared current physics; coordinate/spinor independence does not make its physics independently measured. No source change is requested.
