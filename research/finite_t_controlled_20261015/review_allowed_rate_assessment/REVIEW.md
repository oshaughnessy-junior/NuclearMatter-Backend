# Independent review: stationary allowed-rate assessment

**ACCEPTED as a four-state approximation assessment, with all20 reported model failures retained.** Independent review gives108 passing numerical/normalization/classification/provenance checks and no new numerical failure. This does not accept the stationary model as1%-accurate:16 capture-moment criteria and4 composition-root criteria fail. Author numerical passes and this review's correlated checks are separate evidence. No author/upstream files were edited.

Reviewed `assess.py` SHA256`1f7df184fa062435b8f370f628cb6653c1177aac9f4fd22402f383fa4f6c669e`, manifest`349033305f9abecba42e7778de10729932e1aecc2025b89411e5e5aded96799f`. Read its note/contract/supplement and existing `weak_rates/static_capture.py`. The original transparent balance and the independently accepted three-bath grid provide immutable detailed forward references at their exact states. The supplement correctly pins the later grid review; it does not rewrite initially conditional assessment data.

## Absolute normalization and physics

My fresh Dirac trace contracts static initial/final baryon rest projectors with point V-A vertices and finite-mass charged-lepton projectors at three new cosines -.79,.13,.83. The **unaveraged** spin sum is

`32 GF² Vud² Mn Mp Ee Eν [(1+3gA²)+(1-gA²)ve cosθ]`.

It agrees with the literal trace within the declared3e-13 gate. Averaging the initial neutron spin gives the note's16 factor for beta decay. Capture averages the two initial charged-lepton spins as well, but the spin-two charged-lepton phase density restores that factor. Target density is already the full unpolarized density; no additional target spin factor belongs in the rate. Angular integration then gives

`A = GF² Vud² (1+3gA²)/(2 pi³ hbar)`.

An independent prefactor constructed from the detailed current's GF/Vud, declared gA=1.2723 and hbar matches the existing static module. This is a derivation/check within the stationary LO ansatz; the rest projectors are not asserted to form an exact finite-recoil event. The primary [Chiesa et al. Appendix A, Eqs43-44](https://arxiv.org/html/2412.04570v2#A1) also supplies the stationary spectral phase Ee*pe and opposite lepton occupations; its displayed gA/constant conventions are not silently copied into this comparison. Target blocking factors and their optional corrections are explicitly omitted here.

The approximate number integrands are A*Ee*pe*Eν² times f_minus for EC, f_plus for PC and1-f_minus for beta; energy moments add **one factor of emitted neutrino energy**, not Ee. Here Eν=Ee-Q, Ee+Q, Q-Ee respectively, Q=Mn-Mp, and mu_plus=-mu_e with total lepton mu. EC targets np; PC/beta targets nn. Beta final-electron blocking is retained; baryon final vacancies and neutrino partner blocking are absent in this empty-radiation assessment.

These are static endpoints, not exact finite-mass stationary-target kinematics: for example static PC emission starts at Q+me, whereas exact inverse absorption on a rest proton has the recoil-shifted1.806-MeV threshold. A moving thermal baryon bath has support beyond either rest-target cut. Keeping the physical mass difference does not promote the static integrals into exact-recoil rates.

All rates are s^-1 per target before density multiplication, number sources fm^-3 s^-1 and energy sources MeV fm^-3 s^-1. A single active neutrino helicity is counted; the spectral one-helicity DOS is already absorbed in these integrated moments. Rest-inclusive EOS/current energies require no additional reaction-Q or pair-rest source.

## Independent physical integrals and roots

`probe.py` uses charged-lepton **rapidity** Ee=me*cosh(y), pe=me*sinh(y), so Ee*pe*dEe=Ee*pe²*dy. This differs from the existing sqrt(Ee-me) Gaussian rule and the author's p/T QUADPACK reference. New96/144-node rules with capture upper energies100T retain actual FD occupations and beta blocking. All24 absolute allowed moments agree with the frozen assessment within5.47e-14, with independent rule refinement passing3e-10. The closed vacuum allowed beta Fermi integral also agrees before any model-error interpretation. Its uncorrected lifetime~947.37s and the detailed tree-current~949.32s are model diagnostics, not measured-lifetime calibration.

For every new approximate-root trial I independently reconstruct the free baryon chemical potentials using the relativistic FD fugacity/Bessel series and solve neutral electron-minus-positron charge by original-momentum Gauss quadrature. No saved detailed rate or approximate rate memo enters root feedback. The newly calculated roots are:

| T MeV,nB fm^-3 | Independent allowed kinetic Ye | Shift from detailed Ye |
|---|---:|---:|
|2,1e-7 original|.5753198401595205|+.0192750183573541|
|1,1e-7|.3908749908804932|+.0058182806846895|
|3,1e-7|.5800097903037986|+.0310032105080277|
|2,1e-6|.2877684002015407|+.0091406423314221|

Independent roots agree with the assessment to1.03e-14 in Ye. Final roots are tested against gross summed emission, not relative error against their zero net. Matching lepton chemical potentials precedes all rate and composition comparisons.

## Assessment conclusions and publication limits

The existing24 per-channel moment comparisons and four root criteria are classified identically: all16 EC/PC1% tests fail, all8 beta1% tests pass, and all4 |deltaYe|<=.001 tests fail. PC neutrino-energy excess spans roughly4.2–12.8%; beta discrepancies stay below.64%. These are deterministic discrepancies between the declared approximations at four exact selected states. They are not quadrature error, physical uncertainty, a general accuracy envelope or permission to rescale a channel.

The full reference jointly retains recoil/moving targets, baryon Pauli factors, weak magnetism, pseudoscalar and q² form factors. This assessment changes them together; it cannot isolate which mechanism causes an individual difference without further controlled switches. Affinity is computed from the actual free EOS at each model's own root. Empty incoming radiation and a fixed-T EM-rethermalized pair bath imply positive escaped energy/thermostat heat even at number balance; no trapped chemical or thermal equilibrium follows.

The inexpensive stationary integral remains useful with these explicit errors. It is not a replacement for the detailed source chain in a composition/energy calculation requiring those effects. No fit, calibration, opacity domain, dense-matter/mean-field/modified-Urca prescription, complete weak inventory, long-time evolution or AD fidelity is established. Model-failure gates and all existing upstream failures remain unchanged.

Reproduce from Genesis:

```
OPENBLAS_NUM_THREADS=1 /Users/rossma/miniconda3/bin/python -B outputs/nmb-finite-t-20261015/review_allowed_rate_assessment/probe.py
```

The experiment takes.13s after imports. Frozen detailed moments are read and compared arithmetically, not rerun or called a new detailed-physics validation. Manifest/source/data hashes and before/after evidence are checked separately. No installs, production/paper/git operation or author repair occurred.
