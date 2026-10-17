# Free relativistic capture coefficients: selected energy pilot

This research leaf integrates the actual full-current, free particle-hole differential kernel over the charged-lepton final phase space. It is **capture and inverse capture only**. In particular, antineutrino capture on protons would not include crossed neutron beta decay; no complete antineutrino source is claimed. No GM1, interactions, antinucleons, nuclear targets, or hydrodynamics enter. Upstream EOS/tensor independent reviews were pending at construction; local agreement does not itself qualify them.

## Conventions and implementation

Physical neutron/proton and electron/muon bare masses are inherited from the frozen vacuum leaf. `matter_state(T,nB,Ye,Ymu)` independently solves particle-only free relativistic baryon densities and exact pair-lepton net densities, with `Yp=Ye+Ymu`. All chemical potentials include rest energies. There are no photons. The muon fraction and electron fraction are independent; their chemical potentials are never equated. A zero net muon fraction still includes thermal muon pairs.

The current is exactly the declared Guo vacuum form-factor model of [Guo et al., Eqs26–27](https://arxiv.org/html/2006.12051v2), including vector, axial, weak-magnetism and pseudoscalar contributions. All switches and pole masses are recorded per result. This inherits the independently derived vacuum comparison to [Strumia–Vissani, Eqs4–7](https://arxiv.org/html/astro-ph/0302055). Weak normalization uses the physical bare mean nucleon mass; there is no effective-mass substitution.

With `omega=Enu-El`, `q=|kν-kl|`, and the full unaveraged baryon trace, the underlying tensor has normalization

`W_abs^{ab} = [1/(16 pi² q)] integral dEa dphi T^{ab} fa(1-fb)`.

The inverse tensor retains the same current orientation and separately integrates `fb(1-fa)`. The outgoing lepton azimuth is already integrated. For `pl=sqrt(El²-ml²)`, the differential rates per `dEl dcos(theta)` are

`dchi = GF² Vud² pl/(32 pi² Enu hbar) * L_ab W_abs^{ab} * (1-fl)`,

`dj = GF² Vud² pl/(32 pi² Enu hbar) * L_ab W_inv^{ab} * fl`.

Both are integrated independently in log space, retaining finite logs when ordinary rates underflow. There are no rate floors or discarded underflow channels. Physical empty support is separately labeled. The integrated units are `s^-1`; these are occupation relaxation coefficients, not cross sections or volume emissivities. One incident neutrino helicity is used, with no initial neutrino spin average. Target spin two is already present in the baryon EOS and unaveraged trace. The corresponding neutrino number phase measure for a future incident-energy integration is `Enu² dEnu/[2 pi² (hbar c)³]` per species.

The massive-lepton endpoint is integrated with `El=ml+x²`. Exact support is recomputed at every quadrature node, including timelike transfers. Thermal, degeneracy and recoil estimates place numerical panel boundaries only; they do not broaden support or change occupations. Each run records every panel, the finite upper energy cutoff, and inner/outer orders. There is no cached state, rate, tensor, or density substitution. A finite cutoff requires explicit tail convergence; it is not claimed to be an analytic infinite-domain integral.

For electron/muon neutrino capture the affinity is `mu_l+mu_p-mu_n`; KMS requires `log(j/chi)=-(Enu-affinity)/T`. The charged antiparticle would use effective potential `-mu_l`, the ordered target pair `p->n`, and affinity `-mu_l+mu_n-mu_p`. KMS is a diagnostic but cannot prove quadrature accuracy because its ratio is pointwise constant.

For a radiation source `R=j(1-fnu)-chi*fnu`, material energy exchange is `-Enu*R` per neutrino phase measure. Material electron/muon lepton number changes by `-R` for its neutrino and `+R` for its antineutrino. Rest masses are already included in material EOS energy; no separate reaction-Q heating may be added. A finite-density EOS/transport evolution must use these integrated signed exchanges and its actual new composition, rather than frozen coefficients.

## Independent absolute anchor

`dilute_anchor.py` integrates the full source-invariant vacuum cross section over actual FD targets using target kinetic energy and target angle, independently of the medium kernel's outgoing-lepton and response coordinates. In the material rest frame,

`chi_vac = n_target c <(1-v_target cos(theta)) sigma_vac(Enu*(Ea-p cos(theta))/ma)>`.

The average uses the spin-two target EOS density, checked by its own Laguerre momentum measure; it does not replace the density by a classical estimate. It neglects final lepton/baryon blocking, so the comparison is only made in a deliberately dilute state. Residual blocking and finite quadrature error must be separated from physical-model uncertainty. This is an absolute normalization check with an already declared current, not calibration to a fitted factor.

Selected numerical results, refinement and limitations are appended at freeze. This leaf must not be presented as an externally validated neutron-star opacity or a full neutrino inventory.

## Frozen selected results

Both pilots use `nu_mu+n -> p+mu-`, `Enu=150 MeV`, and the complete declared vacuum current. EOS roots are freshly solved for each state. The free EOS was independently accepted while this leaf was being constructed; the tensor/oracle third-party review was still pending at this leaf's freeze. No results from that pending review are assumed here.

| Material state | Absorption chi, s^-1 | Inverse emission j, s^-1 |
|---|---:|---:|
| T=2 MeV, nB=1e-4 fm^-3, Ye=.29, Ymu=.01 | 2402841.528873935 | 2.407757975770727e-6 |
| T=.2 MeV, nB=1e-15 fm^-3, Ye=.3, Ymu=0 | 2.3663596931957454e-5 | ordinary unrepresented; log(j) about -767.96347 |

The warm state's last joint refinement changes both coefficients by4.06e-12 relative. Its independent energy cutoff, inner tail, inner order, outer lepton order, angular order and periodic-azimuth controls meet this pilot's1e-7 relative numerical tolerance. The conservative recorded maximum versus the finest warm baseline is1.09e-8, from a lower outer-energy-order azimuth check. These tolerances concern selected numerical values, not physical model accuracy.

For the dilute state, the independent moving-target FD vacuum-flux anchor is `chi=2.3663596774394763e-5 s^-1`. The independently integrated target density agrees with the actual free EOS density within5.5e-14 relative. Its highest two anchor refinements agree at represented precision. The finest lepton-integral result differs by6.66e-9 from the anchor, while the last16-to20 outer refinement changes it by6.28e-7. The selected absolute/refinement tolerance is3e-6. This is a declared pilot tolerance after an exploratory profile, rather than an inherited published accuracy requirement. The final20 run and separate independent-worker oracle were held out from its exploratory coarse sweep.

The cold coarse orders4/8/12 fail that same absolute comparison by23.86%,0.4481%,5.79e-6 and remain explicitly unqualified in `validation_results.json`; no source was tuned to their errors. The order16 result passes at6.22e-7. Both final baryon and lepton blocking are negligible for the dilute anchor: the maximum final-proton occupation is below7.2e-12 and the muon occupation is belowexp(-528). The anchor's target distribution itself remains actual FD, not a replacement classical density. All KMS log discrepancies are below2e-14, but are not used to excuse coarse quadrature failure.

The warm forward/inverse integrals separately have represented ordinary values. Every cold inverse integral has a finite retained log but is below double's ordinary representable range; it is never silently returned as a physical zero. No incident-energy-integrated number/energy moments or material evolution were performed. The API supports other channels, but this leaf only qualifies these two muon-neutrino capture points; antineutrino, electron, threshold and degenerate-state integration need their own gates. Current form-factor choices, absence of nuclear correlations/interactions, and absent radiative corrections are model limitations with no uncertainty estimate here.

`validation_results.json` contains54 passing selected report checks and retains the3 coarse cold failures separately. One controls batch was deliberately interrupted before its120s budget; `control_batch_budget_stop.json` and partial results preserve that event. The unfinished control was replaced by a separately budgeted run, without treating the interruption as a physical pass. Each completed individual integral took0.26–42.66s; the final cold20 integral took24.69s. Fresh defaultCLI order8 is a convenience, **not** a universal accuracy qualification: it fails the cold anchor here. Use recorded converged controls for reproduction.

Example reproduction using the campaign's base Python environment:

```bash
/Users/rossma/miniconda3/bin/python run_study.py --state warm --order 16 --angle-order 16 --inner-order 32 --tail 60 --outer-tail 40
/Users/rossma/miniconda3/bin/python run_study.py --state dilute --order 20 --angle-order 20 --inner-order 40 --tail 70 --outer-tail 48
/Users/rossma/miniconda3/bin/python run_study.py --state dilute --anchor --order 24 --angle-order 24 --inner-order 24
```

`CHECKPOINT.json` pins sources, upstream physics, raw results and environment versions. No external moving-target worker results were used to select quadrature nodes, tune the current, or alter a prefactor. That worker's independently frozen event oracle is a subsequent forward gate, not automatic acceptance of this leaf.
