# Hot neutron beta/inverse-three-body spectral emissivity and true absorption

**Frozen author prototype; DIFFERENT independent review required.** Actual separate positive j andchi are calculated for n↔p+e−+anti-nu_e at the same physical free-npe T=2MeV,nB=1e−7fm−3,Ye=.3 bath. The new conditional phase calculation covers selected energies below ANDabove the neutron-rest antineutrino endpoint, then reconstructs complete channel number/energy moments. **690 related assertions PASS**, including repeated node/scenario/metadata checks; this is not690 independent physical benchmarks. No new numerical gate failed. No PCsource or total is read/joined and no production/papers/git/dependencies changed.

## Existing qualification and what was missing

The frozen `relativistic_crossed_thermal/kernel.py` already supplies the full physical differential crossed kernel with independent occupations and exact three-body timelike support. Its OLD integrated API remains restricted to lowT .01–.1MeV and below-rest-endpoint incident energies. The newer `hot_beta_moments/hot_spectrum.py` is explicitly only the exact T2bath atEanti=.42MeV. It was independently accepted with a conditional-event comparison at that one energy; that did not establish all-energy spectrum or whole-channel spectral reconstruction. `hot_beta_moments/conditional.py` provided a useful alternate cone-energy reduction, with generalized-Laguerre parents and a linear rest-neutrino-energy window.

The present leaf adds a resolved original-parent-momentum rule, a different exact conditional delta reduction and explicit moving-parent endpoint handling. It keeps alloldfunctions/guards/records unchanged. In particular original `hot_beta_moments/results.json` remains exactSHA57e2e89e820b0e80b9d0cd4956d67c54e208cd78d5e2609edcd8e1ef54f94d88, checked before, after and at every new physical call. No genericdriver is imported; uniquely named pure-function modules are loaded with genericaliases/sys.path restored.

## Physical state, current and inverse

Fresh accepted physical FD roots give nn=7e−8,np=3e−8fm−3,total mu_n919.8287578100662,mu_p916.8449303084382,mu_e.17446333200826397MeV. Actual e−/e+ particle densities are2.03428636784328e−7/1.73428636784328e−7fm−3, with netcharge np. Electron blocking uses the actual electron occupation, not net charge; positrons enter equilibrium matter thermodynamics but PC is a distinct unjoined channel. Muons/photons and baryon antiparticle species are absent; the inherited antiparticle diagnostics remain in full state metadata. All energies and chemical potentials include rest mass.

The accepted source-labelled [Guo et al. full current](https://arxiv.org/html/2006.12051v2) retains V/A, weak magnetism, induced pseudoscalar and physical-q² form factors, with bare weak mass(mn+mp)/2 and physical free masses. Its inverse uses the literal reciprocal Dirac-adjoint baryon vertex andvbar_antinu gamma(1−gamma5)u_e lepton current. This is inverse3→1 absorption, not a PC mass/occupation substitution. No mean fields, extraQ, radiative/Coulomb corrections, fitted normalization or surrogate is added.

Direct independent occupation products are

    j: fn(En) [1−fp(Ep)] [1−fe(Ee)],
    chi: [1−fn(En)] fp(Ep) fe(Ee).

Both positive branches are integrated separately. The neutrino occupation is excluded from these coefficients: df/dt=j(1−f)−chi f. True absorption ischi; fixed-bath relaxation isj+chi. KMSlog(j/chi)=[mu_n−mu_p−mu_e−Eanti]/T is checked AFTER construction, never used to manufacture an inverse. Fresh explicit full-current decay/inverse spin sums at four newrest events pass reciprocity before the global tests.

## Exact conditional measure and spin/DOS

The common invariant phase is parametrized by a neutron at rest with spin-averaged dGamma*/(dEanti* dEe*)=amplitude_avg2/(64pi³mn hbar). Multiplying the original spin-two parent measurep²dp/[pi²(hbarc)³] andmn/En converts it into the same unaveraged event measure for either direction. For the inverse, this is simply the final-neutron invariant1/(2En) measure expressed in rest variables; it is not a physical inverse-decay proper-time shortcut. The spin-two measure cancels the neutron-spin average. No extra proton/electron degeneracy or anti helicity average is added.

For fixed lab anti energyE and parentgamma,beta,

    E = gamma E* (1+beta cos(theta*)),
    E* = E/[gamma(1+beta cos(theta*))].

Integrating the lab-energy delta overE* gives1/[gamma(1+beta cos(theta*))] times the isotropic angular measure dcos(theta*)/2. This differs from the old conditional reduction integrating that delta overcos(theta*), which givesdE*/(2gamma beta E*). They are related by an exact change of variables. The new form avoids dividing bybeta in the conditional Jacobian and uses the finite small-velocity limit.

The neutron-rest anti endpoint is**0.7820077025790402MeV**. At givenparentvelocity, allowedcos(theta*) is[max(−1,(E/(gamma E0)−1)/beta),1]. ForE>E0, exactminimum parentrapidity isln(E/E0), so pmin=mn sinh[ln(E/E0)]. This lower limit is resolved, not represented by discontinuously dropping generalized-Laguerre nodes. ForE<E0, the parent panel includes the exactswitchp=mn sinh[ln(E0/E)] where the cone starts touching the restendpoint. A sin² cone map resolves the electron–proton threshold square root.

At eachrestE*, the e−p pair has invariantS=mn²−2mnE*. Its exact two-body relative momentum is computed using stable endpoint factors, and its fullCM polar angle is integrated. dEe*/dcos(zeta)=E* p_rel/sqrtS supplies the exact Dalitz Jacobian. Daughter four-vectors are boosted into the bath before evaluating electron/proton FD factors. The remaining electron azimuth is actually integrated. All weak angular correlations remain in the fullspinor amplitude; no averagegap, stationary target or inserted physical width replaces this chain.

The conditional event densitydR/dE already integrates the physical neutrino phase space. Dividing by the single-active-helicity DOSE²/[2pi²(hbarc)³] yields per-occupation j andchi ins−1. Multiplying that DOS ONCE is required when reconstructing volume moments; no second spin or phase-space factor is inserted. Selected electron/proton shell andlab-energy residuals are retained in each raw result.

## Absolute anchors and spectra above the rest endpoint

Initial .42MeV conditional output reproduces the independently accepted old localj=.07083091288795236 andchi=.02144761484832465s−1. The newparent/cone/electron/phi/tail rules8/10/10/6/50T→12/16/16/10/70T pass atE=.2,.42,.5,.8MeV, with maximumchange4.23e−9 at3e−6. The old .42-only hot guard and old low-T guard are not extended or removed.

For an absolute **new above-endpoint** anchor, `tensor_reference.py` integrates the unchanged fullcomplex crossed kernel over actual lab electron energy/angle atE=.8MeV. It uses numerical electron-energy panels, not a neutron-rest endpoint cut. Rules6/8/12/phi4/width14/tail60→10/12/20/phi6/width20/tail80 refine7.13e−10; the final complex response agrees with the new conditional-event coefficient within2.09e-10 at unchanged3e−6. This is numerical/current-phase agreement within the same full tree model, not independent physical measurement. The primary conditional method has exact finite rest-Dalitz bounds; the comparator's electron truncation is a separately selected numerical control.

| LabEanti [MeV] | j [s−1] | truechi [s−1] | relaxation [s−1] |
|0.2|0.127067419498652|0.0344681504740978|0.16153556997275|
|0.42|0.0708309128879524|0.0214476148483246|0.0922785277362769|
|0.5|0.0540717910490092|0.0170411420806974|0.0711129331297066|
|0.8|0.00223445804132388|0.000818171445509981|0.00305262948683386|

PositiveE=.8 output is above the neutron-restendpoint and comes from actual moving-neutron support. Fullbeta transfer isq=kanti+pe, future timelike withme²<=q²<=(mn−mp)²; the inversep+e−+anti→n uses the appropriate positive energy support. Nothing is taken from the PCspace-like branch or given an artificial smoothing width.

## Forward controls, reconstruction and moving tail

Independent adaptive original-momentum FD integrals first reproduce the new conditional source's allowed-parent density. Below the restendpoint pmin=0, that is the fullEOS neutron density; above it the allowed-domain density is correctly smaller, and is not called an EOS-root error. Parent-tail70→90T, azimuth10→14 and a joint18parent/24cone/24electron/14phi/tail90 refinement pass at .8. Input/log guards, direct both-affinity entropy/Le checks and these controls52 PASS.

Incident panels[0,.2,.5,.7,E0,.9,1.05,1.2,1.4]MeV use six andten Gauss nodes perpanel. Coarse48nodes take8.40s. Fine80nodes are collected in two exact40node chunks taking22.59s and11.26s. The largest individual source call stays well below the120s budget. There is no spectral interpolation or rate cache; every node is a new actual density-root/phase calculation. Coarse/fine material controls also increase as declared before collection. Consequently global refinement is a combined incident/material check, complemented by the selected isolatedcontrols, not a bound for each separate error.

One-helicity reconstruction compares separately positiveforward/inverse number and energy against the exact accepted hot beta event totals for emptyincoming radiation,LTE,muanti=mu_eq−1.6MeV andmuanti=mu_eq+1.6MeV. All560 global assertions pass; fine-versus-event maximumdifference is**1.18e-09**, coarse→fine maximum**8.39e-08**, at unchanged3e−6. KMS ratios, zeroemptyinverse and nonzero LTE branches are checked independently of those moment comparisons. The event reference uses different globalrest/parent/orientation integration but the same declared physical current, so agreement is not a model-uncertainty estimate.

Fine empty totals are**R=4.258739504697108e-11fm−3s−1** and**Qanti=2.023774240076259e-11MeVfm−3s−1**. Above the neutron-rest endpoint, actual positive moving-parent emission contributes**R=4.40269072904e-13**, **Q=3.53819249636e-13** in those units, or**1.033801%** of the number and**1.748314%** of the energy. Cutting lab emission at the restendpoint would therefore miss a measured finite contribution in this controlled bath. This is thermal Doppler kinematics of free particles, not a nuclear collision width or interacting direct-Urca threshold prediction.

Spectrum beyond1.4MeV remains physically nonzero. No uniform analytic omitted-tail bound or numerical guarantee for every energy within the convenienceguard is claimed. Selected complete moments are supported by the independent event-total comparison and actualrefinement. Derivatives, extreme soft-log accuracy, highdegeneracy or otherT/nB/Ye states remain unqualified.

## Ledgers, evidence admission and limits

Per netdecay event, material(n,p,e_net) changes(−1,+1,+1) andanti number changes+1. MaterialLe+R andradiationLe−R cancel; baryon and electric charge conserve. Rest-inclusive matter energy loses emittedanti energy, with no extra beta-Q. For commonT FD radiation the directly computed(F−B)log(F/B) is positive in either tested affinity direction and agrees with the matter+radiation entropy decomposition. Empty radiation has no finite initialradiation entropy derivative claim. This leaf returns radiation coefficients/moments, not all material-energy spectra or an evolving EOS; the prior accepted event model supplies its separate material ledger.

`total.validate` requires finite positive-channel logs at every selected integrated node: falseempty support cannot replace a real positive beta coefficient. Ordinary underflow/overflow flags remain distinct from exactsupport. It verifies actual state/source/control/node identity and one-helicity channel units.52 actual-node andsynthetic-tamper checks pass, including replacing BOTHbranches byfalseempty records. Completednode files refuse overwrite and partialcollections are not admitted ascomplete.

The initial manual experiment-plan endpoint literal was mistyped; `initial_plan_endpoint_typo.json` preserves it. The physicaldriver ALWAYS used the exactS.ENDPOINT expression from beforecollection, as proven by raw contracts/nodes. Only the descriptiveplan literal was corrected; no physical source, quadrature node or gate changed. No failed numerical gate or expensive discarded run occurred in this leaf.

Original hot_beta_moments/results.json and everyother pinnedsource are unchanged. `INITIAL_UPSTREAM.json`, `CHECKPOINT.json` and `MANIFEST.json` retain code, fullraw, acceptedreference identities and controls. Allcode stays in the research workspace. No neutronbeta+PC/fullanti or fullflavoropacity inventory, transport/cooling evolution, interacting/paired EOS, radiative/Coulomb uncertainty or ADqualification is granted. A different independent reviewer must assess this new spectral chain and reconstruction before integration.
