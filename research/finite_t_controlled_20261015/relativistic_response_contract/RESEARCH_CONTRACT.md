# Relativistic finite-T charged-current response: implementation contract

15 October2026. Research derivation plus small normalization/kinematic probes; **no finite-T tensor or rate kernel implemented or qualified here**. The free-gas full-current vacuum anchor is a separate leaf undergoing review. This contract specifies the next integration with its own independent forward gate. Frozen upstream code and other contributors' work are preserved.

## Source anchors and convention hierarchy

[Guo et al.2020](https://arxiv.org/html/2006.12051v2), Eqs23–30, supplies mean-field dispersions, the full V/A/WM/PS vertex and invariant phase measure. Its AppendixB offers component invariants; AppendixC offers kinematic bounds. [Roberts & Reddy2017](https://arxiv.org/html/1612.02764v2), Eqs3–14, supplies the spin-density/phase-space normalization and tensor/opacity relation. Its discussion around Eq81 warns that unequal masses/self-energies invalidate a generic charged-current vector-conservation reduction. We derive our equations below with one explicit ordered transition rather than mixing momentum-transfer signs or response normalization definitions between papers. Source couplings and thermal models are not silently equated.

Use natural MeV units inside the tensor: hbar=c=kB=1, metric+---. Restore hbar, hbarc and c only at named output boundaries. The weak vertex mass normalization `Mweak` is explicit and fixed; it is not automatically the Dirac or Landau effective mass. Start with physical free baryon masses and the reviewed vacuum current convention. A common-mass reference is a distinct matched limit.

## Matter and species

Inputs `(nB,T,Ye,Ymu)` have independent net lepton fractions. For this nucleonic species set only, `Yp=Ye+Ymu`, `nn=(1−Yp)nB`, `np=Yp nB`. Each lepton has an explicit immutable mass and its own pair-equilibrated gas, with chemical potentials `(mu_l,−mu_l)` for negative/positive charges. Retain e± and mu±; never set Ye=Yp in the new state factory. The existing electron-neutral `matter_state` factory cannot supply this extension unchanged.

For the initial free baryon model, `Ei(p)=sqrt(p²+mi²)` and total `mu_i` includes mi. FD occupations are per spin: `fi=[exp((Ei−mu_i)/T)+1]^-1`; `ni=2∫d³p fi/(2pi hbarc)³`. Initial baryon-antibaryon reactions are excluded. Select T/density states where antinucleon populations are negligible and report that diagnostic. Charged-lepton antiparticles remain present. Particle-hole tensors do not include all timelike particle-antiparticle cuts.

Return species roots, all masses/potentials, total and kinetic chemical potentials, source energy zero, baryon particle versus net densities, thermodynamic energy/pressure/entropy, Jacobian and validity diagnostics. Use a common constant energy subtraction per conserved baryon only if declared; preserve physical mass differences in both dispersion and EOS. For trapped equilibrium, `zeta_l=mu_l+mu_p−mu_n=mu_nu_l`; antineutrino chemical potentials are−zeta_l when pair-equilibrated. `YLe` and `YLmu` include their separate neutrino-minus-antineutrino populations. Prescribed compositions need not satisfy any weak equilibrium.

## Invariant rate and spin normalization

Fix the **ordered absorption pair** `a→b`: `nu_l(k)+a(pa)→l−(l)+b(pb)`, with `q=k−l=pb−pa`, `omega=Eν−El`, `pb=pa+q`. Define

`dPi_i=d³pi/[(2pi)³ 2Ei]` (baryon normalization uses Ei* in a mean field),
`M=GF Vud/sqrt(2) [ubar_l gamma_mu(1−gamma5)u_nu] [ubar_b Gamma^mu u_a]`.

Let `Mavg²` average **initial baryon spin only**, with factor1/2, and sum final charged spins. No initial-neutrino average is taken: its one inactive spin state is annihilated by V−A. The positive absorption coefficient in natural energy units is

`chi(Eν)=2/(2Eν) ∫dPi_a dPi_l dPi_b (2pi)^4 delta4(k+pa−l−pb) Mavg² fa(1−fb)(1−fl)`.

The leading2 counts target spin because fa is per spin. Equivalently use a full baryon spin trace, without that explicit2 and without target averaging. **Do not use both.** The inverse-emission j uses the same fully spin-summed matrix element and measure, with occupation product `fb fl(1−fa)` instead. Do not insert separate target/lepton degeneracy factors into that inverse product; the common full spin sum already counts them.

The neutrino occupation equation is `dfν/dt=j(1−fν)−chi fν`, where `j_s=j/hbar`, `chi_s=chi/hbar`, `hbar=6.582119569e−22MeVs`. True opacity is `chi_s/c` in cm^-1; stimulated/relaxation opacity is `(j_s+chi_s)/c`. Neither quantity should be substituted for the other during comparisons.

## Tensor definition with closed prefactor

Use full, unaveraged baryon spin trace

`T_ab^(mu nu)=Tr[(slash pb*+mb*) Gamma^mu (slash pa*+ma*) barGamma^nu]`.

Define the absorption response

`W_ab^(mu nu)(omega,q)=∫d³pa/(2pi)³ [(2pi)delta(omega+Ea−Eb)/(4Ea*Eb*)] T_ab^(mu nu) fa(1−fb)`.

W has natural units MeV². Its definition includes the spin-two target sum through T, but **no GF², Vud² or lepton tensor**. With `L_mu nu` the unaveraged V−A lepton trace, integration of the charged-lepton azimuth gives

`dchi/(dEl dc)=GF²Vud²/(32pi²) * pl/Eν * (1−fl) L_mu nu W_ab^(mu nu)`,

where `c=cos(k,l)`, `pl=sqrt(El²−ml²)`, `El>=ml`, `q=sqrt(Eν²+pl²−2Eν pl c)`. This prefactor follows by integrating the baryon momentum delta and using `d³pl/[(2pi)³2El]→pl dEl dc/(8pi²)` after its azimuth. GF² has MeV^-4, L and W each MeV², so dchi/dEl is dimensionless and chi is MeV.

An equivalent `(El,q)` outer integral has q between `|Eν−pl|` and `Eν+pl`; `pl dc=−q dq/Eν`. The positive measure is therefore `GF²Vud² q/(32pi² Eν²) dEl dq`, times lepton blocking and the tensor contraction. No massless-lepton assumption `|omega|<q` is allowed.

## q-axis delta reduction, support and azimuth

Choose q along z. Let `w=omega−(Ub−Ua)` for constant local vector potentials; free gas has U=0. Set `Ea*=sqrt(p²+ma*²)`, `Eb*=Ea*+w`. The energy delta fixes

`x0=cos(pa,q)=[ma*²−mb*²+w²−q²+2w Ea*]/(2pq)`.

Required support is `Ea*>=ma*`, `Eb*>=mb*`, `|x0|<=1`. The derivative of Eb* with respect to x is `pq/Eb*`; hence delta integration contributes **Eb*/(pq)**. Combining this with the invariant measures and `p dp/Ea*=dEa*` gives

`W_ab^(mu nu)=1/(16pi² q) ∫allowed dEa* ∫0^(2pi) dphi T_ab^(mu nu) fa(1−fb)`.

Equivalently, the azimuth-averaged trace has prefactor `1/(8pi q)`. A missing2pi azimuth or use of a target-averaged T would change the rate. For `p=0` or `q=0`, this coordinate is singular; use a separately derived limit or reject/avoid a measure-zero endpoint, rather than adding a physical floor.

For an implementable support solver define `C=ma*²−mb*²+w²−q²`. The angle condition is the polynomial inequality

`(q²−w²) Ea*²−w C Ea*−(C²/4+q² ma*²)>=0`.

Intersect its intervals with the two positive-energy constraints. Spacelike kinetic transfer has one lower threshold; timelike transfer can give a finite interval or no particle-hole support. Handle the linear limit `q²=w²` separately and preserve unresolved floating-point boundaries. A root scan that assumes only a lower threshold does not establish all massive-lepton support. Validate reconstructed angles and the original energy delta at all accepted endpoints.

For direct tensor quadrature, `pa*=(Ea*,p sqrt(1−x0²)cosphi,p sqrt(1−x0²)sinphi,p x0)` and `pb*=(Eb*,pa_vec+q_z)`. Rotate k,l into the same q frame: `kz=(Eν²+q²−pl²)/(2q)`, `kx=sqrt(Eν²−kz²)`, `l_vec=(kx,0,kz−q)`. Check k,l mass shells rather than clipping a materially invalid radicand. All form factors depend on the specified invariant transfer and are constant through this baryon azimuth integral. The spin trace is at most quadratic in the sine/cosine of phi; an explicit periodic8→16-point comparison is a convenient numerical gate before using an analytically averaged tensor. Keep complex off-diagonal/antisymmetric components and contract covariantly; do not retain only scalar/spin diagonals for the full current.

Use log occupations, explicit Fermi-surface/support panels and tail refinements. The positive `fa(1−fb)` representation avoids dividing an FD difference by a near-zero detailed-balance denominator. Approximation of the integration tail and model validity are separate errors.

## Inverse channels and KMS

With the ordered definitions above, swapping the baryon pair and reversing transfer gives the tensor relation

`W_ba^(nu mu)(−omega,−q)=exp[−(omega+mu_a−mu_b)/T] W_ab^(mu nu)(omega,q)`.

Combine this with `fl/(1−fl)=exp[−(El−mu_l)/T]` to obtain independently constructed coefficients obeying

`j/chi=exp[−(Eν−zeta_l)/T]`, `zeta_l=mu_l+mu_b−mu_a`.

For antineutrino absorption `antinu_l+p→l++n`, use a=p,b=n, the antineutrino lepton trace, and effective positive-lepton chemical potential−mu_l. Its zeta is−(mu_l+mu_p−mu_n), giving the expected opposite radiation chemical potential. Test separate forward/reverse integrals before using KMS as a shortcut; it is not transparent kinetic balance.

Crossed neutron decay and its three-body inverse require their own crossed lepton trace and `q=k+l`, `omega=Eν+El` for the inverse transfer into baryons. They are not obtained by feeding positive El absorption into an unchanged lepton tensor. The conditionally supported crossed proton branch also requires an inventory decision. Pure muon decay and crossed flavor-exchange processes couple e/mu families and remain separate kernels from this one-baryon response. No total finite-T muon source can be claimed from capture alone.

## GM1 Hartree mapping and unresolved model choices

The archived NMB convention uses common bare mass939MeV, common Dirac mass `mD=939−Phi`, and local potentials `Un=W−R/2`, `Up=W+R/2`. Its kinetic chemical potentials `nu_i=mu_i−Ui` **include mD**, unlike the rest-subtracted kinetic chemical potential in the older NR reference. Convert the archived fm-unit fields/masses to MeV before evaluating this tensor.

Use `Ei=Ei*+Ui` in occupations and energy conservation; use `pi*=(Ei*,pi)` and mD in spinors and `1/(2Ei*)` state normalization. Physical transfer is `q=(omega,qvec)` and kinetic transfer is `q*=(omega−Up+Un,qvec)` for n→p. Reversing the ordered pair reverses the potential difference. Retain total chemical potentials in KMS.

The selected Guo prescription uses physical q² in the form-factor functions and q* in the WM/PS momentum vertices. In vacuum they agree. This is an **explicit in-medium current prescription requiring independent checking**, not a derivation of a unique conserving interacting response. Keep `Mweak` fixed as the chosen bare normalization in F2/(2Mweak) and GP/Mweak; do not replace it by mD. The initial physical free-gas benchmark uses `(mn+mp)/2`. A GM1 common-bare-mass matched reference must declare whether Mweak=939 or that physical average and use the same choice in its vacuum comparison.

Do not insert physical Q into rates while keeping GM1's common bare-mass EOS unchanged. Its zero-density matched limit is a common-mass vacuum channel, not automatically the physical unequal-mass vacuum benchmark. Introducing a physical mass splitting requires a new consistent EOS/field solve or an explicitly distinct approximation.

No generic constraint `q_mu W_V^(mu nu)=0` or `q*_mu W_V^(mu nu)=0` may eliminate components for unequal masses/self-energies without a derivation. The transverse WM vertex alone does not establish such a relation for the full charged current. Keep all tensor components and test the Ward/contraction identities of the exact chosen model.

Bulk material energy must come from the Hartree thermodynamic functional, including its scalar/vector field-energy terms and avoiding double counting. Summing `Ei*+Ui` over particles and adding the unchanged full vector field energy is not that functional. Source energy transfer and EOS composition/temperature inversion must share rest/field conventions; no second manually added potential/Q heating term belongs on top. Finite-T field roots, thermodynamic derivatives, phase selection and the still-open matched external GM1 forward gate remain separate prerequisites.

## Numerical witnesses and implementation gates

`probes.py` runs **30/30 checks in4.01s**. It tests independently constructed q-axis angles, original delta Jacobians and support polynomials; occupation-factor KMS with/without vector shifts; the relativistic density spin trace approaching spin2; and the algebraic reduction to the frozen NR optical prefactor. It also evaluates a dilute Maxwell-Juttner moving-target normalization witness using the separate full vacuum current:

`chi_s = n_target[cm^-3] c <v_Moller sigma(s)>`,
`v_Moller=1−pa cos(theta_ka)/Ea`, `Erest=Eν(Ea−pa cos(theta_ka))/ma`.

Here the target distribution has the exact classical relativistic Boltzmann shape `exp[−(Ea−ma)/T]`, generated with kinetic-energy Gauss-Laguerre quadrature; density1e−18fm^-3 keeps fugacity below1e−8. Muon Eν150MeV with T=.1→.01→.001MeV approaches the stationary-target cross section for both neutrino signs. At T=.001, relative thermal corrections are9.00834e−6 and5.05281e−6; they are genuine target-motion differences, not integration errors. Refining radial/angular/cross-section quadratures10/10/12→16/16/16 at T=.1 changes the two means by at most1.27e−14 relative. This **does not implement or independently validate the response integration above**: it is a required future reference value for that kernel, with the same vacuum current under review. Raw rates, fugacity and fractional limits are retained.

The NR matching calculation is especially useful for the normalization: for common large M, the full density trace satisfies `T00/(4Ea Eb)→2`, so W00 becomes the spin-two scalar response. Its spatial axial counterparts give the same scalar times ga². Contracting L gives the allowed angular factor, and the rate prefactor becomes

`K=GF²Vud²(hbarc)³/(4pi² hbar)`,

multiplying `∫dEl El pl ∫dc angular_factor S_NR`, exactly the existing `allowed_capture.K`. To compare the complete NR response, choose common masses and the same constant offsets; the physical unequal-mass free gas is only approximately that model after an NR expansion.

The new API should separate `QuasiparticleSpec(mDirac,U,mu_total)` from `WeakCurrentSpec(Mweak,form_factors,qstar_prescription)` and `LeptonSpec(mass,chemical_potential,flavor,charge)`. Return all16 complex W components, ordered-pair/transfer conventions, physical and kinetic q, quadrature/error/representability metadata and validity diagnostics. No GF is included in W. Return absorption and inverse-emission coefficients with their explicit unit boundary and no hidden response rescaling.

In particular, the current vacuum `trace` function derives its weak normalization mass from the same channel masses used for spinors. **It cannot be called unchanged with GM1 effective masses.** Its explicit `vertices(q,M,ff)` building block can be reused by a new contraction with qstar, the independent bare Mweak and form factors evaluated at physical q². Then assemble the new trace using kinetic spinor momenta and mDirac. This avoids a silent mD replacement in F2/GP while retaining source-defined vertices.

Next deliverable: explicit free tensor implementation with exact support intervals, separately integrated j,chi and all component/current switches. Its release gates are (1) direct delta/full-phase-space versus q-axis reduction, (2) dilute moving-target/cσ normalization, (3) independently matched NR response/current limits, (4) KMS, CP/crossing and tensor positivity, (5) energy/number moments with independent muon/electron compositions, and (6) only then GM1 mapping under a declared current ansatz. No warm result, opacity prefactor calibration or production release is inferred from this contract.
