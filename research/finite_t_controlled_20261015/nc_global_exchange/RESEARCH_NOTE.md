# Global double-energy NC exchange in a prescribed FD neutron bath

**Author-frozen research leaf; independent review required.** This computes global number, energy and entropy exchange for prescribed isotropic neutrino occupations and a fixed uniform free-NR neutron bath. It uses the reviewed LO neutral-current kernel and FD nuclear blocking. It does not evolve matter/radiation or solve hydro. Interactions, RPA/spin correlations, weak magnetism, relativistic-current corrections, nuclei and flavor conversion remain absent.

## Measures, physical sources and entropy

The one-helicity density of neutrino states is g(E)=E²/[2π²(ħc)³] [fm⁻³ MeV⁻¹]. The differential scattering kernel K(E→E',c) [s⁻¹ MeV⁻¹] already contains outgoing E'² phase space; no second outgoing DOS is multiplied into it. Define the actual forward and reverse event densities

    A = g(E) K(E→E',c) f(E)[1−f(E')],
    B = g(E') K(E'→E,c) f(E')[1−f(E)],
    J = A−B.

Both response kernels are evaluated directly in log form. Reverse gain is not replaced by an imposed KMS exponential. The measured relation K(reverse)/K(forward)=(E/E')² exp[−(E−E')/T] cancels the DOS ratio in the event ratio. Nuclear chemical potentials cancel because NC scattering does not change species. Final-neutrino Pauli blocking appears in both directions. Products are assembled before exponentiation; their difference uses a signed expm1 expression scaled by the larger log event density.

For unordered E<E' pairs integrated over dE dE' dc, global radiation energy change is (E'−E)J, bath heating Q=(E−E')J, and radiation entropy change is [h'(f')−h'(f)]J, where h'(f)=ln[(1−f)/f]. Entropy is measured in kB. Since ln(A/B)=h'(f')−h'(f)+(E−E')/T, the pair total entropy source is J ln(A/B)>=0. The prescribed fixed bath contributes Q/T; its density is unchanged. These have units fm⁻³ s⁻¹ for number and entropy/kB, and MeV fm⁻³ s⁻¹ for energy.

The paired number source is zero by construction and its positive entropy form is structural evidence only. The separate **full-square direct integral** instead evaluates C(E) using both actual kernels, then integrates g(E)C(E), E g(E)C(E), and h'(f(E))g(E)C(E). It uses outgoing-energy panels adapted to each incoming energy and direct-cosine angular quadrature, whereas paired events use the half-energy triangle and q angular quadrature. Number cancellation is not enforced in this direct integral. Heat and total entropy are compared between these independently ordered physical integrations. In the direct vector, radiation/bath energy conservation and the sum Srad+Q/T are algebraic bookkeeping identities; those identities alone are not claimed as independent validation.

For FD radiation, global number conservation gives Srad=−Q/Tnu and Stotal=Q(1/T−1/Tnu), at arbitrary neutrino chemical potential. NC scattering therefore admits LTE with arbitrary mu, rather than enforcing mu=0. The nonthermal case uses its actual occupation entropy and no FD temperature identity.

## Experiments, finite boxes and results

The bath is T=2 MeV,n_n=.002 fm⁻³, common kinetic mass939 MeV, spin2 and rest offset0. Occupations are:

- Hot FD: Tnu=3 MeV,munu=4 MeV.
- Cool FD: Tnu=1.5 MeV,munu=2 MeV.
- Packet: f=.15FD(E;3,4)+.3 exp[−((E−12)/4)²], strictly between0 and1.
- LTE FD: Tnu=2 MeV,munu=4 MeV.

The default experiment restricts **both** energies to0<E,E'<72 MeV. This is a closed collision box: its number source should vanish even though transitions to excluded energies are omitted. Physical energy-node Gauss rules are piecewise in incoming energy and resolve thermal/Fermi/recoil redistribution scales in outgoing energy. Independent controls compare energy12/inner24→20/40, angular48→96, and direct-square energy16/inner32/angular96. Coarse4/6/angular8 rules are tested against the unchanged3e−5 gate; failed candidates are retained.

Separate experiments actually extend the physical box to108 and144 MeV. Crossing integrals with one energy below72 and its partner in72–108 explicitly measure leakage for the **retained** low-energy population when the exterior radiation is prescribed. In `crossing` output the first component is this retained-box number source; its energy/entropy components count the complete crossing pair, including both radiation sides. That convention differs from the closed-box first component. A separate upper-upper72–108 integral completes the independent finite-box heat/entropy decomposition. A closed-box zero is not evidence that energy-boundary leakage is absent.

Preferred finite-box108 results, one neutrino helicity:

| Radiation | Bath heating Q [MeV fm⁻³ s⁻¹] | Radiation entropy [kB fm⁻³ s⁻¹] | Radiation+bath entropy [kB fm⁻³ s⁻¹] |
|---|---:|---:|---:|
| Hot FD | .0183617974776 | −.00612059915921 | .00306029957961 |
| Cool FD | −.000154326242531 | .000102884161687 | .0000257210404219 |
| Packet | .00858400563980 | .0157710281671 | .0200630309870 |
| LTE | ≈0 | ≈0 | ≈0 |

Hotter radiation heats the bath; cooler radiation extracts heat. The packet also heats this bath. LTE paired heat is~1e−19 and direct total entropy~−2e−18 in these units, consistent with roundoff when scaled by gross event rates, rather than a physical entropy decrease.

All **245 qualified core/control checks pass**. The20 coarse adequacy diagnostics contain12 passes and **8 preserved failures**: four packet heat/entropy errors4.67e−4–1.52e−3 and four positive gross-event-count errors3.60e−5–5.17e−5. No acceptance tolerance was weakened. Direct global number source is at most2.08e−15 of the positive event scale. Paired/direct positive gross event counts agree to3.21e−8 relative. Hot/packet direct108 heat and entropy agree with paired108 within4.39e−11 relative. The other states have direct72 comparisons plus independent physical-cutoff checks.

The hot72→108 heat correction is2.08e−5 relative; it is resolved rather than silently omitted. Extending108→144 changes hot/packet moments by at most1.36e−9 relative. Hot retained-box crossing number source is+3.72331763e−8 fm⁻³ s⁻¹, and its complete-pair heat is3.45976984e−7 MeV fm⁻³ s⁻¹. Full108 equals low-low72 plus low-high crossing plus high-high72–108 within4.80e−13 relative for heat/entropy. These are finite-cutoff numerical checks, not an infinite-tail bound or deep-tail relative accuracy certificate.

## Reproduction, provenance and scope

Run one bounded case per process:

    python study.py hot
    python study.py cold
    python study.py packet
    python study.py LTE

`qualify.py` validates every previously completed result's exact bath, radiation and current physical-source hashes before comparing it, and performs new direct108, paired144 and upper-upper integrations. It uses the immutable selected records in `initial_bounded_partial.json` for the first three cases and `results_LTE.json` for LTE. It does not reuse an evolving source or pretend an unfinished integral completed. `check_controls.py` exercises finite/domain controls. Every run record retains actual cutoff, all quadrature controls, full FD bath, prescribed radiation, panels, runtime and physical-source hashes.

The initial four-case orchestration was manually interrupted at114.42 seconds, before the120-second ceiling, after the first three cases completed and during an unfinished LTE continuation. Its checkpoint, initial driver and budget-stop record remain preserved. The driver was changed to one case per process; the physical integrator was unchanged. LTE was rerun completely in28.88 seconds. Independent extra qualification took40.36 seconds; individual completed physical calls are separately timed. The original rough profiling result and its pre-crossing source are retained as historical, excluded from accepted comparisons.

This is global exchange for selected **prescribed** spectra and a fixed free-gas bath. It does not supply radiation thermalization times, evolving matter, interacting opacities, long-time transport, or a parameter-domain certificate. Stronger degeneracy, lower temperatures, near-LTE relative accuracy at arbitrarily small perturbations, infinite-energy limits and a continuum radiation/matter evolution require separate qualification. The original continuum colder overflow exclusion remains; this warm leaf neither changes nor promotes that frozen API. Method agreement measures selected numerical integration error, not physical model uncertainty. No production or paper files were changed.
