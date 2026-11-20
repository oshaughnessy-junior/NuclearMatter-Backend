# Independent numerical review: selected GM1 thermal opening

**Accept the stated selected differential-response claim. 109/109 completed review checks pass.** This review owns only its new directory, changes no author source and verifies frozen source/evidence hashes unchanged. It introduces a held-out temperature/angle and independently computes forward density roots, original delta-shell integrals and exact cold Fermi geometry. It does not qualify a total Urca rate, a fitted width, a parameter rectangle or an external interacting EOS model.

## New point and forward roots before rates

The new point uses density.97 of the author's zero-neutrino triangle threshold, **T=.21MeV**, Eν=.3MeV, outgoing El equal to the corresponding cold beta mu_e, and **cos(theta)=−.67**. Neither this temperature nor this angle is in the author's main or earlier interior-angle probes. The cold beta fractions Ye,Ymu are preserved.

For both frozen cold fields and independently re-solved warm GM1 fields, I integrate baryon particle densities and lepton signed net densities directly in momentum, with explicit Fermi-surface panels. Generic momentum-root inversions recover each chemical potential before any response comparison. Maximum forward density error is1.16e−14 relative; independently solved total/kinetic species chemical potentials differ at most1.34e−11MeV. This verifies actual FD density matching rather than holding chemical potentials fixed or assuming Maxwell-Boltzmann populations. Nuclear field/root/local-stability qualification is inherited from the previously reviewed GM1 factory, not falsely repeated as an external EOS comparison.

The frozen closure is explicitly a prescribed cold Hamiltonian with thermal populations. The warm closure re-solves the Hartree fields at the **same fixed cold fractions**, not warm beta equilibrium. At this new point zeta_e=mu_e+mu_p−mu_n is−.00153606908975MeV for frozen fields and−.00145801720009MeV for warm fields. Those actual nonzero affinities must remain in detailed balance.

## Independent original delta-shell calculation

`probe.py` computes an independent transverse-momentum integral. At fixed p_perp it solves the **original longitudinal energy delta** with generic Brent roots, constructs explicit external baryon/lepton spinor currents, and integrates p_perp²/(2mD T) and azimuth. It does not call the author's q-axis energy integral, cold-support function, or held-out helper. Previously frozen independent gamma/spinor primitives are reused openly; this is a reused independent physical method at new inputs, not a claim of new external software validation.

The physical lepton four-vectors use omega=Eν−El. Scalar form factors are independently reconstructed at physical omega²−q², while the explicit WM/PS vertex uses kinetic q*=(omega−Up+Un,qvec) and **bare939MeV**, with mD in baryon spinors. At the new point physical q² is−173.80391145MeV² while q*² is−22828.61873344MeV²; confusing these would be a substantial convention change. No vacuum neutron-proton Q or fitted threshold shift is added.

The new integration's20→32 transverse order and60→100 transverse-energy tails agree within8.64e−14 relative. Its full complex forward/inverse W tensors and separately integrated j,chi agree with the frozen kernel at48 energy nodes/8 azimuths/tail90 within5.33e−13. Original energy-delta residuals are at most5.23e−12MeV. These are actual forward coefficients and tensors, not agreement inferred solely from pointwise KMS.

| closure | chi [s^-1 MeV^-1 per cosine] | j [same units] | independently cold signed gap [MeV] |
|---|---:|---:|---:|
| frozen cold fields |.026431297662593337|.006288124045177072|−3.1289770524012965|
| self-consistent warm fields |.026405527076896487|.006284328405388943|−3.129182337655493|

The independent transverse values are retained separately in results.json. Both ordinary values are positive and represented. No floor, inserted width or channel omission produces that positivity.

## Cold window at identical physical kinematics

For each Hamiltonian, the original delta root at p_perp=0 independently gives its minimal allowed initial kinetic total energy. Fill density-defined cold Fermi spheres using that same mD and vector shifts. The cold absorption conditions are **simultaneously** Ea<nu_nF and Eb=Ea+w>nu_pF, so the permitted interval is max(Emin,nu_pF−w)<Ea<nu_nF. Its signed gaps above are strictly negative. Thus the finite-T coefficient is positive where the corresponding cold occupation window at the **same finite Eν,El,angle and Hamiltonian** is empty. Replacing warm fields by their own cold-filled Fermi spheres still leaves the new window closed; the thermal opening is not explained by that instantaneous Hamiltonian becoming cold-open.

The author's elementary FD/Boltzmann bound is physically consistent: for Ea=Emin+x, the occupation ratio to the initial Boltzmann/no-final-blocking integrand is `1/[(1+za exp(−x/T))(1+zb exp(−x/T))]`. Positivity of the explicit full-current squared matrix element bounds its positive weighted integral between1/[(1+za)(1+zb)] and1. This review does not claim a new numerical reproduction of the author's analytic polynomial-integral values; the new original-delta forward calculation above is its independent numerical witness.

## Finite-E onset versus the zero-neutrino triangle

`check_onsets.py` independently locates both cold density onsets from **Fermi momentum geometry**, without the author's Emin formula or support-root routine. At the backscattered finite-Eν onset, the maximal cold neutron is at its Fermi energy; the outgoing proton has energy nu_pF+Eν. The opening condition is therefore

`pFn − sqrt[(nu_pF+Eν)²−mD²] − pFe − Eν = 0`.

At Eν→0 this reduces to the usual electron Fermi triangle. Re-solving the cold states and these separate density roots reproduces the finite-Eν.3 selected onset.2758839462424fm^-3 and zero-neutrino onset.2780845280468fm^-3 within3e−11fm^-3. Their distinction is physical kinematics, not a fitted thermal width. The finite-E root follows the author's declared family with El equal to each density's cold mu_e; it is not an onset for all spectral kinematics or an integrated rate.

All18 saved main rows are independently checked for labeling: only the.97-density cases are called thermal opening. At and above the zero-neutrino triangle, the chosen finite incident energy already gives a cold-open absorption window and cannot be labeled a thermal turn-on. All actual affinity and bare-mass metadata remain consistent. The cold electron surface is a differential boundary convention; it does not by itself establish an integrated T0 emissivity.

## Evidence, limitations and next work

The main new witness gives34PASS in.74s, and the independent geometry/metadata audit gives75PASS in.06s. Two reviewer-only setup/reporting failures are retained: a mismatched helper constant/spinor-layout assumption, and JSON serialization of complex arrays. Original drivers/partial records are preserved; the corrected import usage and encoder do not change physical kernels, tolerances or selected inputs. No physical numerical gate failed. All experiments are far below120s.

Accepted scope is selected **differential electron-neutrino capture and inverse** at prescribed cold fractions, with separate frozen-field versus re-solved-field closures. Small closure differences are not current/EOS uncertainty estimates. Warm beta equilibrium, complete crossed/antineutrino inventory, lepton/incident-energy integrals, a uniform broadening law, muon-sector thermal opening, gradients, material evolution, global phase selection and external finite-T GM1 agreement remain open. Internal agreement does not remove the phenomenological-current, no-pairing/no-correlation or rounded-coupling GM1 model limitations. No author repair is needed within the reviewed scope.
