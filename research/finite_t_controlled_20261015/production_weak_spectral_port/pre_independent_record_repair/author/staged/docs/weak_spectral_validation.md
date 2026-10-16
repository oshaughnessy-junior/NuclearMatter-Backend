# Selected free charged-current spectra: validation and open gates

This is a portable CPU reference for one explicitly prescribed chemical bath:
T=2, mu_n=919.8287578100662, mu_p=916.8449303084382,
mu_e=.17446333200826397 MeV. The API performs no EOS/neutrality/beta root.
It implements three named electron channels and their direct inverse material
occupation products. It is not the full weak inventory or a transport engine.

The numerical protocol was locked before any candidate rate calculation.
The base controls are lepton8, angle12, baryon-energy12, phi4, inner tail50T,
outer tail30T; decay uses parent8, cone10, electron10, phi6.
The fine profile is lepton12, angle16, baryon-energy16, phi6, inner70T,
outer40T; decay parent12, cone16, electron16, phi10. Bounds on orders and
energies restrict requests and resource use; they do not certify an arbitrary
point or every allowed control profile. Tests should use actual refinements.
No fit, calibration, hidden source table, cached moment or stationary-target
substitute enters any runtime coefficient.

Ten source-labelled INTERNAL saved spectra are retained in the test fixture:
EC/PC at2,5,10MeV; neutron decay at.2,.42,.5,.8MeV. These agree with the
ported direct branches within1.92e-11 relative in the selected fine rules;
base/fine changes are at most4.23e-9 for these rows, against a predeclared
3e-6 numerical gate. They are shared-Hamiltonian numerical checks, not
external/empirical uncertainty measurements. NEW predeclared energies are
EC3.7/7.3, PC3.3/8.2 and beta.31/.63/.83. Detailed author evidence records
separate direct j/chi, controls, physical support/bounds and actual runtime.

Fresh original longitudinal-delta phase integration with literal external
spin sums at EC(E,El,c)=(4.3,5.59,.27) and PC(1.73,.62,-.24) agrees with
the q-axis coefficients to1.08e-13, with independent transverse-momentum
refinement. A separate timelike EC point(4.3,5.59,.9999) tests literal
current agreement. The PC1.7MeV moving-target point remains positive below
the stationary proton1.806066208MeV threshold. Beta.8/.83MeV remains positive
above its.782007703MeV neutron-rest endpoint. No physical collision width,
threshold smoothing or imposed lab endpoint is involved.

The exact support polynomial retains spacelike, timelike and exact linear
solutions followed by positive-energy restrictions. Singular q=0,
degenerate roots, collapsed physical/mapped panels and unrepresentable
positive nodes raise unresolved rather than returning zero. There is no
panel omission based on a floating width, log-underflow or collapsed map.
Optional omission would require new rigorous complete weighted BOTH-branch
bounds and independent qualification; it is not implemented here.

Selected isolated inner tail, outer tail, phi and individual orders at
EC7.3MeV and beta.83MeV are recorded separately from the combined profiles.
The physical branch products are integrated independently before their KMS
ratio is tested. Agreement of KMS is necessary and cannot prove absolute
phase/spin normalization, a missing tail or convergence.

The reconstruction entrypoint returns an explicitly finite-range PILOT,
with all actual nodes, positive weights, exact bath/radiation/controls and
completed status. Whole-moment qualification is always false. The author
pilots cover2–6MeV for EC/PC and.2–.8MeV for beta, with2/4incident Gauss
nodes and common empty/LTE/changed-affinity radiation. Lower and upper
omitted ranges are nonzero. Whole R/Q reconstruction, isolated incident
energy tails and continuum error remain OPEN. Finite-bin E/Le identities
are algebraic bookkeeping rather than an independent species-energy or
continuum normalization check. Stage A event totals remain the independently
qualified source of total number/species/radiation-energy moments.

The unchanged Stage A40current/moment tests and13cold/unit regressions are
separate gates. Stage A's initial nonzero-signed/no-log guard failure and
subsequent independent repair remain historical evidence. Original research
coarse/cold/reconstruction/underflow and guard failures are not relabelled
passing by this port. An initial author support TEST fixture incorrectly
assumed PC E5,El.6,c1 was forbidden; its positive spacelike thermal tail is
physical. The failed assertion/source is preserved; the empty fixture was
corrected to the analytically forbidden PC past-timelike point E2,El3,c1.
No physical source equation or tolerance changed.

Different numerical/physics review and real wheel packaging are separate
requirements. Minimum Python3.9/NumPy1.21/SciPy1.7 runtime and normal full CI
are not executed in this author environment. Current NumPy2.3.5 lies outside
the separate campaign validation pin<2.3. No AD, EOS-family, mean-field,
correlation/pairing, radiative/Coulomb, model-uncertainty, arbitrary-bath,
angular redistribution or transport-opacity certificate is claimed.

The2→4incident-node pilots change forward R/Q by up to0.34% for EC,
1.71% for PC and5.29% for beta in their finite ranges. These are unresolved
incident quadrature sensitivities, not3e-6-converged whole moments. Small
E/Le/entropy ledger residuals do not turn these pilots into accurate totals.
The completed final author test run has86passing tests: unchanged40Stage A,
13cold/unit, and33new spectral/reconstruction tests. Separate selected
numerical/saved-arithmetic assessment has250passing checks. These counts
are overlapping verification operations, not numbers of independent models.
Pre-handoff malformed-bath ordering and incomplete/forged reconstruction-record
constructor failures are preserved separately; their guard repairs do not
change any physical equation or valid selected payload.
