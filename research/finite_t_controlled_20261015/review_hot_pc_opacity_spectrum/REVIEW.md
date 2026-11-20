# Independent review: hot positron-capture opacity spectrum

**Selected physical results accepted; assembly admission NOT accepted until the false-zero guard is repaired and independently rechecked.** Initial review is frozen. There are 121 passing checks and one reproducible admission failure. No author or upstream file was changed.

Reviewed `hot_pc_opacity_spectrum/spectrum.py` SHA256 `5fbc8fc21fc6b3b26bf43331ebe04b31bf66145b1f7bf51c977dfa47e5f78b0b`, manifest `1c60b6e64e31aecc32684a50c780182c176e136d05fea854d8791caf62eaf323`. The independently reviewed physical model remains free relativistic n,p plus pair-equilibrated electrons, T=2 MeV, nB=1e-7 fm^-3, Ye=.3, no muons. This review is independent of this leaf's author. It uses previously qualified current building blocks and an independent delta/spinor implementation; it does not establish external nuclear-model accuracy.

## Physics and normalization

The named reaction is n+e+ <-> p+antinue. For transfer q=k_antinu-l_positron and w=Eantinu-El, the response is ordered p -> n. Absorption uses fp(Ep)[1-fn(Ep+w)][1-fplus(El)], whereas emission independently integrates fn(Ep+w)[1-fp(Ep)]fplus(El). Here mu_plus=-mu_e and all baryon chemical potentials and energies include their physical rest masses. Therefore log(j/chi)=-(Eantinu-(mu_n-mu_p-mu_e))/T. This is a check after both coefficients are calculated, not a construction of emission from absorption.

The reciprocal full massive current and antilepton spin trace have the correct baryon ordering and crossing signs. The scalar amplitude used in the spectral code is averaged over the original neutron/positron spin states (four states); restoring the fully summed amplitude and using per-spin occupations gives the stated response coefficient pe/(32 pi^3 q Eantinu hbar) times the energy/azimuth integral of the averaged amplitude. Three new direct original-delta explicit-spinor comparisons pass with maximum relative discrepancy 3.68e-13. Shared prior form-factor/current conventions are model choices; numerical agreement does not measure their uncertainty. Primary convention: [Guo et al., arXiv:2006.12051](https://arxiv.org/html/2006.12051v2).

Physical PC lepton kinematics permit only spacelike transfer on this unequal-mass baryon shell. Future-timelike q cannot satisfy k=l+q with massless incoming antineutrino and massive outgoing positron; past-timelike q would require an initial proton at least as massive as the final neutron plus a future causal four-vector. This excludes timelike transfer here without altering genuine moving-target support. Two new past-timelike points give exact empty support. A new Eantinu=1.73 MeV point is positive and agrees with the independent spinor calculation: the stationary 1.806-MeV threshold must not be imposed on a warm moving-target bath.

The collision operator per one active antineutrino occupation is j(1-f)-chi*f. Chi is true absorption in s^-1; j+chi is relaxation, and dividing by c produces inverse length. Total number and energy use one-helicity DOS E^2/[2 pi^2 (hbar*c)^3], without a second spin factor. For net reaction R, material delta counts are neutron -1, proton +1, positron -1, net electron +1, antineutrino +1. Radiation energy gain is Eantinu*R, material loss its negative, without adding a separate mass-Q term. Material electron-lepton gain R cancels the antineutrino lepton loss. Common-T FD radiation gives total entropy R*(mu_eq-mu_rad)/T >=0. The empty-radiation initial entropy derivative is singular and is not assigned a finite entropy value.

## Fresh independent phase-space evidence

Command: `/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_hot_pc_opacity_spectrum/probe.py`.

`results.json`: 21 PASS, no failures. New differential points (Eantinu,El,cos)=(1.73,.574,-.31),(6.13,4.83,.47),(12.37,10.99,-.61). The independent calculation solves the original cylindrical pz delta using transverse kinetic variable u=r_perp^2/(2mp), rather than the author's baryon-energy support reduction. Direct forward and inverse occupancies are retained separately. The full-spinor oracle supplies an additional amplitude and normalization comparison.

At new incident energy 3.47 MeV, a different outer coordinate y=log(1+p_positron/T) and an exact spacelike forward-angle boundary replace the author's positron square-root coordinate and full cosine rule. Two independent outer/angle/inner/azimuth refinements (8/12/16/6 and12/18/24/8), tail60->85T and charged-lepton ceiling90->110 MeV, agree to1.54e-10. Fine logj=-1.5428339361719257 and logchi=-1.212516020981826, agreeing with a fresh author-function evaluation to4.98e-14. These two new integrals cost .81 and2.66s. They qualify this selected energy, not a broad domain or all soft asymptotics.

## Saved-total reconstruction

Command: `/Users/rossma/miniconda3/bin/python outputs/nmb-finite-t-20261015/review_hot_pc_opacity_spectrum/assembly_probe.py`.

`assembly_results.json`:100 PASS plus the one admission failure below. All manifest-listed bytes verify. All64 fine node indices and their metadata validate. A separate75-digit Decimal reconstruction retains finite log contributions including log~-13836, separately accumulates both directions, and reproduces the saved number/energy totals. This is independent arithmetic on SAVED physical nodes, not another fresh global spectrum integration. The saved reconstruction agrees with the independent actual CM event reference within the author's3e-6 gate (author maximum2.12e-8). Fine empty number=1.1173192771062813e-6 fm^-3 s^-1 and antineutrino energy=1.1994615318987417e-5 MeV fm^-3 s^-1. LTE net number/energy cancel at displayed precision; nonequilibrium mu_rad=mu_eq-.8MeV gives R=3.5140645223807835e-7 and Q=3.848695833777809e-6, positive entropy1.4056258089523134e-7 fm^-3 s^-1.

The unchanged10 coarse failures remain failures. Middle->fine selected refinement passes; this does not prove a uniform continuum or omitted-tail error bound. Very soft finite logs establish representability handling only; they are not relative-accuracy certificates. Original NumPy-bool reporter, syntax failure, quota hold/reset anomaly, interrupted partial and resumed node evidence remain preserved.

## Required narrow repair

`total.validate_row` accepts a synthetic positive-energy row whose independently physical integrated j and chi are BOTH replaced by `represented(None)` (exact zero with reason `empty_support`), with relaxation=0 and log_kms_actual=None. The full old and mutated rows are saved in `assembly_results.json`. This makes the consumer skip both coefficients and skip the KMS gate. At finite T in this bath, integrated PC coefficients have positive moving-target FD tails; exact empty support is unavailable. Ordinary output underflow must retain its finite log and cannot justify a log=None replacement. The current saved rows are not corrupted, but admission needs to distinguish integrated positive channels from legitimately empty differential support. No author repair has been made by this reviewer.

## Scope remaining open

Accept the selected physical spectral coefficients and their existing finite-domain total reconstruction, with all recorded coarse and soft qualifications. Do not admit the assembly consumer as guarded until repair review. No joined complete antineutrino inventory, new hydrodynamics, generic warm-domain certificate, GM1/RPA/medium physics, Coulomb/radiative corrections, uniform uncertainty, or derivative fidelity is established. The finite energy ceiling is a numerical truncation tested at selected settings, not an exact physical cutoff.
