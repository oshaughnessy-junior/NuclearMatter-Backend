# DESIGN: multi-branch (twin-star) EOS sequences in RIFT

*Written 2026-11-08 against backend `fix/sequence-hc-grid` (8c20741) and RIFT
`rift_O4d_junior_calmarg_in_loop_nm_backend` (9c894020).*

**Status, 2026-11-09: implemented as option (a) with w_b = 1.** Merged in oshaughn
`rift_O4c` (PRs oshaughn#261, oshaughn#262) and oshaughnessy-junior `rift_O4d`
(oshaughnessy-junior#536, #537, #539, #541). CIP takes `--eos-affine-parameter
{mass,central}`; `mass` is unchanged. `central` adds branch coordinates
`eos_branch_u1/u2`, adds ln n1 + ln n2 to the integrand, writes the branch labels to
`<samples>_eos_branch.dat`, and refuses `--fixed-parameter` on mass or tidal
coordinates. Open questions 1 and 7 are settled. The "Current RIFT behaviour" and
"Proposed sequence" sections record the June branch and are kept for history.

## Recommendation

The affine parameter along the sequence becomes a run-level choice.

- `affine="mass"` (default) is the v1 contract. Masses are the sampled coordinates
  and only the primary branch exists. Existing runs, files and `lambda_of_m_indx`
  callers are unchanged.
- `affine="central"` uses a central coordinate (NMB stores `h_c`; lalsuite uses
  `p_c`). Every stable branch is included, and each star is a single-valued
  function of the central coordinate.

Inside CIP, `affine="central"` should be implemented as masses plus one discrete
branch label per component. The mass prior then carries the Jacobian below. CIP
keeps its `mc`, `delta_mc` coordinates and the trained fit. Raw `h_c` sampling is
option (b) below. It defines the same model once the prior is fixed, and it samples
worse.

**Decided 2026-11-08 (Richard):** the prior is dM per stable branch (w_b = 1 below).
A mass with two stable stars counts twice. This is the usual one-branch prior
applied to each branch.

**Correction, 2026-11-08:** the RIFT findings below were read from the June branch
`rift_O4d_junior_calmarg_in_loop_nm_backend`. oshaughn `rift_O4c` (259baea8d)
already ports `_stable_branches` to `EOSSequenceNMB` (defect 1). It also supports
lalsimulation 7.0.0 families through `LALSimNeutronStarFamilyAdapter`
(`branches_for_m`, `for_branch`, `--using-eos-branch`; PR oshaughn#184). Its
`docs/eos-interface-contract.md` rejects `--using-eos-branch` for `nmbseq:` until a
central-coordinate path exists. Defect 2 does not apply to `rift_O4c`:
`convert_waveform_coordinates_with_eos` sets Lambda = -inf for `m >= mMaxMsun`, so
those samples are rejected (zero prior above M_max). Under the primary contract a twin
draw with M_max = 1.03 therefore loses most of the BNS mass range.

## What lalsuite provides

The multi-branch NS family is in lalsimulation 7.0.0 (tagged 2026-10-31). It is not
in 6.2.1. Source: `lalsimulation/lib/LALSimNeutronStarFamily.c` (Davis, Oertel,
Suleiman, Wade; commits through 2026-06).

| Function | What it returns |
|---|---|
| `XLALCreateSimNeutronStarFamily(eos, min_fam)` | 100 log-spaced `p_c` solves from `exp(75.5)` Pa to the EOS maximum pressure. A branch is a maximal run with `M[j] > M[j-1]`. Branch ends are refined by a GSL minimizer. The `min_fam` argument is new and required. |
| `...FamNumberOfBranches(fam)` | Number of stable branches, ordered by `p_c` |
| `...Fam{Min,Max}MassPerBranch`, `...Fam{Min,Max}CentralPressurePerBranch` | Branch limits |
| `...Fam{Radius,CentralPressure,BaryonicMass,LoveNumberK2,K3,K4}OfMassPerBranch(m, fam, b)` | Steffen interpolation on branch `b`. XLAL_EDOM error off the branch. No interpolators if the branch has 5 points or fewer. |
| `...Fam*OfMass(m, fam)` | `REAL8Vector` with one entry per branch that contains `m`. Length 0 if none. |
| `...FamMassOfCentralPressure(p, fam)` | Mass on the branch containing `p`. Error in an unstable gap. |
| `...FamMaxMass`, `...FamMinMass` | Max and min over all branches |
| `...FamMassTOVLimit` | Max over branches, or 0 if the last branch does not turn over |
| Deprecated `SimNeutronStarRadius`, `LoveNumberK2`, `CentralPressure` | Scalar. Raise XLAL_EFUNC when `m` has twins. |

lalsuite returns `k2`, not Lambda. Lambda = (2/3) k2 (R c^2 / G M)^5. NMB's
`branches(e)` and lalsuite's `branch_id` both count branches in central-coordinate
order. The grids differ: NMB starts at the minimum-mass star and drops runs shorter
than 2 rows, while lalsuite starts at `p_c = exp(75.5)` Pa. Check that the branch
counts agree before mapping indices.

lalsuite gives structure only. It has no prior or inference over branches.

RIFT's `EOSManager` calls the one-argument constructor (10 calls) and the deprecated
scalars (`SimNeutronStarRadius` 4, `LoveNumberK2` 2, `MaximumMass` 10). We read
the headers and did not run it. Under 7.0.0 the constructor calls should fail with
an argument error, and the scalars error at twin masses.

## The twin draws in the rebuilt archive

`paper1_gp/analyses/prior_archive/build/draws.nss.h5` (backend 8c20741), counted
with NMB's `_stable_branches` rule:

| Quantity | Value (median, 5-95%) |
|---|---|
| Draws with 2 / 3 stable branches | 192 / 3 of 1000 |
| Primary-branch M_max, twins | 1.03 (0.57-1.46) M_sun |
| All-branch M_max, twins | 1.40 (1.23-1.71) M_sun |
| Twins with all-branch M_max >= 2.0 M_sun | 0 of 195 |
| Single-branch M_max | 2.17 (1.51-2.81); 512 of 805 >= 2.0 |
| Later branch heavier than the primary | 168 of 195 |
| Twin mass interval (both branches exist) | 0.06 (0.00-0.27) M_sun |
| Lambda ratio, primary/later, at interval midpoint | 8.3 (1.9-71) |
| Twins with mass dip under 2% | 40, interval about 0.005 M_sun |

The archive README reports 190 twins. The script used here counts 195.

For this archive, most BNS masses (1.1-1.6 M_sun) lie above the primary branch and
on the later branch.

## Current RIFT behaviour

1. **RIFT's reader predates ecdd270.** `EOSSequenceNMB._stable_rising` and
   `EOSSequencePCA` keep every row up to the global stable maximum, unstable rows
   included, then sort by mass. For 169 of 195 twin draws the resulting M to Lambda
   table mixes branches. RIFT therefore does not implement the primary-branch
   contract that `docs/rift.rst` describes. The fix is to port `_stable_branches`.
2. **Masses above m_max get Lambda = 0.** `convert_waveform_coordinates_with_eos`
   zero-fills Lambda for `m >= mMaxMsun`, and `EOSSequenceSingleIndex` returns 1e-8
   above 0.999 `m_max`. Under the primary contract, a twin draw with M_max = 1.03
   makes both GW170817 components zero-tide objects. The GW170817 likelihood favours
   low Lambda, so these draws can gain evidence. The same applies to any
   low-M_max draw.
3. The inherited `lambda_of_m_indx` uses `np.interp` without `left`/`right`, so off
   the branch it returns the end value. The ordering statistic Lambda(m_ref) uses
   m_ref = 1.37 M_sun, which is above the primary M_max for at least 142 twin draws.

## Options compared

Z_e is the per-EOS evidence, pi(m1, m2) is CIP's EOS-independent mass prior (uniform
in component masses inside the downselect box), and B_e(m) is the set of branches of
EOS e that contain m, with n_e(m) = |B_e(m)|.

### Mass affine (v1, default)

- **CIP change.** None beyond defects 1 and 2. `m_max_of_indx` is the primary M_max.
- **Mass prior.** pi(m1, m2), with Lambda set by the m > m_max convention.
- **Evidence for twins.** Only the primary branch counts. For this archive that
  makes most BNS masses fall above m_max.

### (a) Mass coordinates, marginalized over branches

Z_e = ∫ dm1 dm2 pi(m1, m2) sum over b1 in B_e(m1), b2 in B_e(m2) of
w_b1(m1) w_b2(m2) L(m1, m2, Lambda_b1(m1), Lambda_b2(m2)).

- **CIP change.**
  - `EOSSequenceFromFile(..., affine=)` is passed to the reader. The reader
    stores per-branch (M, R, Lambda, h_c) arrays from NMB `branches(e)`.
  - `lambda_of_m_indx(m, indx, branch=0)` keeps its current return for
    `branch=0`. A new `branches_at_m_indx(m, indx)` returns B_e(m), as lalsuite's
    vector accessors do.
  - `m_max_of_indx(indx, branch="primary"|"all")`.
  - Sampling adds coordinates `u1, u2` drawn uniform on [0, 1). Then
    b_i = floor(u_i n_e(m_i)) picks a branch, and
    `convert_waveform_coordinates_with_eos` looks up Lambda on b_i. This is a Monte
    Carlo sum over branches that leaves the integrand and the fit unchanged.
  - ILE needs no change in the fixed-EOS path, because the fit is over
    Lambda-tilde and any training Lambda values are valid.
- **Mass prior.** With w_b = 1/n_e(m), p(m | e) = pi(m), the same as v1. The
  alternative is w_b = 1, which makes each stable star count once per unit mass.
  Then p(m | e) is proportional to n_e(m) pi(m), CIP adds the log-weight
  ln n_e(m1) + ln n_e(m2), and the twin interval holds extra prior mass.
- **Evidence for twins.** Between primary M_max and later-branch M_max (1.03-1.40
  M_sun at the median), zero-tide stars become later-branch stars with finite
  Lambda. This is the main change from v1. In the twin interval, w_b = 1/n
  averages likelihoods whose Lambda differ by about 8x. With w_b = 1, Z_e also
  changes by the factor ∫ n_e pi / ∫ pi. For a 0.06 M_sun interval inside about
  0.4 M_sun of posterior mass support, that factor is at most about 0.15 nat. The
  per-run Monte Carlo scatter of the fixed-EOS path is about 1 nat (PAPER-NOTES
  §0b).

### (b) Sample the central coordinate per component

Sample s1, s2 in the stable set S_e (the union of branches), with
m_i = M_e(s_i) and Lambda_i = Lambda_e(s_i).

- **CIP change.**
  - New low-level coordinates `hc1, hc2` with `prior_map` entries.
    `convert_waveform_coordinates_with_eos` maps them to (m, Lambda) through new
    reader methods `M_of_hc_indx` and `Lambda_of_hc_indx`.
  - `m_max_of_indx` is not needed, because the domain is S_e.
  - `mc-range` and `eta-range` become derived cuts, and `m2 <= m1` becomes
    `M(s2) <= M(s1)`.
  - In lalsuite, use `FamMassOfCentralPressure` to get the branch and mass, then
    `...OfMassPerBranch`.
- **Mass prior.** If p(s | e) = pi(M(s)) |dM/ds|, the mass prior is n_e(m) pi(m),
  which is (a) with w_b = 1. If p(s | e) is uniform in h_c, then p(m | e) is
  proportional to the sum over b of 1/|dM/dh_c|. This density is EOS-dependent and
  peaks at each branch maximum (integrable).
- **Evidence for twins.** With the Jacobian prior, the result is the same as (a)
  with w_b = 1. With a uniform h_c prior, Z_e favours EOS whose branch maxima lie
  near the observed masses. That is an assumption about NS formation.
- **Cost.** mc is measured to about 1e-3 and becomes a derived, EOS-dependent
  function of (s1, s2), so sampling in s is inefficient. The tabular ordering
  method cannot use (b), because each EOS index has a different S_e.

### The tabular ordering method

The tabular method fits lnL(mc, delta_mc, chi, ordering). Lambda is assigned when
the grid is built (`util_ManualOverlapGrid`). Once branch labels vary, Lambda is not
a function of (m, ordering), so the fit would average lnL over branches (biased).
The tabular path should stay mass-affine. Multi-branch per-EOS evidence then comes
from the fixed-EOS path (`CIPPerEOSEvidence`, `--using-eos nmbseq:FILE:e`).

## Proposed sequence of work

1. Fix defect 1: port `_stable_branches` to RIFT's `EOSSequenceNMB` and
   `EOSSequencePCA`. Settle defect 2 (open question 2).
2. Add `--eos-affine-parameter {mass,central}` to CIP and `affine=` to
   `EOSSequenceFromFile`. The default is `mass`.
3. Implement `central` as (a), with the branch label in the fixed-EOS path only.
4. Port `EOSConcrete` to the 7.0.0 API: primary branch for `mass`, per-branch
   accessors for `central`. Pin lalsimulation below 7 until this is done.

## Open questions for Richard

1. Prior along the central coordinate: Jacobian with branch counting (w_b = 1),
   normalized mixture (w_b = 1/n), or uniform in h_c (or in baryon mass)?
2. For m above m_max, `rift_O4c` gives zero prior (sample rejected). Keep that?
   Central mode also rejects masses below the lowest branch.
3. Does paper 1 apply an M_max >= 2 M_sun pulsar constraint? If so, all 195 twin
   draws get zero weight, and the branch contract does not change paper-1 results.
4. Are the 40 draws with dips under 2% physical? Should a branch need a minimum
   dip or interval width?
5. Central coordinate for the interface: h_c (NMB storage), p_c (lalsuite), or
   rho_c? Once the prior is defined in mass they are equivalent.
6. Should the tabular ordering method stay mass-only, or should its fit take
   Lambda1 and Lambda2 as coordinates?
7. Who ports `EOSConcrete` to lalsimulation 7.0.0, and do we pin below 7 until then?
8. Do the 3 three-branch draws count in `"all"`?
