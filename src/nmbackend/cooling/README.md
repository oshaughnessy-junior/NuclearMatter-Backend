# nmbackend.cooling: NSCool driver

This package runs D. Page's NSCool on one star and returns its cooling curve.
It does not contain NSCool. The build script downloads and builds it.

## Source

| Item | Value |
|---|---|
| Code | NSCool, D. Page (UNAM), Fortran 77 |
| URL | https://www.astroscu.unam.mx/neutrones/NSCool/NSCool.tar.gz |
| Version | unversioned tarball, files dated to 2010-02-01 |
| sha256 | `c671fb959c732af595834152416cb60336009aa0e218b65ead55cae32f6af35a` (fetched 2026-09-29) |
| Licence | none stated in the tarball or on the web page |
| Guides | same URL: `NSCool_Guide_1.1_*.pdf`, `NSCool_Guide_0.0_Control.pdf`, `TOV_Guide.pdf` |

With no licence we do not redistribute NSCool. `build_nscool.sh` fetches it,
checks the sha256, applies `nscool_nmbackend.patch` and compiles.

## Build

```bash
src/nmbackend/cooling/build_nscool.sh      # installs into ~/.local/share/nmbackend/nscool
```

Set `NSCOOL_HOME` to change the prefix (the driver reads the same variable).
Set `NSCOOL_TARBALL` to build from a local tarball. The CIT build of 2026-09-29
used gfortran 8.5.0 with `-O3 -fno-automatic -std=legacy -w` on ldas-pcdev13.
It took 10 s. `NSCool/BUILD_INFO.txt` records each build.

The patch changes five files:

| File | Change |
|---|---|
| `Code/pairing.inc.f`, `Code/NSCool_READ.inc.f` | optional trailing lines `SXP1S0`, `SXN3P2` (default 1) and `CA3P2` (default 0.5) in the pairing file; NSCool prints the values it read, and the driver checks them |
| `Code/neutrino_core.f` (`nu_n3p2_B_pbf`) | n3P2 pair-breaking axial coefficient a_A = c_A g_A^2 with c_A = `CA3P2`. The release has c_A = 2. We use 0.5, Leinson's correction for the anomalous axial contribution, as in A. W. Steiner's fork (github awsteiner/nscool). This cuts n3P2 pair-breaking emission by 4. |
| `Code/precool.f` (`get_Tc`) | in core zones, p1S0 and n3P2 gaps are evaluated at the Fermi momentum the star has at baryon density n/s_x; outside the core density range k_F scales as n^(1/3) from the nearest core zone; crust zones are not rescaled |
| `TOV/TOV.f` | drop the hard-coded `/Users/dany/...` EOS prefix; call `gread` as the function it is (gfortran rejects `CALL`) |

With s_x = 1 and `GapModel(ca_n3p2=2.0)` the patched binary reproduces the
bundled Model_1 output to the printed digits. Default runs differ from it,
because the default c_A is 0.5. A build without `CA3P2` fails the driver's
echo check; rerun `build_nscool.sh`.

## Use

```python
from nmbackend.cooling import NSCool, GapModel
ns = NSCool()
c = ns.run(mass=1.4, gaps=GapModel(p1s0=3, sx_p1s0=0.8, sy_p1s0=0.4), eta=1e-9)
c.t_yr, c.Teff_inf, c.L_gamma_inf   # NSCool output, redshifted to infinity
c.Ts, c.L_gamma                     # local surface values
```

- `mass`: the driver builds the star with NSCool's TOV solver from `eos_table`.
  It raises `NSCoolError` if the star is on the unstable branch.
  Or pass `profile=` for a ready profile.
- `eos_table`, `crust_table`: default the bundled APR table and the HZD-NV crust.
- `GapModel`: NSCool gap codes (`SFP1S0`, `SFN3P2`, `SFN1S0`; see `get_Tc` in
  `precool.f`) and the scalings of Zhou+2026 (arXiv:2609.34338) Eq. 10,
  Delta(n) = s_y Delta_0(n/s_x). The code applies s_y through NSCool's own
  `FP1S0`/`FN3P2` factors and s_x through the patch.
- `eta`: written to NSCool's `ETA`. NSCool divides it by 0.443 before the
  envelope formula (see Notes).
- `star=True` also returns the core-zone profile (n_b, k_F, T_c).

Each run uses a new temporary directory. Pass `workroot=` to put these on a
node-local disk.

## Isothermal star

`NSCool.isothermal()` takes the arguments of `run()` and returns NSCool's
heat capacity C, L_nu and L_gamma for a star at uniform redshifted T, on a
log grid of T. `Code/NSCool_iso.out` is the setup part of `NSCool.f` plus
`nscool_isothermal_tail.f`, which calls `neutrino`, `heating`, `specheat`
and `fteff` as the time loop does: C = sum cv dV, L_nu = sum e^(2 phi) q_nu dV.

```python
from nmbackend.cooling import NSCool, isothermal_curve
s = NSCool().isothermal(mass=1.4, eta=1e-9, rho_b=1.6e14)
c = isothermal_curve(s, blanket=True)   # t(T) = int C dT / (L_nu + L_gamma)
c.t_yr, c.Teff_inf
```

With `rho_b` set, the driver marches NSCool's steady flux equation from
T_b = T at 1e10 g/cm3 in to rho_b at L = L_gamma(T_b). `blanket=True` then
makes only the star below rho_b isothermal. A 961-point scan takes 1 s.
Other approximations: envelope densities stay at NSCool's initial ones; the
crust layer's C and L_nu use T_in; `C_core` sums zones i <= icore.
Accuracy against full NSCool: nmb-papers
`paper7_cooling_population/notes/RESULTS-C1-isothermal.md`.

## EOS table format (interface with W2, `nscool_io`)

NSCool reads one core+crust table, as in the bundled `EOS/APR_EOS_Cat.dat`.
The TOV solver reads the same file.

1. Seven header lines. Line 1 is `6 Imax Icore` (integers). The first number must
   be 6: `TOV.f` skips six lines, and NSCool skips `Itext` lines after line 1.
   So line 7 must be blank: TOV reads it as data and fails on text.
2. `Icore` core rows, densest first, 17 columns:
   `rho[g/cm3] P[dyn/cm2] nbar[fm^-3] Ye Ymu Yn Yp Yla Ysm Ys0 Ysp mstp mstn mstla mstsm msts0 mstsp`.
   `rho` is the total energy density over c^2 (TOV uses it as such), not the
   rest-mass density. Y is the number per baryon. `mst*` is the Landau effective mass over the bare
   mass. NSCool uses these m* in the core (version `'NEW'`).
3. Crust rows below, densest first: `rho P n A_cell A_ion Z`. TOV uses rho, P and n.
   The cooling code takes crust composition from the separate crust file.

Limits in the Fortran:

| Limit | Source |
|---|---|
| Icore <= 500 | `get_core_chemistry` arrays |
| all rows <= 1000 | `TOV.f` `limit` |
| path <= 60 characters | TOV `title`; the driver symlinks the table to `eos.dat` |
| sum of baryon Y within 1e-2 of 1, and net charge per baryon below 1e-2 | `get_core_chemistry`, on the interpolated core zones; else Fortran `PAUSE` |

The core-crust boundary of the cooling grid is `rhocore` (default 1.6e14 g/cm3).
It is a `run()` argument. The table does not set it.

## Validation and cost

`tests/test_cooling_nscool.py` checks the driver against NSCool's bundled
`Model_1/Teff_Try_KEEP.dat` (APR, 1.4 Msun, SFB / gap "a" / T73, iron
envelope). All 246 time steps agree to the printed four digits. This holds for
the bundled profile and for a profile built by the driver's TOV call. The
figure is `docs/cooling/nscool_validation.png`
(`python -m nmbackend.cooling.validate --fig ...`).

One curve takes 3.1 s of wall time on one core of ldas-pcdev11 (Xeon E5-2630 v4).
TOV adds 0.01 s.

## Notes for users

- NSCool divides the input `ETA` by 0.443 (`NSCool_READ.inc.f`, line 78) and
  passes the result to `fteff_acc`. That function is Eqs. (A7)-(A10) of
  Potekhin, Chabrier and Yakovlev (1997), where eta = g_s14^2 Delta M / M
  (their Eq. A5). So the envelope formula sees eta = ETA / 0.443 = 2.26 ETA.
  Neither the source nor the guides explain the factor. `run(eta=x)` means
  PCY97 eta = 2.26 x. `IsothermalStar.eta_nscool` reports the value used.
- NSCool's default time step (`Code/Data_Files/NUM_PARAM.dat`: DT0 = TVAR = 1.2)
  is not converged at late times. The error in log10 T_eff^inf is about
  0.25 (DT0 - 1) at 1 Myr for a GM1 1.6 Msun star, and larger in the
  photon-cooling drop. For reference curves, copy the tree and lower DT0
  and TVAR together. Some stars with pairing then stall (for example APR
  1.1 Msun at DT0 = 1.02 ran for over 25 min); give `run()` a timeout.
- The T73 and CCDK proton gaps close below n = 0.6 fm^-3. For APR the dUrca core of a
  2.1 Msun star lies above that. So s_x <= 1, the range of Zhou+2026, cannot
  quench dUrca there. With CCDK, s_x = 2 can.
- s_x < 1 moves the density where a gap closes down into the core. With CCDK at
  1.4 Msun, s_x = 0.5 leaves 52 of 112 core zones unpaired.
- NSCool Ioffe proton gaps (`SFP1S0` = 201-203) are called with the neutron k_F
  in `get_Tc`, and s_x does not act on them.
