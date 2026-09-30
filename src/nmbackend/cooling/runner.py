"""Run NSCool (D. Page) on one star and return its cooling curve.

NSCool is built outside the repository by ``build_nscool.sh``; see
README.md in this directory for the source, the patch and the input
formats.  Each call writes NSCool's input files into a fresh directory,
runs the binary there and parses the T_eff file.

NSCool prints the redshifted (observed at infinity) effective
temperature and photon luminosity.  ``CoolingCurve`` carries those and
the local surface values T_s = T_inf / e^phi_s, L_local = L_inf / e^(2 phi_s)
with e^(2 phi_s) = 1 - 2GM/(R c^2) at NSCool's outer grid radius.
"""
from __future__ import annotations

import inspect
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np

__all__ = ["GapModel", "CoolingCurve", "IsothermalStar", "NSCool", "NSCoolError",
           "default_home", "parse_teff_file", "parse_star_file",
           "parse_isothermal"]

PathLike = Union[str, os.PathLike]

#: G Msun / c^2 in cm
_GM_SUN_C2_CM = 1.4766250e5


class NSCoolError(RuntimeError):
    """NSCool or its TOV solver failed or produced no curve."""


def default_home() -> Path:
    """Built NSCool tree: $NSCOOL_HOME/NSCool, else the build-script default."""
    root = os.environ.get("NSCOOL_HOME",
                          str(Path.home() / ".local/share/nmbackend/nscool"))
    return Path(root) / "NSCool"


@dataclass(frozen=True)
class GapModel:
    """Pairing gaps: NSCool model codes plus Zhou+2026 Eq. 10 scalings.

    Delta_x(n) = s_y * Delta_x,0(n / s_x) for x = p1S0 and n3P2, with n the
    baryon density.  s_y maps to NSCool's FP1S0 / FN3P2 (Tc multipliers;
    Tc is proportional to Delta).  s_x is applied by the nmbackend patch in
    the core zones: the gap is evaluated at the Fermi momentum the star has
    at n / s_x.  Where n / s_x lies outside the core (above the centre, or
    below the crust-core boundary) k_F scales as n^(1/3) from the nearest
    core zone.  Crust zones are not rescaled.

    Model codes are NSCool's SFP1S0 / SFN3P2 / SFN1S0 values (precool.f,
    subroutine get_Tc); 0 switches a channel off.  Defaults are the
    bundled Model_1: n1S0 SFB, n3P2 minimal-cooling gap "a", p1S0 T73.

    ``ca_n3p2`` is the axial coefficient of n3P2 pair-breaking emission,
    a_A = c_A g_A^2 (nmbackend patch).  The default 0.5 follows Leinson's
    correction, as in A. W. Steiner's NSCool fork; the 2010 release, and
    its bundled Model_1 output, use 2.
    """
    n1s0: float = 1.0
    n3p2: float = 101.0
    p1s0: float = 3.0
    sx_p1s0: float = 1.0
    sy_p1s0: float = 1.0
    sx_n3p2: float = 1.0
    sy_n3p2: float = 1.0
    ca_n3p2: float = 0.5

    def __post_init__(self):
        for name in self.__dataclass_fields__:   # numpy scalars -> float
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.n3p2 == 150.0:
            raise ValueError("SFN3P2=150 (user Gaussian gap) is not supported")
        for name in ("sx_p1s0", "sy_p1s0", "sx_n3p2", "sy_n3p2", "ca_n3p2"):
            if not 0 < getattr(self, name) < np.inf:
                raise ValueError(f"{name} must be finite and > 0")

    def pairing_file(self) -> str:
        return (
            "PAIRING CONTROL:---------------------------------------\n"
            f"{self.n1s0:<10.10g} SFN1S0\n"
            f"{self.n3p2:<10.10g} SFN3P2\n"
            f"{self.p1s0:<10.10g} SFP1S0\n"
            "0.        SFL1S0\n"
            "1.        FN1S0\n"
            f"{self.sy_n3p2:<10.10g} FN3P2\n"
            f"{self.sy_p1s0:<10.10g} FP1S0\n"
            "1.        FL1S0\n"
            "0.        SFQUARK\n"
            f"{self.sx_p1s0:<10.10g} SXP1S0\n"
            f"{self.sx_n3p2:<10.10g} SXN3P2\n"
            f"{self.ca_n3p2:<10.10g} CA3P2\n")


class CoolingCurve(NamedTuple):
    """One NSCool run.  Times in yr, temperatures in K, luminosities in erg/s."""
    t_yr: np.ndarray
    Teff_inf: np.ndarray      # redshifted effective temperature
    L_gamma_inf: np.ndarray   # redshifted photon luminosity
    L_nu_inf: np.ndarray      # redshifted neutrino luminosity
    L_heat_inf: np.ndarray
    mass: float               # gravitational mass, Msun
    radius_cm: float          # NSCool grid edge at rhosurf, not the TOV surface
    star: Optional[dict] = None   # per-zone profile if run(star=True)

    @property
    def exp_phi_s(self) -> float:
        return float(np.sqrt(1.0 - 2.0 * _GM_SUN_C2_CM * self.mass / self.radius_cm))

    @property
    def Ts(self) -> np.ndarray:
        """Local (unredshifted) surface temperature."""
        return self.Teff_inf / self.exp_phi_s

    @property
    def L_gamma(self) -> np.ndarray:
        """Local photon luminosity."""
        return self.L_gamma_inf / self.exp_phi_s ** 2

    def at(self, t_yr, quantity: str = "Teff_inf") -> np.ndarray:
        """log-log interpolation of a column at ages t_yr."""
        y = getattr(self, quantity)
        return np.exp(np.interp(np.log(np.asarray(t_yr, float)),
                                np.log(self.t_yr), np.log(y)))


_HEADER_RE = re.compile(r"Mstar\s*=\s*([-+.\dEe]+)\s+Rstar\s*=\s*([-+.\dEe]+)")


def parse_teff_file(path: PathLike) -> CoolingCurve:
    """Parse the T_eff file NSCool writes with PTEFF=1."""
    text = Path(path).read_text()
    m = _HEADER_RE.search(text)
    if m is None:
        raise NSCoolError(f"{path}: no Mstar/Rstar header")
    rows = []
    for line in text.splitlines():
        f = line.split()
        if len(f) == 6 and f[0].isdigit():
            rows.append([float(x) for x in f[1:]])
    if not rows:
        raise NSCoolError(f"{path}: no time steps")
    a = np.array(rows)
    return CoolingCurve(a[:, 0], a[:, 1], a[:, 2], a[:, 3], a[:, 4],
                        float(m.group(1)), float(m.group(2)))


_STAR_COLS = ("i", "rad", "emas", "rho", "pres", "nb", "kfe", "kfmu", "kfp",
              "kfn", "kfla", "kfsm", "kfs0", "kfsp", "Tcn", "Tcp", "Tcla")


def parse_star_file(path: PathLike) -> dict:
    """Core-zone arrays (i = 0..icore) from the Star file NSCool writes with PSTAR=1.

    Radii in m, densities in g/cm^3 (rho) and fm^-3 (nb), Fermi momenta
    in fm^-1, critical temperatures in K (1 K means unpaired).
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if len(f) >= len(_STAR_COLS) and f[0].isdigit():
            try:
                row = [float(x) for x in f[:len(_STAR_COLS)]]
            except ValueError:
                continue
            if row[0] == 0 and rows:
                break                  # first profile only
            rows.append(row)
    if not rows:
        raise NSCoolError(f"{path}: no zones")
    a = np.array(rows)
    return {k: a[:, j] for j, k in enumerate(_STAR_COLS)}


class IsothermalStar(NamedTuple):
    """NSCool physics summed over an isothermal star (``NSCool.isothermal``).

    ``T`` is the redshifted interior temperature (K).  C in erg/K; all
    luminosities in erg/s at infinity; ``eta_nscool`` is the eta NSCool's
    envelope formula receives (input ETA / 0.443).  ``T_in`` answers a
    different question from the other columns: with T_b = T at NSCool's
    outer grid point, it is the redshifted temperature at density rho_b,
    from NSCool's steady heat-flux equation at L = L_gamma(T).
    """
    T: np.ndarray
    C: np.ndarray
    L_nu: np.ndarray
    L_heat: np.ndarray
    L_gamma: np.ndarray
    Teff_inf: np.ndarray
    L_durca: np.ndarray
    L_murca_brem: np.ndarray
    L_pbf: np.ndarray
    L_nu_crust: np.ndarray
    C_core: np.ndarray
    T_in: np.ndarray          # T at rho_b when T_b = T (flux march); = T if rho_b unset
    mass: float
    radius_cm: float
    exp_phi_s: float
    eta_nscool: float


def parse_isothermal(text: str) -> IsothermalStar:
    """Parse the stdout of ``NSCool_iso.out``."""
    m = re.search(r"star M R e\^phi_s =\s*(\S+)\s+(\S+)\s+(\S+)", text)
    e = re.search(r"eta used by fteff =\s*(\S+)", text)
    rows = [[float(x) for x in l.split()[1:]] for l in text.splitlines()
            if l.startswith("ISO ")]
    if m is None or e is None or not rows:
        raise NSCoolError("no isothermal scan in NSCool_iso output")
    a = np.array(rows).T
    if len(a) == 11:                             # binary without the T_in column
        a = np.vstack([a, a[0]])
    return IsothermalStar(*a, float(m.group(1)), float(m.group(2)),
                          float(m.group(3)), float(e.group(1)))


_I_DAT = """PRINT OUT CONTROL:----------------------------------------
0	PSCREEN
0.      DEBUG
0       ISTEP DEBUG
1	PTEFF
0	PTEMP
{pstar}	PSTAR
1	IDUMP1
111	IDUMP2
421	IDUMP3
{tempmin:.3e}	TEMPMIN
{tempini:.3e}	TEMPINI
ELECTRON SPECIFIC HEAT in OUTER CRUST CONTROL:------------
0       ICVEL_NODEG
EFFECTIVE MASS CONTROL:-----------------------------------
5	EMNCO
5	EMNCR
3	EMP
INITIAL ROTATIONAL PERIOD:--------------------------------
0.1	p0: initial spin period
TPRINT: times at which you select the profiles---------
1.0e-10
1.0e+7
"""

_BOUND = """ENVELOPE MODEL & BOUNDARY CONDITION:----------------------
3	IFTEFF   (0=use table, 10,11=magnetar with Fe and Acc)
{eta:.6e}	ETA (for accreted env., i.e., IFTEFF=3)
3.	MAG_COEFF (for magnetars, i.e., IFTEFF=10,11)
0.0	TB_ACC0
'none-at-all.nut'
"""

_STRUCT = ("{rhocore:.4e}   {rhodrip:.4e}   {rhoenv:.4e}   {rhosurf:.4e}"
           "  :  rhocore  rhodrip  rhoenv  rhosurf\n{icore}  {idec}  :  icore idec\n")


class NSCool:
    """Driver for a built NSCool tree (see ``build_nscool.sh``).

    Parameters
    ----------
    home : built ``NSCool`` directory (default: ``default_home()``).
    workroot : parent of the per-run directories (default: system temp).
    """

    def __init__(self, home: Optional[PathLike] = None,
                 workroot: Optional[PathLike] = None):
        self.home = Path(home) if home is not None else default_home()
        self.workroot = workroot
        for exe in (self.exe, self.tov_exe):
            if not exe.is_file():
                raise FileNotFoundError(
                    f"{exe} missing: run nmbackend/cooling/build_nscool.sh")

    @property
    def exe(self) -> Path:
        return self.home / "Code" / "NSCool.out"

    @property
    def tov_exe(self) -> Path:
        return self.home / "TOV" / "TOV"

    @property
    def default_eos(self) -> Path:
        """Bundled APR core+crust table (Akmal, Pandharipande & Ravenhall)."""
        return self.home / "EOS" / "APR_EOS_Cat.dat"

    @property
    def default_crust(self) -> Path:
        return self.home / "EOS" / "Crust" / "Crust_EOS_Cat_HZD-NV.dat"

    # ------------------------------------------------------------------
    def _tov(self, d: Path, cmds: str, timeout: float):
        """Run NSCool's TOV in ``d``; return (rhoc [fm^-3], M [Msun]) of each star."""
        try:
            r = subprocess.run([str(self.tov_exe)], cwd=d, input=cmds,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise NSCoolError(f"TOV timed out after {timeout} s") from e
        rhoc = [float(x) for x in re.findall(r"rhoc:\s*([-+.\dEe]+)", r.stdout)]
        mass = [float(x) for x in re.findall(r"masses\(g,b,p\):\s*([-+.\dEe]+)", r.stdout)]
        if r.returncode != 0 or not mass or len(rhoc) != len(mass):
            raise NSCoolError(f"TOV failed (rc={r.returncode}):\n{r.stdout[-2000:]}")
        return rhoc, mass

    def make_profile(self, eos_table: PathLike, mass: float, out: PathLike,
                     rhoc_start: float = 0.3, timeout: float = 120.0) -> Path:
        """Build a stellar profile at gravitational mass ``mass`` with NSCool's TOV.

        ``rhoc_start`` (fm^-3) seeds the target-mass search.  The star must be
        on the stable branch: a second TOV solve at 1.002 rho_c must give a
        larger mass, else ``NSCoolError``.  Returns ``out``.
        """
        out = Path(out)
        with tempfile.TemporaryDirectory(dir=self.workroot) as d:
            d = Path(d)
            (d / "Profile").mkdir()
            os.symlink(Path(eos_table).resolve(), d / "eos.dat")
            head = "cgs\neos='eos.dat'\ngeteos\n"
            rhoc, m = self._tov(d, head + f"rhoc={rhoc_start}\ngo\n"
                                f"targetmass={mass}\ngo\ndump\ngo\nEOF\n", timeout)
            prof = d / "Profile" / "prof.dat"
            if not prof.is_file():
                raise NSCoolError("TOV wrote no profile")
            if abs(m[-1] - mass) > 1e-4 * mass:
                raise NSCoolError(f"TOV reached M={m[-1]}, target {mass}")
            _, m2 = self._tov(d, head + f"rhoc={1.002 * rhoc[-1]:.6e}\ngo\nEOF\n",
                              timeout)
            if not m2[-1] > m[-1]:
                raise NSCoolError(f"TOV star at rho_c={rhoc[-1]} fm^-3 is on the "
                                  f"unstable branch (dM/drho_c <= 0)")
            shutil.copy(prof, out)
        return out

    # ------------------------------------------------------------------
    def _setup(self, d: Path, *, mass, profile, eos_table, crust_table, gaps,
               eta, rhocore, rhodrip, rhoenv, rhosurf, icore, idec, tempini,
               tempmin, neutrino_file, conduct_file, star) -> None:
        """Write NSCool's input files for one star into run directory ``d``."""
        if (mass is None) == (profile is None):
            raise ValueError("give exactly one of mass, profile")
        eos_table = Path(eos_table) if eos_table else self.default_eos
        crust_table = Path(crust_table) if crust_table else self.default_crust
        m1 = d / "Model_1"      # NSCool prefixes the master file with Model_1/
        m1.mkdir()
        os.symlink(self.home / "Code", d / "Code")   # Code/Data_Files/*
        os.symlink(eos_table.resolve(), d / "eos.dat")
        os.symlink(crust_table.resolve(), d / "crust.dat")
        if profile is None:
            self.make_profile(eos_table, mass, d / "prof.dat")
        else:
            os.symlink(Path(profile).resolve(), d / "prof.dat")
        files = {
            "I_Struct.dat": _STRUCT.format(rhocore=rhocore, rhodrip=rhodrip,
                                           rhoenv=rhoenv, rhosurf=rhosurf,
                                           icore=icore, idec=idec),
            "I_Bound.dat": _BOUND.format(eta=eta),
            "I_Pairing.dat": gaps.pairing_file(),
            "I.dat": _I_DAT.format(tempmin=tempmin, tempini=tempini,
                                    pstar=int(star)),
        }
        for name, text in files.items():
            (m1 / name).write_text(text)
        for name in (neutrino_file, conduct_file, "I_Heat_0.dat",
                     "I_Bfield_0.dat", "I_Accretion_0.dat"):
            shutil.copy(self.home / "I_Files" / name, m1 / name)
        master = ["'NEW'", "BASIC MODEL FILES:", "'crust.dat'", "'eos.dat'",
                  "'prof.dat'", "OTHER MODEL FILES:",
                  "'Model_1/I_Struct.dat'", "'Model_1/I_Bound.dat'",
                  "'Model_1/I_Pairing.dat'", f"'Model_1/{neutrino_file}'",
                  f"'Model_1/{conduct_file}'", "'Model_1/I_Heat_0.dat'",
                  "'Model_1/I_Bfield_0.dat'", "'Model_1/I_Accretion_0.dat'",
                  "OUTPUT FILES:", "'Model_1/I.dat'", "'Model_1/Teff.dat'",
                  "'Model_1/Temp.dat'", "'Model_1/Star.dat'", ""]
        (m1 / "Cool.in").write_text("\n".join(master))

    def _exec(self, exe: Path, d: Path, stdin: str, gaps: GapModel,
              timeout: float, keep: Optional[PathLike]) -> str:
        """Run ``exe`` in ``d``; check the run and the s_x echo; return stdout."""
        env = dict(os.environ, OMP_NUM_THREADS="1")
        try:
            r = subprocess.run([str(exe)], cwd=d, input=stdin,
                               capture_output=True, text=True,
                               timeout=timeout, env=env)
        except subprocess.TimeoutExpired as e:
            raise NSCoolError(f"{exe.name} timed out after {timeout} s") from e
        log = r.stdout + r.stderr    # gfortran writes PAUSE to stderr
        (d / "stdout.txt").write_text(log)
        if keep is not None:
            shutil.copytree(d, keep, symlinks=True, dirs_exist_ok=True)
        if r.returncode != 0 or "Done" not in r.stdout or "PAUSE" in log.upper():
            raise NSCoolError(f"{exe.name} failed (rc={r.returncode}):\n{log[-2000:]}")
        m = re.search(r"nmbackend sx_p1s0 sx_n3p2 =\s*(\S+)\s+(\S+)", r.stdout)
        if m is None or not np.allclose([float(m.group(1)), float(m.group(2))],
                                        [gaps.sx_p1s0, gaps.sx_n3p2], rtol=1e-5):
            raise NSCoolError("NSCool did not read the s_x values written "
                              f"({m.group(0) if m else 'no echo: unpatched build?'})")
        m = re.search(r"nmbackend ca_n3p2 =\s*(\S+)", r.stdout)
        if m is None or not np.isclose(float(m.group(1)), gaps.ca_n3p2, rtol=1e-5):
            raise NSCoolError("NSCool did not read CA3P2 "
                              f"({m.group(0) if m else 'no echo: build predates CA3P2?'})")
        return r.stdout

    def run(self, *, mass: Optional[float] = None,
            profile: Optional[PathLike] = None,
            eos_table: Optional[PathLike] = None,
            crust_table: Optional[PathLike] = None,
            gaps: GapModel = GapModel(),
            eta: float = 0.0,
            rhocore: float = 1.6e14, rhodrip: float = 4.0e11,
            rhoenv: float = 1.0e8, rhosurf: float = 1.0e10,
            icore: int = 111, idec: int = 60,
            tempini: float = 1.0e10, tempmin: float = 1.0e4,
            neutrino_file: str = "I_Neutrino_1.dat",
            conduct_file: str = "I_Conduct_32.dat",
            star: bool = False,
            timeout: float = 600.0,
            keep: Optional[PathLike] = None) -> CoolingCurve:
        """Cool one star.

        Give either ``mass`` (built from ``eos_table`` with NSCool's TOV) or
        a ready ``profile``.  ``eos_table`` is an NSCool 'NEW'-format core+crust
        table (README.md); default the bundled APR.  ``eta`` is written to
        NSCool's ``ETA``; NSCool divides it by 0.443 before the envelope
        formula of Potekhin, Chabrier & Yakovlev (1997) (README.md).
        Grid defaults follow the bundled Model_1
        (I_Struct_1.6e14-4.0e11-1e10_normal.dat).
        ``star=True`` also returns NSCool's per-zone profile (``parse_star_file``).
        ``keep`` copies the run directory there for inspection.
        """
        opts = dict(mass=mass, profile=profile, eos_table=eos_table,
                    crust_table=crust_table, gaps=gaps, eta=eta,
                    rhocore=rhocore, rhodrip=rhodrip, rhoenv=rhoenv,
                    rhosurf=rhosurf, icore=icore, idec=idec, tempini=tempini,
                    tempmin=tempmin, neutrino_file=neutrino_file,
                    conduct_file=conduct_file, star=star)
        with tempfile.TemporaryDirectory(dir=self.workroot) as d:
            d = Path(d)
            self._setup(d, **opts)
            self._exec(self.exe, d, "Cool.in\n", gaps, timeout, keep)
            curve = parse_teff_file(d / "Model_1" / "Teff.dat")
            if star:
                curve = curve._replace(star=parse_star_file(d / "Model_1" / "Star.dat"))
            return curve

    def isothermal(self, *, T_grid=(1e4, 1e10, 601), rho_b: float = 0.0,
                   timeout: float = 600.0,
                   keep: Optional[PathLike] = None, **kw) -> "IsothermalStar":
        """C(T), L_nu(T), L_gamma(T) of the star with uniform redshifted T.

        Takes the star and physics arguments of ``run`` and returns NSCool's
        own heat capacity and luminosities summed over zones at each
        redshifted temperature of the log grid ``T_grid = (T_min, T_max, n)``
        (``NSCool_iso.out``, built from ``nscool_isothermal_tail.f``).
        ``rho_b`` (g/cm^3) > ``rhosurf`` also fills ``T_in``, the temperature
        at rho_b below a blanket at T_b = T (``isothermal_curve``).
        """
        if not self.iso_exe.is_file():
            raise FileNotFoundError(f"{self.iso_exe} missing: rerun build_nscool.sh")
        tmin, tmax, n = T_grid
        opts = {k: v.default for k, v in inspect.signature(self.run).parameters.items()
                if k not in ("timeout", "keep")}
        unknown = set(kw) - set(opts)
        if unknown:
            raise TypeError(f"unknown arguments {sorted(unknown)}")
        opts.update(kw)
        with tempfile.TemporaryDirectory(dir=self.workroot) as d:
            d = Path(d)
            self._setup(d, **opts)
            out = self._exec(self.iso_exe, d,
                             f"Cool.in\n{int(n)} {tmin:.8e} {tmax:.8e} {rho_b:.8e}\n",
                             opts["gaps"], timeout, keep)
        return parse_isothermal(out)

    @property
    def iso_exe(self) -> Path:
        return self.home / "Code" / "NSCool_iso.out"
