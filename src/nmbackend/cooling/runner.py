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

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Optional, Union

import numpy as np

__all__ = ["GapModel", "CoolingCurve", "NSCool", "NSCoolError",
           "default_home", "parse_teff_file", "parse_star_file"]

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
    Tc is proportional to Delta).  s_x is applied by the nmbackend patch,
    which evaluates the gap at the Fermi momentum the star has at n / s_x.

    Model codes are NSCool's SFP1S0 / SFN3P2 / SFN1S0 values (precool.f,
    subroutine get_Tc); 0 switches a channel off.  Defaults are the
    bundled Model_1: n1S0 SFB, n3P2 minimal-cooling gap "a", p1S0 T73.
    """
    n1s0: float = 1.0
    n3p2: float = 101.0
    p1s0: float = 3.0
    sx_p1s0: float = 1.0
    sy_p1s0: float = 1.0
    sx_n3p2: float = 1.0
    sy_n3p2: float = 1.0

    def __post_init__(self):
        if self.n3p2 == 150.0:
            raise ValueError("SFN3P2=150 (user Gaussian gap) is not supported")
        for name in ("sx_p1s0", "sy_p1s0", "sx_n3p2", "sy_n3p2"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be > 0")

    def pairing_file(self) -> str:
        return (
            "PAIRING CONTROL:---------------------------------------\n"
            f"{self.n1s0!r:<10} SFN1S0\n"
            f"{self.n3p2!r:<10} SFN3P2\n"
            f"{self.p1s0!r:<10} SFP1S0\n"
            "0.        SFL1S0\n"
            "1.        FN1S0\n"
            f"{self.sy_n3p2!r:<10} FN3P2\n"
            f"{self.sy_p1s0!r:<10} FP1S0\n"
            "1.        FL1S0\n"
            "0.        SFQUARK\n"
            f"{self.sx_p1s0!r:<10} SXP1S0\n"
            f"{self.sx_n3p2!r:<10} SXN3P2\n")


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

    Radii in cm, densities in g/cm^3 (rho) and fm^-3 (nb), Fermi momenta
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
    def make_profile(self, eos_table: PathLike, mass: float, out: PathLike,
                     rhoc_start: float = 0.3, timeout: float = 120.0) -> Path:
        """Build a stellar profile at gravitational mass ``mass`` with NSCool's TOV.

        ``rhoc_start`` (fm^-3) seeds the target-mass search; it must lie on
        the stable branch below the target.  Returns ``out``.
        """
        out = Path(out)
        with tempfile.TemporaryDirectory(dir=self.workroot) as d:
            d = Path(d)
            (d / "Profile").mkdir()
            os.symlink(Path(eos_table).resolve(), d / "eos.dat")
            cmds = ("cgs\neos='eos.dat'\ngeteos\n"
                    f"rhoc={rhoc_start}\ngo\ntargetmass={mass}\ngo\n"
                    "dump\ngo\nEOF\n")
            try:
                r = subprocess.run([str(self.tov_exe)], cwd=d, input=cmds,
                                   capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired as e:
                raise NSCoolError(f"TOV timed out after {timeout} s") from e
            prof = d / "Profile" / "prof.dat"
            masses = re.findall(r"masses\(g,b,p\):\s*([-+.\dEe]+)", r.stdout)
            if r.returncode != 0 or not prof.is_file() or not masses:
                raise NSCoolError(f"TOV failed (rc={r.returncode}):\n{r.stdout[-2000:]}")
            if abs(float(masses[-1]) - mass) > 1e-4 * mass:
                raise NSCoolError(f"TOV reached M={masses[-1]}, target {mass}")
            shutil.copy(prof, out)
        return out

    # ------------------------------------------------------------------
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
        table (README.md); default the bundled APR.  ``eta`` is the
        light-element envelope parameter of Potekhin, Chabrier & Yakovlev
        (1997), eta = g_s14^2 Delta M_light / M.  Grid defaults follow the
        bundled Model_1 (I_Struct_1.6e14-4.0e11-1e10_normal.dat).
        ``star=True`` also returns NSCool's per-zone profile (``parse_star_file``).
        ``keep`` copies the run directory there for inspection.
        """
        if (mass is None) == (profile is None):
            raise ValueError("give exactly one of mass, profile")
        eos_table = Path(eos_table) if eos_table else self.default_eos
        crust_table = Path(crust_table) if crust_table else self.default_crust
        with tempfile.TemporaryDirectory(dir=self.workroot) as d:
            d = Path(d)
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
            env = dict(os.environ, OMP_NUM_THREADS="1")
            r = subprocess.run([str(self.exe)], cwd=d, input="Cool.in\n",
                               capture_output=True, text=True, timeout=timeout,
                               env=env)
            (d / "stdout.txt").write_text(r.stdout + r.stderr)
            if keep is not None:
                shutil.copytree(d, keep, symlinks=True, dirs_exist_ok=True)
            if r.returncode != 0 or "Done" not in r.stdout or "PAUSE" in r.stdout.upper():
                raise NSCoolError(f"NSCool failed (rc={r.returncode}):\n"
                                  f"{(r.stdout + r.stderr)[-2000:]}")
            curve = parse_teff_file(m1 / "Teff.dat")
            if star:
                curve = curve._replace(star=parse_star_file(m1 / "Star.dat"))
            return curve
