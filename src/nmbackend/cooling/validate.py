"""Validation figure and G0 timing for the NSCool driver.

    python -m nmbackend.cooling.validate --fig nscool_validation.png
    python -m nmbackend.cooling.validate --timing 10

Panel (a) compares the driver with NSCool's bundled Model_1 output
(Teff_Try_KEEP.dat: APR, 1.4 Msun, SFB / gap "a" / T73, iron envelope).
Panel (b) shows the mass dependence on the same EOS; panel (c) shows p1S0
scalings (Zhou+2026 Eq. 10) quenching dUrca at 2.1 Msun.
"""
from __future__ import annotations

import argparse
import platform
import tempfile
import time

import numpy as np

from .runner import GapModel, NSCool, parse_teff_file

BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#6b6a64"
RAMP = ["#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"]


def make_figure(path: str, nscool: NSCool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = parse_teff_file(nscool.home / "Model_1" / "Teff_Try_KEEP.dat")
    prof = nscool.run(profile=nscool.home / "TOV/Profile/Prof_APR_Cat_1.4.dat")
    tov = nscool.run(mass=1.4)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log")
        a.set_xlim(1e-1, 3e6); a.set_ylim(8e4, 4e6)
        a.set_xlabel("age [yr]")
        a.grid(True, which="major", color="#e4e3dc", lw=0.6)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    ax[0].set_ylabel(r"$T_{\rm eff}^\infty$ [K]")

    a = ax[0]
    a.plot(ref.t_yr[::4], ref.Teff_inf[::4], "o", ms=5, mfc="none", mec=INK,
           label="NSCool bundled output")
    a.plot(prof.t_yr, prof.Teff_inf, color=BLUE, lw=2, label="driver, bundled profile")
    a.plot(tov.t_yr, tov.Teff_inf, color=ORANGE, lw=2, ls="--",
           label="driver, TOV at 1.4")
    dev = np.max(np.abs(prof.Teff_inf / ref.Teff_inf - 1))
    a.set_title(f"(a) APR 1.4 Msun: max |dT/T| = {dev:.1e}", fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower left")

    a = ax[1]
    for m, c in zip((1.0, 1.4, 1.8, 2.0, 2.1), RAMP):
        cc = nscool.run(mass=m)
        a.plot(cc.t_yr, cc.Teff_inf, color=c, lw=2, label=f"{m:.1f}")
    a.set_title("(b) mass [Msun], Model_1 gaps", fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower left")

    a = ax[2]
    # CCDK p1S0 (SFP1S0=7), n3P2 off as in Zhou+2026.  Unscaled, the gap
    # closes at n = 0.58 fm^-3, below the dUrca core; s_x >= 2 covers it.
    for (sx, sy), c, ls in zip(((1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (2.0, 0.2)),
                               (INK, BLUE, ORANGE, MUTED), ("-", "-", "-", "--")):
        cc = nscool.run(mass=2.1, gaps=GapModel(p1s0=7.0, n3p2=0.0,
                                                sx_p1s0=sx, sy_p1s0=sy))
        a.plot(cc.t_yr, cc.Teff_inf, color=c, lw=2, ls=ls,
               label=fr"$s_x={sx}$, $s_y={sy}$")
    a.set_title("(c) 2.1 Msun (dUrca), CCDK p1S0, no n3P2", fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8, loc="lower left")

    fig.tight_layout()
    fig.savefig(path, dpi=150)


def time_curves(nscool: NSCool, repeats: int) -> None:
    """Wall time per curve, one process at a time (NSCool is single-threaded)."""
    print(f"host {platform.node()}  cpu {platform.processor() or platform.machine()}")
    cases = [("1.4, bundled profile", dict(profile=nscool.home / "TOV/Profile/Prof_APR_Cat_1.4.dat")),
             ("1.4, TOV + cool", dict(mass=1.4)),
             ("2.0, TOV + cool (dUrca)", dict(mass=2.0)),
             ("1.4, eta=1e-9", dict(mass=1.4, eta=1e-9))]
    for label, kw in cases:
        w, n = [], 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            c = nscool.run(**kw)
            w.append(time.perf_counter() - t0)
            n = len(c.t_yr)
        w = np.array(w)
        print(f"{label:28s} steps {n:4d}  wall median {np.median(w):.2f} s"
              f"  min {w.min():.2f}  max {w.max():.2f}")
    w = []
    with tempfile.TemporaryDirectory(dir=nscool.workroot) as d:
        for _ in range(repeats):
            t0 = time.perf_counter()
            nscool.make_profile(nscool.default_eos, 1.4, f"{d}/prof.dat")
            w.append(time.perf_counter() - t0)
    print(f"{'TOV alone':28s}             wall median {np.median(w):.2f} s")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--fig")
    p.add_argument("--timing", type=int, default=0, metavar="REPEATS")
    p.add_argument("--workroot", help="parent of per-run directories")
    a = p.parse_args(argv)
    n = NSCool(workroot=a.workroot)
    if a.fig:
        make_figure(a.fig, n)
    if a.timing:
        time_curves(n, a.timing)


if __name__ == "__main__":
    main()
