"""Build the external pinned RNS source and export a polytropic reference pair."""
from __future__ import annotations

import argparse
from pathlib import Path

from nmbackend.rotating import build_rns, solve_polytrope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="cgca/rns source/rns.v2.0 at pinned commit")
    parser.add_argument("output", type=Path, help="fresh build/run directory")
    parser.add_argument("--energy-center", type=float, default=0.2)
    parser.add_argument("--index", type=float, default=1.0)
    parser.add_argument("--axis-ratio", type=float, default=0.8)
    parser.add_argument("--radial-points", type=int, default=129)
    parser.add_argument("--angular-points", type=int, default=65)
    args = parser.parse_args()
    exe = build_rns(args.source, args.output / "build", radial_points=args.radial_points,
                    angular_points=args.angular_points)
    _, star = solve_polytrope(exe, args.output / "run", energy_center=args.energy_center,
                             index=args.index, axis_ratio=args.axis_ratio)
    print(star.metadata)


if __name__ == "__main__":
    main()
