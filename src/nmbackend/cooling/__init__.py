"""Neutron-star cooling: a driver for D. Page's NSCool (built separately).

See README.md in this directory for the build and the input formats.
"""
from .runner import (CoolingCurve, GapModel, NSCool, NSCoolError,
                     default_home, parse_star_file, parse_teff_file)

__all__ = ["CoolingCurve", "GapModel", "NSCool", "NSCoolError",
           "default_home", "parse_star_file", "parse_teff_file"]
