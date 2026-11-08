"""Rotating-mode numerical foundation; full GR action assembly is not supplied."""
from .pencil import Pencil,Spectrum,ModeError,solve,select_simple,directional_derivative
from .conventions import frequencies,conjugate_partner,stationary_patterns,exterior_contour
from .tracking import match_branches
from .rmode import toroidal_rmode

__all__=['Pencil','Spectrum','ModeError','solve','select_simple','directional_derivative',
         'frequencies','conjugate_partner','stationary_patterns','exterior_contour',
         'match_branches','toroidal_rmode']
from .background import ModeBackground,rns_polytropic_background
__all__ += ['ModeBackground','rns_polytropic_background']
from .maclaurin import bar_mode,neutrality_thresholds
__all__ += ['bar_mode','neutrality_thresholds']
