"""solve_reference must not report r_max as the stellar surface."""
import math

import numpy as np

from nmbackend.priors import draw_bridge_eos
from nmbackend.reference.scipy_tov import solve_reference


def test_no_surface_within_r_max_gives_nan():
    # This star has R ~ 800-2800 km; the old code returned R = 60.0 exactly.
    t = draw_bridge_eos(np.random.default_rng(1), corr_frac=0.3)["table"]
    ref = solve_reference(t, 0.02)
    assert not (math.isfinite(ref.R_km) and ref.R_km == 60.0)
    assert math.isnan(ref.R_km) and math.isnan(ref.M_msun)
    assert math.isnan(ref.Lambda)


def test_large_r_max_reaches_surface():
    t = draw_bridge_eos(np.random.default_rng(1), corr_frac=0.3)["table"]
    ref = solve_reference(t, 0.02, r_max=1e5)
    assert 60.0 < ref.R_km < 1e5
    assert math.isfinite(ref.M_msun) and ref.M_msun > 0
