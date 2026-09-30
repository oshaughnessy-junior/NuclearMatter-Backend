"""A sampler loop over many EOS draws must not compile per draw.

Each cached XLA executable holds ~100 memory maps, so per-draw compiles end in
an abort at vm.max_map_count (see test_tov.py::test_repeat_solve_does_not_recompile).
Each test warms up on one draw, then requires zero backend compiles on a second
draw with different parameters and a fresh model instance.
"""
import jax
import jax.monitoring
import numpy as np
import pytest

from nmbackend.eos import MetaModelEOS, SkyrmeEOS, RMFTEOS, DDRMFEOS, HybridEOS
from nmbackend.eos.skyrme import skyrme_from_saturation
from nmbackend.eos.rmft import couplings_from_saturation
from nmbackend.eos.ddrmf import couplings_from_saturation_dd
from nmbackend.priors import draw_bridge_eos
from nmbackend.tov import build_family
from nmbackend.tov.family import default_hc_grid
from nmbackend.tov.solver import _HAVE_DIFFRAX

_COMPILES = []
jax.monitoring.register_event_duration_secs_listener(
    lambda ev, *a, **k: _COMPILES.append(ev)
    if ev == "/jax/core/compile/backend_compile_duration" else None)


def _compiles(f):
    n0 = len(_COMPILES)
    jax.block_until_ready(f())
    return len(_COMPILES) - n0


@pytest.mark.parametrize("model, draws", [
    (MetaModelEOS, [{"L_sym": 60.0}, {"L_sym": 50.0, "K_sat": 240.0}]),
    (SkyrmeEOS, [skyrme_from_saturation(L=46.0),
                 skyrme_from_saturation(L=55.0, K=240.0)]),
    (RMFTEOS, [couplings_from_saturation(L=80.0),
               couplings_from_saturation(L=70.0, K=260.0)]),
    (DDRMFEOS, [couplings_from_saturation_dd(L=55.0),
                couplings_from_saturation_dd(L=50.0, K=250.0)]),
], ids=["metamodel", "skyrme", "rmft", "ddrmf"])
def test_eos_table_compiles_once(model, draws):
    _compiles(lambda: model().eos_table(draws[0]))
    t1 = model().eos_table(draws[1])
    assert _compiles(lambda: model().eos_table(draws[1])) == 0
    assert np.all(np.isfinite(np.asarray(t1.p)))


@pytest.mark.skipif(not _HAVE_DIFFRAX, reason="diffrax required")
def test_bridge_draw_and_family_compile_once():
    def draw(seed):             # raw node counts differ: 762 and 766
        t = draw_bridge_eos(np.random.default_rng(seed), corr_frac=0.3)["table"]
        return build_family(t, default_hc_grid(t, 20))
    _compiles(lambda: draw(1))
    assert _compiles(lambda: draw(2)) == 0


@pytest.mark.skipif(not _HAVE_DIFFRAX, reason="diffrax required")
def test_hybrid_family_compiles_once():
    def fam(p_trans, L):        # raw outer-segment counts differ: 432 and 435
        t = HybridEOS(had_params={"L_sym": L}, p_trans_mev=p_trans).segments[1]
        return build_family(t, default_hc_grid(t, 20))
    _compiles(lambda: fam(80.0, 60.0))
    assert _compiles(lambda: fam(90.0, 55.0)) == 0
