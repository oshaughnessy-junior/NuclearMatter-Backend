"""NSCool star EOS file writer (nmbackend.nscool_io)."""
import math

import jax
import numpy as np
import pytest

from nmbackend.eos import Composition, RMFTEOS, MetaModelEOS
from nmbackend.eos.coolcomp import cooling_composition
from nmbackend.eos.crust import _DYN_CM2_TO_MEVFM3, _G_CM3_TO_MEVFM3
from nmbackend.durca import durca_onset
from nmbackend.nscool_io import read_star_eos, write_tables

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def gm1(tmp_path_factory):
    m = RMFTEOS()
    tab, cc = m.eos_table(), cooling_composition(m)
    path = str(tmp_path_factory.mktemp("nscool") / "GM1_Cat.dat")
    return tab, cc, write_tables(tab, cc, path=path), read_star_eos(path)


def test_layout(gm1):
    """Line 1 is "6 Imax Icore" (TOV.f skips six lines), 17 core columns,
    rows densest first, NSCool's per-row charge and baryon checks pass."""
    tab, _, info, f = gm1
    assert (f["itext"], f["imax"], f["icore"]) == (6, info["imax"],
                                                   info["icore"])
    assert f["imax"] == len(np.asarray(tab.n)) and f["imax"] <= 1000
    core, crust = f["core"], f["crust"]
    assert core.shape == (info["icore"], 17)
    allr = np.vstack([core[:, :3], crust])
    assert np.all(np.diff(allr, axis=0) < 0.0)
    ye, ymu, yn, yp = core[:, 3], core[:, 4], core[:, 5], core[:, 6]
    assert np.max(abs(yp - ye - ymu)) < 1e-4
    assert np.max(abs(yn + yp - 1.0)) < 1e-5
    assert np.all(core[:, 7:11] == 0.0) and np.all(core[:, 13:] == 0.0)
    assert crust[-1, 0] < 1e8                       # reaches the envelope
    assert info["n_core"] == pytest.approx(core[-1, 2], rel=1e-5)


def test_rows_are_the_eos_table(gm1):
    """TOV.f sees the nmbackend table itself (6 significant digits)."""
    tab, _, _, f = gm1
    rows = np.vstack([f["core"][:, :3], f["crust"]])[::-1]
    np.testing.assert_allclose(rows[:, 2], np.asarray(tab.n), rtol=1e-5)
    np.testing.assert_allclose(rows[:, 0] * _G_CM3_TO_MEVFM3,
                               np.asarray(tab.eps_mevfm3), rtol=1e-5)
    np.testing.assert_allclose(rows[:, 1] * _DYN_CM2_TO_MEVFM3,
                               np.asarray(tab.p_mevfm3), rtol=1e-5)


def test_file_composition_gives_durca_onset(gm1):
    """Onset from the file matches durca.py on the model composition and the
    GM1 literature value n_DU = 0.28 fm^-3 (Raduta+2018, as in
    test_durca.py)."""
    _, cc, _, f = gm1
    c = f["core"][::-1]
    on = durca_onset(Composition(n=c[:, 2], x_p=c[:, 6], x_e=c[:, 3],
                                 x_mu=c[:, 4]))
    on0 = durca_onset(cc)
    assert abs(on.n_DU - on0.n_DU) < 1e-4
    assert abs(on.x_p_DU - on0.x_p_DU) < 1e-4
    assert abs(on.n_DU - 0.28) < 0.01
    n = np.asarray(cc.n)
    np.testing.assert_allclose(c[:, 12], np.interp(c[:, 2], n,
                                                   np.asarray(cc.mst_n)),
                               rtol=1e-5)


def test_n_max_truncates(gm1, tmp_path):
    """Rows stop at the first node at or above n_max; the file stays valid."""
    tab, cc, full, _ = gm1
    info = write_tables(tab, cc, path=str(tmp_path / "t.dat"), n_max=0.5)
    f = read_star_eos(info["path"])
    n = np.asarray(tab.n)
    top = n[np.searchsorted(n, 0.5)]
    assert f["core"][0, 2] == pytest.approx(top, rel=1e-5)
    assert info["imax"] == full["imax"] - int(np.sum(n > top))
    assert info["n_core"] == full["n_core"]


def test_rejects_bad_input(tmp_path):
    m = MetaModelEOS(muons=True, crust="polytrope")
    with pytest.raises(ValueError, match="SLy crust"):
        write_tables(m.eos_table(), cooling_composition(m),
                     path=str(tmp_path / "x.dat"))
    m = RMFTEOS()
    tab = m.eos_table()
    nan = tab._replace(p_mevfm3=tab.p_mevfm3 * math.nan)
    with pytest.raises(ValueError, match="non-finite"):
        write_tables(nan, cooling_composition(m), path=str(tmp_path / "y.dat"))
