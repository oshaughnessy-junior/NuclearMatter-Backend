"""RNS exchange failures plus optional actual executable/physics smoke tests."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nmbackend.rotating.rns import build_rns, export_driver, load_background, solve_polytrope


def _fixture(path, *, reorder=False, omega=0.0, nan=False):
    rows = []
    for r in (0., 0.5, 1.):
        for mu in (0., 0.5, 1.):
            rows.append([r, mu, -0.1, -0.01, 0.045, 0., 0., 0.2, 0.02, 0.15])
    if reorder:
        rows[0], rows[1] = rows[1], rows[0]
    if nan:
        rows[1][2] = np.nan
    np.savetxt(path, rows, delimiter=",", comments="", header=(
        f"# r_ratio=1 r_e=1 Omega={omega}\n"
        "r,mu,rho,gama,alpha,omega,velocity_sq,energy,pressure,enthalpy"))


def test_exchange_conventions_and_corruption(tmp_path):
    path = tmp_path / "star.csv"
    _fixture(path)
    star = load_background(path)
    assert star.fields["rho"].shape == (3, 3)
    np.testing.assert_allclose(star.s, [0, 1/3, 1/2])
    for options in ({"reorder": True}, {"nan": True}, {"omega": 0.1}):
        _fixture(path, **options)
        with pytest.raises(ValueError):
            load_background(path)


def test_wrong_source_and_invalid_requests(tmp_path):
    (tmp_path / "main.c").write_text("unversioned source")
    with pytest.raises(ValueError, match="does not match"):
        export_driver(tmp_path, tmp_path / "export.c")
    for kwargs in ({"axis_ratio": 0.5}, {"energy_center": -1}, {"index": np.nan}):
        with pytest.raises(ValueError):
            solve_polytrope("absent", tmp_path / "run", **kwargs)
    assert not (tmp_path / "run").exists()


@pytest.fixture(scope="module")
def rns_exe(tmp_path_factory):
    source = os.environ.get("RNS_SOURCE")
    if not source:
        if os.environ.get("RNS_REQUIRED") == "1":
            pytest.fail("RNS_REQUIRED=1 but RNS_SOURCE is missing")
        pytest.skip("External pinned RNS source not configured")
    return build_rns(Path(source), tmp_path_factory.mktemp("rns") / "build")


@pytest.mark.parametrize("q", [1.0, 0.8])
def test_external_rotating_reference(rns_exe, tmp_path, q):
    spherical, rotating = solve_polytrope(rns_exe, tmp_path / "run", axis_ratio=q)
    # Pinned N=1 e_c=.2 baseline; this is smoke, not independent RNS accuracy.
    np.testing.assert_allclose(spherical.metadata["Mass"], 0.152, rtol=0.01)
    np.testing.assert_allclose(spherical.metadata["R_e"], 0.899, rtol=0.01)
    assert spherical.metadata["Omega"] == 0
    for name in ("rho", "gama", "alpha"):
        np.testing.assert_allclose(spherical.fields[name],
            np.broadcast_to(spherical.fields[name][:, :1], spherical.fields[name].shape),
            atol=1e-4)
    assert rotating.metadata["r_ratio"] == q
    assert rotating.metadata["Mass_b"] > rotating.metadata["Mass"]
    assert rotating.metadata["Omega"] <= rotating.metadata["Omega_K"]
    if q < 1:
        assert rotating.metadata["Omega"] > 0
        assert rotating.metadata["R_e"] > spherical.metadata["R_e"]
