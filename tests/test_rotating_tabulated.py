"""Native RNS EOS columns and independent dimensional-rotation checks."""
from types import SimpleNamespace

import numpy as np
import pytest

from nmbackend.rotating.tabulated import (
    C_CGS, LENGTH_CM, MEVFM3_TO_ERGCM3, _native_log_interp,
    export_eos, load_tabulated_background,
)


def table():
    return SimpleNamespace(n=np.array([.01,.02,.04,.08]),
        eps_mevfm3=np.array([10.,20.,40.,80.]),
        p_mevfm3=np.array([1.,2.,4.,8.]), h=np.array([.01,.02,.04,.08]))


def test_native_columns_and_units(tmp_path):
    path = tmp_path/'eos.dat'
    export_eos(table(),path)
    assert path.read_text().splitlines()[0] == '4'
    a = np.loadtxt(path,skiprows=1)
    np.testing.assert_allclose(a[:,0]*C_CGS**2,table().eps_mevfm3*MEVFM3_TO_ERGCM3)
    np.testing.assert_allclose(a[:,1],table().p_mevfm3*MEVFM3_TO_ERGCM3)
    np.testing.assert_allclose(a[:,2]/C_CGS**2,table().h)
    np.testing.assert_allclose(a[:,3]/1e39,table().n)
    bad = table(); bad.h[0] = 0
    with pytest.raises(ValueError):
        export_eos(bad,path)


def test_native_four_point_interpolation():
    x = np.linspace(-3.,2.,20)
    q = np.linspace(-2.9,1.9,43)
    np.testing.assert_allclose(_native_log_interp(q,x,2*x**3-x+1),2*q**3-q+1,atol=1e-13)


def test_physical_omega_is_normalized_before_velocity_check(tmp_path):
    eos = tmp_path/'eos.dat'; export_eos(table(),eos)
    path = tmp_path/'star.csv'
    omega = 1000.
    internal = omega*LENGTH_CM/C_CGS
    rows = []
    for r in (0.,.25,.5):
        for mu in (0.,.5,1.):
            v2=(internal*r*np.sqrt(1-mu**2))**2
            rows.append([r,mu,0,0,0,0,v2,20*MEVFM3_TO_ERGCM3/C_CGS**2/1e15,.001,.02])
    np.savetxt(path,rows,delimiter=',',comments='',header=(
        '# r_ratio=.8 r_e=1 Omega=1000 Omega_K=2000 Mass=2e33 Mass_b=2.2e33 R_e=1e6 J=1e49\n'
        'r,mu,rho,gama,alpha,omega,velocity_sq,energy,pressure,enthalpy'))
    star = load_tabulated_background(path,eos)
    assert star.metadata['Omega_rad_s'] == 1000
    assert star.metadata['Omega'] == pytest.approx(internal)
    assert star.metadata['R_e_km'] == 10
    np.testing.assert_allclose(star.fields['n_fm3'],.02)
    assert star.metadata['velocity_max_error'] < 1e-12
    assert 'polytropic' not in star.metadata['units']
