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


def _external_rns_source():
    import os
    from pathlib import Path
    source = os.environ.get('RNS_SOURCE')
    if source:
        return Path(source)
    if os.environ.get('RNS_REQUIRED') == '1':
        pytest.fail('RNS_REQUIRED=1 needs RNS_SOURCE')
    pytest.skip('Set RNS_SOURCE for pinned external native regression')


def test_pinned_headers_rejected_before_build(tmp_path):
    import shutil
    from nmbackend.rotating.tabulated import build_tabulated_rns, RNS_SOURCE_SHA256
    source = _external_rns_source()
    altered = tmp_path/'source'
    altered.mkdir()
    for name in RNS_SOURCE_SHA256:
        shutil.copyfile(source/name, altered/name)
    with (altered/'consts.h').open('a') as stream:
        stream.write('\n/* altered constants header */\n')
    with pytest.raises(ValueError, match='consts.h'):
        build_tabulated_rns(altered,tmp_path/'build')
    assert not (tmp_path/'build').exists()


def test_native_large_table_and_failed_spin_are_distinct(tmp_path):
    import json
    import subprocess
    from nmbackend.rotating.tabulated import (
        build_tabulated_rns, solve_tabulated, SPIN_MAX_ITERATIONS,
        _guard_equilibrium,
    )
    source = _external_rns_source()
    # Analytic cold gamma=2 barotrope: eps=m*n+K*n^2, p=K*n^2,
    # h=ln[(eps+p)/(m*n)]. More than 200 points exercises actual native arrays.
    n = np.geomspace(1e-4,1.,500)
    eos = SimpleNamespace(n=n,eps_mevfm3=939.*n+100.*n*n,
                         p_mevfm3=100.*n*n,h=np.log1p(200.*n/939.))
    build = tmp_path/'build'
    exe = build_tabulated_rns(source,build,radial_points=65,angular_points=33,
                              accuracy=1.e-7)
    ec=(939.*.5+100.*.5**2)*MEVFM3_TO_ERGCM3/C_CGS**2
    spherical,rotating = solve_tabulated(exe,tmp_path/'success',eos,
                                       energy_center_g_cm3=ec,axis_ratio=.9)
    assert spherical.metadata['Mass_msun']>0
    assert rotating.metadata['Mass_msun']>0
    assert rotating.metadata['r_ratio']==pytest.approx(.9)
    manifest=json.loads((build/'build.json').read_text())
    assert manifest['accuracy']==pytest.approx(1.e-7)
    assert manifest['spin_max_iterations']==SPIN_MAX_ITERATIONS
    assert manifest['generated_equilibrium_sha256']
    # Recompile only a private generated copy with an intentionally impossible
    # one-iteration cap. This exercises native failure propagation on the same
    # physical EOS; it does not tamper with the pinned external checkout.
    eq=build/'tabulated_equil.c'
    eq.write_text(_guard_equilibrium((source/'equil.c').read_text(),max_iterations=1))
    subprocess.run(manifest['command'],check=True,capture_output=True,text=True)
    failed=tmp_path/'failed'
    with pytest.raises(subprocess.CalledProcessError):
        solve_tabulated(exe,failed,eos,energy_center_g_cm3=ec,axis_ratio=.9)
    assert 'NMB_RNS_SPIN_ITERATION_LIMIT' in (failed/'run.log').read_text()
    assert not (failed/'run.json').exists()
    # The original pinned preamble already relaxes the spherical initial
    # model with spin before the export tail. Its failure must stop all exports.
    assert not (failed/'rns_spherical.csv').exists()
    assert not (failed/'rns_rotating.csv').exists()


def test_export_rejects_bad_shape_and_conversion_before_writing(tmp_path):
    bad = table()
    bad.n = bad.n[:,None]
    with pytest.raises(ValueError):
        export_eos(bad,tmp_path/'shape.dat')
    assert not (tmp_path/'shape.dat').exists()
    bad = table()
    bad.n = np.arange(1.,5.)*1.e270
    bad.eps_mevfm3 = np.arange(1.,5.)*1.e300
    with np.errstate(over='ignore'):
        with pytest.raises(ValueError,match='conversion'):
            export_eos(bad,tmp_path/'overflow.dat')
    assert not (tmp_path/'overflow.dat').exists()


@pytest.mark.parametrize('accuracy',[0.,-1.e-6,1.e-4,np.nan,np.inf])
def test_build_rejects_invalid_or_looser_accuracy(tmp_path,accuracy):
    from nmbackend.rotating.tabulated import build_tabulated_rns
    with pytest.raises(ValueError,match='accuracy'):
        build_tabulated_rns(tmp_path/'absent',tmp_path/'build',accuracy=accuracy)
    assert not (tmp_path/'build').exists()
