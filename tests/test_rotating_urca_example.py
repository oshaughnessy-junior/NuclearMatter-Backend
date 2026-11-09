"""Adversarial cache/input contracts; analytic fixtures never claim equilibria."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('urca_example',
    Path(__file__).resolve().parents[1]/'examples'/'rotating_urca_accessibility.py')
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache(tmp_path, monkeypatch):
    path = tmp_path/'cache'; path.mkdir()
    for name in ('eos.dat','rns_rotating.csv','rns_spherical.csv'):
        (path/name).write_text('valid fixture '+name)
    eos_hash = digest(path/'eos.dat')
    record = dict(binary_sha256='binary',eos=dict(sha256=eos_hash),
        command=['rns','-q','tab','-e','1e15'],
        rotating=dict(csv_sha256=digest(path/'rns_rotating.csv'),r_ratio=.9),
        spherical=dict(csv_sha256=digest(path/'rns_spherical.csv')))
    (path/'run.json').write_text(json.dumps(record))
    star = SimpleNamespace(metadata=dict(r_ratio=.9,e_center=1.))
    monkeypatch.setattr(example,'load_tabulated_background',lambda *a:star)
    return path,dict(binary_hash='binary',eos_hash=eos_hash,q=.9,energy=1e15),record,star


def test_cached_csv_integrity_and_native_parameters(tmp_path,monkeypatch):
    path,kwargs,record,star = cache(tmp_path,monkeypatch)
    assert example._cached_background(path,**kwargs) is star
    (path/'rns_rotating.csv').write_text('Mass_b changed without updating provenance')
    with pytest.raises(ValueError,match='checksum mismatch'):
        example._cached_background(path,**kwargs)
    record['rotating']['csv_sha256']=digest(path/'rns_rotating.csv')
    (path/'run.json').write_text(json.dumps(record))
    with pytest.raises(ValueError,match='requested parameter mismatch'):
        example._cached_background(path,**{**kwargs,'q':.8})
    star.metadata['e_center']=1.1
    with pytest.raises(ValueError,match='native parameter mismatch'):
        example._cached_background(path,**kwargs)


def test_incomplete_cache_fails_closed(tmp_path):
    path=tmp_path/'partial'; path.mkdir()
    with pytest.raises(ValueError,match='Invalid/incomplete cache.*quarantine'):
        example._cached_background(path,binary_hash='x',eos_hash='x',q=.9,energy=1e15)


def args_and_npz(tmp_path, **changes):
    n=np.array([.01,.02,.04,.08])
    values=dict(table_n=n,table_eps=n,table_p=n/10,table_h=n,
                table_cs2=n*0+0.1,table_eps_mevfm3=n*1000,table_p_mevfm3=n*100,
                comp_n=n,comp_x_p=n*0+.1,comp_x_e=n*0+.08,comp_x_mu=n*0+.02)
    values.update(changes)
    source=tmp_path/'eos.npz'; np.savez(source,**values)
    executable=tmp_path/'binary'; executable.write_bytes(b'not executed')
    return example.parser().parse_args(['--eos-npz',str(source),'--executable',str(executable),
        '--output',str(tmp_path/'out'),'--initial-nc','.03','--q-grid','.9'])


@pytest.mark.parametrize('changes',[
    dict(comp_n=np.array([0,.02,.04,.08])),
    dict(comp_x_e=np.array([.2,.2,.2,.2])),
])
def test_bad_composition_rejected_before_solve(tmp_path,changes):
    with pytest.raises(ValueError,match='composition|Composition'):
        example.run(args_and_npz(tmp_path,**changes))


@pytest.mark.parametrize('key,value',[
    ('q_grid',[]),('q_grid',[np.nan]),('timeout',0),('timeout',np.inf),
    ('nc_bracket',[.8,np.inf]),('initial_nc',[]),('onset_margin_atol',0),
])
def test_invalid_runtime_arguments_rejected(tmp_path,key,value):
    args=args_and_npz(tmp_path)
    setattr(args,key,value)
    with pytest.raises(ValueError):
        example.run(args)


def test_residual_acceptance_is_not_just_a_report():
    example._accept_residual(3e-11,1e-6,'volume')
    for value in (-.5,np.nan,np.inf):
        with pytest.raises(ValueError,match='residual'):
            example._accept_residual(value,1e-6,'volume')


def test_absent_muons_cannot_be_counted_accessible():
    np.testing.assert_array_equal(
        example._channel_accessibility(np.array([1.,1.,-1.]),np.array([0.,.1,.1])),
        [False,True,False])


@pytest.mark.parametrize('left_margin,left_present,right_margin,right_present,expected',[
    (-1.,False,1.,True,True),  # Channel appears inside the coarse interval.
    (-1.,True,1.,True,True),
    (-1.,False,1.,False,False),
    (1.,False,2.,True,False),
    (1.,False,2.,False,False),
    (0.,True,1.,True,True),
])
def test_onset_bracket_allows_channel_appearance(
        left_margin,left_present,right_margin,right_present,expected):
    left=dict(D_mu_fm_inv=left_margin,central_mu_present=left_present)
    right=dict(D_mu_fm_inv=right_margin,central_mu_present=right_present)
    assert example._onset_bracket(left,right,'mu') is expected
