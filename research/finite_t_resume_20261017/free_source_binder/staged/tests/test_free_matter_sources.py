"""Fault-injection guards using saved ACTUAL channel rows, not new physics claims."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import pytest
from nmbackend.reference import free_matter as F
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _free_matter_sources as S
from nmbackend.reference._weak_quadrature import positive, signed

DATA = json.loads((Path(__file__).parent/'data/free_matter_reference_internal.json').read_text())


def records():
    out=[]
    for row in DATA['initial_channel_records']:
        out.append(W.MomentResult(row['channel'], W.FreeElectronBath(**row['bath']),
            W.Radiation(**row['radiation']), W.Controls(**row['controls']), W.Current(),
            tuple(W.PositiveQuantity(**q) for q in row['forward']),
            tuple(W.PositiveQuantity(**q) for q in row['backward']),
            tuple(W.SignedQuantity(**q) for q in row['net']),None,
            row['max_event_kms_error'],tuple(tuple(x) for x in row['diagnostics']),
            tuple(tuple(x) for x in row['particle_stoichiometry']),
            row['energy_ledger_scaled'],row['direct_energy_ledger_scaled']))
    return tuple(out)


@pytest.fixture
def state():
    return F.state_density(2.,1e-7,.3,photons='absent')


def synthetic_assembly(state, rows):
    with patch.object(W,'channel_moments',side_effect=list(rows)):
        return F.source_moments(state,radiation=W.Radiation.empty(),controls=F.SourceControls.fine())


def test_saved_records_interface_and_rhs(state):
    source=synthetic_assembly(state,records())
    rhs=F.instantaneous_rhs(state,source,baryon_density_dot_fm3_s=2.5e-8)
    assert source.complete_channels == ('EC','PC','NEUTRON_DECAY')
    assert source.material_energy_source.value == -source.neutrino_energy_escape.value
    assert rhs.Ye_dot_s == pytest.approx(DATA['initial_R']/1e-7,rel=2e-9)
    assert abs(source.ledgers.material_energy_scaled)<2e-8
    assert rhs.source_identity==source.identity


@pytest.mark.parametrize('fault', ['wrong_channel','wrong_bath','wrong_control','absent_forward','bad_meta','bad_stoich','wrong_net'])
def test_malformed_result_rejects(state,fault):
    rows=list(records());r=rows[0]
    if fault=='wrong_channel':r=replace(r,channel='PC')
    if fault=='wrong_bath':r=replace(r,bath=replace(r.bath,mu_e=r.bath.mu_e+.01))
    if fault=='wrong_control':r=replace(r,controls=replace(r.controls,tail=70.))
    if fault=='absent_forward':r=replace(r,forward=(positive(),)+r.forward[1:])
    if fault=='bad_meta':r=replace(r,diagnostics=r.diagnostics+(('bad',float('nan')),))
    if fault=='bad_stoich':r=replace(r,particle_stoichiometry=(('neutron',True),)+r.particle_stoichiometry[1:])
    if fault=='wrong_net':r=replace(r,net=(signed(r.net[0].log_abs+1,r.net[0].sign),)+r.net[1:])
    rows[0]=r
    with pytest.raises((ValueError,ArithmeticError)):
        synthetic_assembly(state,rows)


def test_timeout_never_returns_partial(state):
    calls=[]
    def timed(channel,*args):
        calls.append(channel)
        if channel=='PC':raise TimeoutError('injected bounded incomplete quadrature')
        return records()[0]
    with patch.object(W,'channel_moments',side_effect=timed):
        with pytest.raises(TimeoutError):
            F.source_moments(state,radiation=W.Radiation.empty(),controls=F.SourceControls.fine())
    assert calls==['EC','PC']


def test_source_cannot_be_relabelled_or_reconstructed(state):
    source=synthetic_assembly(state,records())
    for wrong in [replace(source),replace(source,state_identity='same affinity is not state identity')]:
        with pytest.raises(ValueError):
            F.instantaneous_rhs(state,wrong,baryon_density_dot_fm3_s=0.)
    with pytest.raises(ValueError):
        F.instantaneous_rhs(replace(state,Ye=.31),source,baryon_density_dot_fm3_s=0.)
    with pytest.raises(ValueError):
        F.instantaneous_rhs(state,source,baryon_density_dot_fm3_s=True)


def test_positive_output_underflow_retained_but_rhs_rejects(state):
    rows=[]
    for r in records():
        rows.append(replace(r,forward=tuple(positive(q.log-1000) for q in r.forward),
            net=tuple(signed(q.log_abs-1000,q.sign) for q in r.net)))
    source=synthetic_assembly(state,rows)
    assert source.neutrino_energy_escape.value is None
    with pytest.raises(S.UnrepresentedSourceError) as error:
        F.instantaneous_rhs(state,source,baryon_density_dot_fm3_s=0.)
    assert error.value.source is source


def test_state_and_radiation_before_call(state):
    with patch.object(W,'channel_moments',side_effect=AssertionError('must not call')):
        with pytest.raises(ValueError):
            F.source_moments(replace(state,h_e=state.h_e+1),radiation=W.Radiation.empty(),controls=F.SourceControls.fine())
        with pytest.raises(ValueError):
            F.source_moments(state,radiation=W.Radiation.fd(2.1,0.,0.),controls=F.SourceControls.fine())


def test_stale_prerequisite_pin_before_source_call(state):
    original=Path.read_bytes
    def changed(path):
        content=original(path)
        return content+b'\n# synthetic stale source bytes' if path.name=='_weak_two_body.py' else content
    with patch.object(Path,'read_bytes',changed), patch.object(W,'channel_moments',side_effect=AssertionError('must not call')):
        with pytest.raises(RuntimeError,match='Prerequisite source changed'):
            F.source_moments(state,radiation=W.Radiation.empty(),controls=F.SourceControls.fine())


def test_stale_prerequisite_pin_before_rhs(state):
    source=synthetic_assembly(state,records())
    original=Path.read_bytes
    def changed(path):
        content=original(path)
        return content+b'\n# synthetic stale source bytes' if path.name=='_weak_decay.py' else content
    with patch.object(Path,'read_bytes',changed):
        with pytest.raises(RuntimeError,match='Prerequisite source changed'):
            F.instantaneous_rhs(state,source,baryon_density_dot_fm3_s=0.)
