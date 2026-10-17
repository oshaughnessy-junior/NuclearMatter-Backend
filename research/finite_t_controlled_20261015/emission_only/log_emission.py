"""Log-domain emission moments for the complete NAMED antineutrino inventory.

No true-absorption or transfer positivity requirement. Common kinetic mass,
constant physical-Q offsets and FD density roots match existing references.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys,math,time
import numpy as np
from scipy.special import logsumexp
from numpy.polynomial.legendre import leggauss

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
for sub in ['nr_baryon_eos','capture_endpoint','quantum_slab','crossed_opacity']:
    sys.path.insert(0,str(ROOT/sub))
import free_nucleons as eos
import guarded_capture
import crossed_decay
from allowed_capture import OpticalState
from free_density_response import PairState
import static_capture as static

DOS=1/(2*math.pi**2*static.HBARC**3)
LOG_MIN=math.log(float(np.nextafter(0.,1.)))
LOG_MAX=math.log(float(np.finfo(float).max))
CHANNELS=('positron_capture','neutron_decay')

@dataclass(frozen=True)
class Bath:
    neutron:eos.Species
    proton:eos.Species
    mu_e:float
    def __post_init__(self):
        if not math.isfinite(self.mu_e):raise ValueError('Finite prescribed total electron chemical potential required')
        if self.neutron.T!=self.proton.T or self.neutron.mass!=self.proton.mass:
            raise ValueError('One common T and kinetic mass required')
    @property
    def T(self):return self.neutron.T
    def pair(self):
        n,p=self.neutron,self.proton
        return PairState(n.T,n.mass,n.offset,p.offset,n.mu,p.mu)

def make_bath(T,nn,np_,mass,mu_e):
    """FD-density-matched nucleons; prescribed electrons, no neutrality imposed."""
    return Bath(eos.state_density(T,nn,static.Q,mass),eos.state_density(T,np_,0.,mass),float(mu_e))

def represented(log_value):
    """No floor: unavailable ordinary value is None, logarithm is retained."""
    if not math.isfinite(log_value):raise ArithmeticError('Non-finite logarithmic emission result')
    try:
        value=math.exp(log_value)
    except OverflowError:
        return dict(log_value=float(log_value),representable=False,value=None,reason='overflow',subnormal=False)
    if value==0.:
        return dict(log_value=float(log_value),representable=False,value=None,reason='underflow',subnormal=False)
    return dict(log_value=float(log_value),representable=True,value=value,reason=None,subnormal=bool(value<np.finfo(float).tiny))

def log_spectra(bath,Enu,*,energy_order=24,angle_order=24,tail=40.):
    """Both named log j channels, including formally tiny finite-T gap tails."""
    guarded_capture.validate_controls(Enu,energy_order,angle_order,tail)
    pair=bath.pair()
    pc=guarded_capture.log_coefficients(OpticalState(pair.reverse(),-bath.mu_e),Enu,energy_order,angle_order,tail)
    nd=crossed_decay.log_coefficients(crossed_decay.DecayState(pair,bath.mu_e),Enu,energy_order,angle_order,tail)
    vals=dict(positron_capture=float(pc[0]),neutron_decay=float(nd[0]))
    if not all(math.isfinite(v)for v in vals.values()):raise ArithmeticError('Unexpected non-finite finite-T channel log j')
    vals['combined']=float(logsumexp(list(vals.values())))
    return vals

def default_edges(bath,cutoff=30.):
    if not math.isfinite(cutoff)or cutoff<=static.Q+static.ME:raise ValueError('Cutoff must extend beyond both static thresholds')
    nd,pc=static.Q-static.ME,static.Q+static.ME
    edges={0.,.1,.4,nd,1.1,pc,2.,3.,4.,6.,9.,12.,18.,cutoff}
    for threshold in [nd,pc]:
        width=threshold*math.sqrt(bath.T/bath.neutron.mass)
        edges.update(threshold+k*width for k in [-24,-8,-2,2,8,24])
    return sorted(x for x in edges if 0<=x<=cutoff)

def log_moments(bath,*,outer_order=12,internal_order=24,angle_order=24,tail=40.,edges=None,max_seconds=None):
    guarded_capture.validate_controls(1.,internal_order,angle_order,tail)
    if isinstance(outer_order,bool)or not isinstance(outer_order,int)or outer_order<4:raise ValueError('Integer outer order>=4 required')
    edges=default_edges(bath)if edges is None else list(edges)
    if len(edges)<2 or edges[0]!=0 or not all(math.isfinite(v)for v in edges)or any(hi<=lo for lo,hi in zip(edges[:-1],edges[1:])):
        raise ValueError('Finite increasing energy edges starting at zero required')
    if max_seconds is not None and (not math.isfinite(max_seconds)or max_seconds<=0):raise ValueError('Positive finite runtime bound required')
    start=time.monotonic()
    xs,ws=leggauss(outer_order);panels=[]
    channel_panels={c:[[],[]]for c in CHANNELS}
    nd_threshold=static.Q-static.ME
    for lo,hi in zip(edges[:-1],edges[1:]):
        # Exact variable map, never a physical support restriction. It resolves
        # either side of the static square-root endpoints in the heavy limit.
        u=.5*(xs+1)*math.sqrt(hi-lo)
        energies=hi-u*u if hi<=nd_threshold else lo+u*u
        weights=.5*math.sqrt(hi-lo)*ws*2*u
        terms={c:[[],[]]for c in CHANNELS}
        for E,w in zip(energies,weights):
            if max_seconds is not None and time.monotonic()-start>max_seconds:raise TimeoutError('Emission moment runtime bound reached')
            logs=log_spectra(bath,float(E),energy_order=internal_order,angle_order=angle_order,tail=tail)
            for c in CHANNELS:
                for k in range(2):terms[c][k].append(math.log(DOS*w)+(2+k)*math.log(E)+logs[c])
        cp={c:[float(logsumexp(terms[c][k]))for k in range(2)]for c in CHANNELS}
        for c in CHANNELS:
            for k in range(2):channel_panels[c][k].append(cp[c][k])
        panels.append(dict(lo=lo,hi=hi,log_number_energy=cp))
    totals={c:[float(logsumexp(channel_panels[c][k]))for k in range(2)]for c in CHANNELS}
    totals['combined']=[float(logsumexp([totals[c][k]for c in CHANNELS]))for k in range(2)]
    values={c:dict(number_fm3_s=represented(v[0]),energy_MeV_fm3_s=represented(v[1]))for c,v in totals.items()}
    return dict(log_moments=totals,values=values,panels=panels,outer_order=outer_order,internal_order=internal_order,angle_order=angle_order,internal_tail=tail,cutoff_MeV=edges[-1],units='number fm^-3 s^-1; energy MeV fm^-3 s^-1; one neutrino helicity')
