"""Full-energy one-helicity charged-current moments for a fixed NR FD bath."""
from __future__ import annotations
from pathlib import Path
import sys,math,time
import numpy as np
from scipy.special import expn
from numpy.polynomial.legendre import leggauss
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'quantum_slab'))
import fixed_bath as slab

MEV_ERG=1.602176634e-6
EDGES=(0.,.1,.5,.78233336,1.29333236,2.,4.,8.,16.,24.,40.,70.,120.,180.,240.,320.)
NAMES=('emission_number_cm3_s','emission_energy_MeV_cm3_s','incoming_left_number_cm2_s','incoming_right_number_cm2_s','outgoing_left_number_cm2_s','outgoing_right_number_cm2_s','slab_emission_number_cm2_s','slab_absorbed_boundary_number_cm2_s','incoming_left_energy_MeV_cm2_s','incoming_right_energy_MeV_cm2_s','outgoing_left_energy_MeV_cm2_s','outgoing_right_energy_MeV_cm2_s','slab_emission_energy_MeV_cm2_s','slab_absorbed_boundary_energy_MeV_cm2_s')

def validate(length_cm,outer_order,internal_order,tail,edges):
    if not math.isfinite(length_cm) or length_cm<=0:raise ValueError('finite positive length required')
    for order in (outer_order,internal_order):
        if isinstance(order,bool) or not isinstance(order,int) or order<4:raise ValueError('integer quadrature orders>=4 required')
    if not math.isfinite(tail) or tail<=0:raise ValueError('finite positive internal tail required')
    a=np.asarray(edges,float)
    if a.ndim!=1 or len(a)<2 or not np.all(np.isfinite(a)) or a[0]!=0 or np.any(np.diff(a)<=0):raise ValueError('finite increasing energy edges starting at zero required')


def spectral(matter,flavor,E,length_cm,order=48,tail=40.):
    if not math.isfinite(length_cm) or length_cm<=0:raise ValueError('finite positive length required')
    c=slab.coefficients(matter,flavor,E,order=order,tail=tail)
    left,right=slab.boundary_occupations(E)
    f=slab.fluxes(c,length_cm,left,right)
    A=slab.LIGHT*slab.DOS*E*E*1e39
    escape=.5-float(expn(3,c['relaxation_cm_inv']*length_cm))
    emit=A*escape*c['feq'];absorbed=A*escape*.5*(left+right)
    en=slab.DOS*E*E*1e39*c['j_s']
    v=np.array([en,E*en,f['incoming_left'],f['incoming_right'],f['outgoing_left'],f['outgoing_right'],emit,absorbed,
                E*f['incoming_left'],E*f['incoming_right'],E*f['outgoing_left'],E*f['outgoing_right'],E*emit,E*absorbed])
    return v,c


def integrate(matter,flavor,*,length_cm=1e5,outer_order=12,internal_order=48,tail=40.,edges=EDGES):
    validate(length_cm,outer_order,internal_order,tail,edges)
    if flavor not in ('nue','antinue'):raise ValueError('nue/antinue required')
    xs,ws=leggauss(outer_order);panels=[];nodes=[];start=time.monotonic()
    for lo,hi in zip(edges[:-1],edges[1:]):
        integral=np.zeros(len(NAMES))
        for x,w in zip(xs,ws):
            E=float(lo+(x+1)*(hi-lo)/2);weight=float(w*(hi-lo)/2)
            v,c=spectral(matter,flavor,E,length_cm,internal_order,tail)
            integral+=weight*v
            nodes.append(dict(E=E,weight=weight,values=v.tolist(),guard_used=c['capture_panel_guard']['used']))
        panels.append(dict(lo=lo,hi=hi,values=integral.tolist()))
    total=np.sum([p['values'] for p in panels],axis=0)
    answer=dict(zip(NAMES,map(float,total)))
    answer['radiation_number_gain_cm2_s']=float(total[6]-total[7])
    answer['radiation_energy_gain_MeV_cm2_s']=float(total[12]-total[13])
    answer['bath_energy_source_erg_cm2_s']=-answer['radiation_energy_gain_MeV_cm2_s']*MEV_ERG
    sign=1 if flavor=='nue' else -1
    answer['bath_electron_lepton_source_cm2_s']=-sign*answer['radiation_number_gain_cm2_s']
    answer['emission_energy_erg_cm3_s']=answer['emission_energy_MeV_cm3_s']*MEV_ERG
    return dict(flavor=flavor,values=answer,panels=panels,nodes=nodes,outer_order=outer_order,internal_order=internal_order,internal_tail=tail,length_cm=length_cm,runtime_seconds=time.monotonic()-start)
