"""Minimally gauged instantaneous static OPE vector current, not a full weak rate.

Outgoing Q=-(k1+k2), k1=p3-p1,k2=p4-p2. Spin-pair then isospin-pair basis.
No retardation, two-body charge, arbitrary longitudinal completion, axial MEC.
"""
from __future__ import annotations
from pathlib import Path
import importlib.util,sys,hashlib,math,numbers
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
PINS={'ope_weak_diagrams/external_legs.py':'d8400295e4bc286ed5453c6da70d5f3d1cfc76e94f926b3eb7603e0ea55d7d14','ope_nucleon_amplitudes/ope.py':'a4d25710817060fc4fdcfb3f4b013012f67ead9c3cb07dab9407275e6eebefd7'}
def verify_sources():
 for name,h in PINS.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Frozen source changed: '+name)
verify_sources();spec=importlib.util.spec_from_file_location('static_vector_external_legs',ROOT/'ope_weak_diagrams/external_legs.py');W=importlib.util.module_from_spec(spec);sys.modules[spec.name]=W;spec.loader.exec_module(W)
I=np.eye(2);TAU=W.ope.PAULI;TISO=sum(np.kron(t,t)for t in TAU)
def vector(x):
 if np.iscomplexobj(x):raise ValueError('Real transfer required')
 x=np.asarray(x,float)
 if x.shape!=(3,)or not np.all(np.isfinite(x)):raise ValueError('Finite three-transfer required')
 with np.errstate(over='ignore',invalid='ignore'):square=x@x
 if not math.isfinite(square):raise ValueError('Representable transfer invariant required')
 return x

def direct(k1,k2,model=W.ope.Model(),*,gv=1.,raising=False):
 """Separate seagull and pion-in-flight spatial matrices, units MeV^-3.

 Derived by varying static pion gradient and derivative pion-NN interaction.
 Each piece is finite at Q=0; no Q division or imposed longitudinal projector.
 """
 verify_sources();k1=vector(k1);k2=vector(k2)
 if not isinstance(raising,(bool,np.bool_))or isinstance(gv,(bool,np.bool_))or not isinstance(gv,numbers.Real)or not math.isfinite(gv):raise ValueError('Boolean charge and finite real gv required')
 a1=np.einsum('i,ijk->jk',k1,TAU);a2=np.einsum('i,ijk->jk',k2,TAU);D1=k1@k1+model.m_pi**2;D2=k2@k2+model.m_pi**2;r=W.LOWER.conj().T if raising else W.LOWER;R=np.kron(r,I);B=R@TISO-TISO@R
 sea=[];flight=[]
 for i in range(3):
  spin=np.kron(TAU[i],a2)/D2-np.kron(a1,TAU[i])/D1
  sea.append(model.coefficient*float(gv)*np.kron(spin,B))
  flight.append(-model.coefficient*float(gv)*(k1[i]-k2[i])*np.kron(np.kron(a1,a2)/(D1*D2),B))
 sea=np.array(sea);flight=np.array(flight)
 if not np.all(np.isfinite(sea))or not np.all(np.isfinite(flight)):raise ArithmeticError('Nonfinite static current')
 return dict(seagull=sea,pion_flight=flight,total=sea+flight)

def antisymmetrized(p,model=W.ope.Model(),**kwargs):
 if np.iscomplexobj(p):raise ValueError('Real external momenta required')
 p=np.asarray(p,float)
 if p.shape!=(4,3)or not np.all(np.isfinite(p)):raise ValueError('Four finite three-momenta required')
 a=direct(p[2]-p[0],p[3]-p[1],model,**kwargs);ps=p[[0,1,3,2]];b=direct(ps[2]-ps[0],ps[3]-ps[1],model,**kwargs)
 return {name:a[name]-np.einsum('ij,ajk->aik',W.ope.EXCHANGE,b[name])for name in a}

def combined(p,omega,Q,*,model=W.ope.Model(),pole_gap=.05,gv=1.,raising=False):
 """Static vector continuity-consistent external legs+minimal pion current ONLY.

 Current tuple(V0,Vx,Vy,Vz), external conservation and offpole gate inherited.
 Does not add axial pieces or supply physical full transverse-current accuracy.
 """
 external=np.array([W.amplitude(p,omega,Q,c,model=model,pole_gap=pole_gap,gv=gv,raising=raising)for c in['V0','Vx','Vy','Vz']]);mec=antisymmetrized(p,model,gv=gv,raising=raising);total=external.copy();total[1:]+=mec['total']
 return dict(external=external,interaction=mec,total=total)
