"""Pinned NMB PBF function bodies plus separately labelled gap/current contracts.

AST functions are literal legacy prescription, not a modern pairing response.
"""
from pathlib import Path
import ast,hashlib,math,numbers
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
P=Path(__file__).resolve().parent
PINS={'core_nu.py':'520d3b458c5eddbed1e1e1a5c6270dcc12643eb5','superfluid.py':'e16a07cfa7b92530649822358c552b1fa7069042'}

def load_pinned():
    ns={'jnp':jnp,'np':np}
    for f,names in [('superfluid.py',{'below','u_1s0','u_3p2b','f32'}),('core_nu.py',{'_ctrl_1s0','_ctrl_3p2b','_pbf','pbf_1s0','pbf_n3p2','pbf_neutron'})]:
        p=P/'source_snapshot'/f;b=p.read_bytes()
        if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()!=PINS[f]:raise RuntimeError('Pinned scientific source changed')
        nodes=[]
        for x in ast.parse(b).body:
            if isinstance(x,ast.FunctionDef) and x.name in names:nodes.append(x)
            elif isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['_C011','_C3P2'] for t in x.targets):nodes.append(x)
        if {x.name for x in nodes if isinstance(x,ast.FunctionDef)}!=names:raise RuntimeError('AST inventory changed')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns

LEGACY=load_pinned()

def gap(tau,variant):
    if isinstance(tau,bool) or not isinstance(tau,numbers.Real) or not math.isfinite(tau) or not 0<tau<=1:raise ValueError('finite0<tau<=1 required; zero-T limit is not a finite gap/T')
    if variant=='YK1999_A':return math.sqrt(1-tau)*(1.456-.157/math.sqrt(tau)+1.764/tau)
    if variant=='YK1999_B_minimum':return math.sqrt(1-tau)*(.7893+1.188/tau)
    if variant=='NSCool_NMB_B_legacy':return math.sqrt(1-tau)*(.7893+1.764/tau)
    raise ValueError('named gap convention required')

def current_coefficients(mst,kf):
    if any(isinstance(x,bool) or not isinstance(x,numbers.Real) or not math.isfinite(x) or x<=0 for x in [mst,kf]):raise ValueError('positive finite Landau ratio,kF required')
    vf=kf*197./(mst*940.)
    return dict(vF_over_c_legacy=vf,singlet_legacy_axial_only=1.60*vf**2*(mst*mst+11/42),triplet_legacy_default_axial=.5*1.26**2,triplet_vector_removed_bare_axial=2*1.26**2,triplet_original_bare_vector_plus_axial=1+2*1.26**2,qualification='Coefficients only; current/gap variants cannot be mixed without matching conventions. No finite-q vector or full anomalous-vertex response implemented.')

def legacy(channel,T,Tc,mst,kf,*,ca=.5):
    """Admitted finite-positive-temperature scalar execution of exact NMB AST.

    T,Tc kelvin; kF fm^-1; mst Landau/bare ratio; total Q cgs,3flavors.
    1S0 supports either species in the legacy prescription; 3P2B is neutron.
    Raw LEGACY remains available for excluded endpoint replays.
    """
    if channel not in ['1S0','3P2B']:raise ValueError('named channel required')
    if any(isinstance(x,bool) or not isinstance(x,numbers.Real) or not math.isfinite(x) for x in [T,Tc,mst,kf,ca]) or T<=0 or Tc<0 or mst<0 or kf<0 or ca<0:raise ValueError('finite T>0,Tc,mst,kF,ca>=0 required')
    # Selected scalar numerical admission only; no positivity/cold certificate.
    if Tc>0 and T/Tc<.01:raise ValueError('outside selected legacy scalar tau>=.01; raw endpoints separately replayed')
    fun=LEGACY['pbf_1s0'] if channel=='1S0' else LEGACY['pbf_n3p2']
    q=float(fun(T,Tc,mst,kf) if channel=='1S0' else fun(T,Tc,mst,kf,ca))
    if not math.isfinite(q) or q<0:raise ArithmeticError('unqualified legacy scalar')
    return q
