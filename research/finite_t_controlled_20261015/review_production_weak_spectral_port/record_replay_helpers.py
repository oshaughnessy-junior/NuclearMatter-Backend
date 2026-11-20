"""Exact typed replay of frozen JSON records; never computes physical kernels."""
from dataclasses import asdict
import json

def coefficient(W,z):return W.SpectralQuantity(**z)
def signed_or_positive(W,z):
 if z is None:return None
 return W.SignedQuantity(**z)if'log_abs'in z else W.PositiveQuantity(**z)
def tree_tuple(x):
 if isinstance(x,list):return tuple(tree_tuple(v)for v in x)
 if isinstance(x,dict):return{k:tree_tuple(v)for k,v in x.items()}
 return x

def spectrum(W,z):
 a=dict(z);a['bath']=W.FreeElectronBath(**z['bath']);a['controls']=W.SpectralControls(**z['controls']);a['j']=coefficient(W,z['j']);a['true_chi']=coefficient(W,z['true_chi']);a['geometry']=tree_tuple(z['geometry']);a['differential_coordinates']=tree_tuple(z['differential_coordinates']);return W.SpectralResult(**a)
def reconstruction(W,z):
 a=dict(z);a['bath']=W.FreeElectronBath(**z['bath']);a['radiation']=W.Radiation(**z['radiation']);r=dict(z['rule']);r['nodes']=tuple(r['nodes']);r['weights']=tuple(r['weights']);a['rule']=W.EnergyRule(**r);a['controls']=W.SpectralControls(**z['controls']);a['nodes']=tuple(spectrum(W,v)for v in z['nodes']);a['forward']=tuple(W.PositiveQuantity(**v)for v in z['forward']);a['backward']=tuple(W.PositiveQuantity(**v)for v in z['backward']);a['net']=tuple(W.SignedQuantity(**v)for v in z['net']);a['entropy_production']=signed_or_positive(W,z['entropy_production']);a['units']=tuple(z['units']);return W.ReconstructionResult(**a)
def json_identity(record,raw):return json.loads(json.dumps(asdict(record)))==raw
