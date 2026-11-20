"""Actual detailed CM references, fixed pair bath, explicit controls."""
from pathlib import Path
import importlib.util,sys,math,hashlib,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('fit_physical_scattering',ROOT/'neutrino_lepton_scattering/scatter.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
T=2.;MU=.17446333200826397;COARSE=(24,24,24,32,60.);FINE=(32,40,40,48,80.);KEYS=['rate','transport_rate','outgoing_energy_moment','neutrino_energy_change']
SOURCES=['neutrino_lepton_scattering/scatter.py','leptonic_pair_annihilation/pair.py','relativistic_capture_oracle/events.py']
def pins():return{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()for f in SOURCES}
PINS=pins()
def detailed(E,controls=FINE):
 if type(E)not in(int,float)or not math.isfinite(E)or not 1<=E<=30:raise ValueError('Finite1..30MeV incident energy required')
 if pins()!=PINS:raise RuntimeError('Frozen physical sources changed')
 n,i,o,ph,tail=controls;rows=[];start=time.monotonic()
 for pos in[False,True]:rows.append(S.thermal(T,MU,E,S.P.Couplings('e'),positron=pos,radial=n,incident=i,outgoing=o,phi=ph,tail=tail,radiation_mu=None,budget=110.))
 return dict(E=E,total={k:sum(z[k]for z in rows)for k in KEYS},targets=rows,controls=list(controls),pins=PINS,seconds=time.monotonic()-start,units='R/transport s^-1 per incidentnue; Qout/delta MeV s^-1; no finalnu blocking')
