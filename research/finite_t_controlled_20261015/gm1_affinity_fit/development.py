"""Diagnose old frozen failure, fit only training/development targets; no holdouts."""
from pathlib import Path
import sys,json,time,math,hashlib,atexit
import numpy as np
from numpy.polynomial import chebyshev as ch
sys.dont_write_bytecode=True
from potential import *
ROOT=P.parent;sys.path.insert(0,str(ROOT/'rmf_reference'));import gm1_finite_t as E
start=time.monotonic();training=[];phase='start';checks=[]
oldfiles=['gm1_potential_fit/potential.py','gm1_potential_fit/coefficients.json','review_gm1_fit/results.json','rmf_reference/gm1_finite_t.py'];hashes={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in oldfiles}
def save():(P/'development_partial.json').write_text(json.dumps(dict(phase=phase,training=training,checks=checks,runtime_seconds=time.monotonic()-start,source_sha256=hashes),indent=2)+'\n')
atexit.register(save)
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
# Diagnose original independent failure BEFORE changing coefficient/model data.
import importlib.util
spec=importlib.util.spec_from_file_location('frozen_old_potential',ROOT/'gm1_potential_fit/potential.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
n,T,Y=.1811,.12,.1817;s=E.state(n,T,Y);pred=old.evaluate(n,T,Y);dmu=pred['mu_n']-pred['mu_p']-s.mu_n_mev+s.mu_p_mev;diagnosis=dict(n=n,T=T,Y=Y,exact=s.asdict(),old_prediction=pred,delta_mu=dmu,balance_factor_error=abs(math.expm1(dmu/T)),gate=.01,passed=abs(math.expm1(dmu/T))<=.01,role='DEVELOPMENT: old independent failure, no longer a heldout')
(P/'old_failure_reproduced.json').write_text(json.dumps(diagnosis,indent=2)+'\n');gate('old1percentfailure still reproduced',float(diagnosis['passed']),0.);gate('old failure exact forward',s.max_normalized_equation_residual,2e-9)
# Predeclared compact tensor basis; C0 includes physical derivative constraints.
deg=(6,6);weights=dict(free=1/.0002,aY=1/.0004,an=1/.005);ns=[.18,.19,.20,.21,.22,.23,.24,.25];ys=[.18,.20,.22,.24,.26,.28,.30,.32];points=[(n,Y)for n in ns for Y in ys]+[(.1811,.1817)]
(P/'PROTOCOL.json').write_text(json.dumps(dict(domain=DOMAIN,balance_minimum_temperature=.12,balance_gate=.01,degrees=deg,parameters=196,fields=['cold','T2','T4','T6'],temperature_anchors=[0,2,3],training_n=ns,training_Y=ys,extra_development=[.1811,.1817],cold_derivative_weights=weights,bulk_gates=dict(free=.005,energy=.005,pressure=.01,entropy=.01,Cv=.01,Hessian=.02),holdouts='NOT evaluated in this program; coefficients/hash freeze precedes NEW blocked physical holdouts',oldfailedpoint='development only'),indent=2)+'\n')
for n,Y in points:
 if time.monotonic()-start>108:raise TimeoutError('Bounded EOS training experiment')
 phase='training';cold=E.cold_state(n,Y);w2=E.state(n,2,Y);w3=E.state(n,3,Y);mass=cold.dirac_mass_mev/E.HBARC;gamma=sum(E.kf(z*n)*math.hypot(E.kf(z*n),mass)for z in [1-Y,Y])/(3*E.HBARC);a2=-gamma/(2*n);a0=cold.free_energy_density/n-M
 # Exact O(T²) Sommerfeld leading field; two physical warm states fix O(T4/T6).
 rhs=np.array([w2.free_energy_density/n-cold.free_energy_density/n-a2*4,w3.free_energy_density/n-cold.free_energy_density/n-a2*9]);a4,a6=np.linalg.solve(np.array([[16.,64.],[81.,729.]]),rhs)
 training.append(dict(n=n,Y=Y,cold=cold.asdict(),warm2=w2.asdict(),warm3=w3.asdict(),cold_field=a0,T2_field=a2,T4_field=float(a4),T6_field=float(a6),cold_aY=cold.mu_p_mev-cold.mu_n_mev,cold_an=cold.pressure/n**2))
 for z in [cold,w2,w3]:gate('training exact forward',z.max_normalized_equation_residual,2e-9)
 save()
# Build derivative design from the SAME potential basis, not independent fitted mu functions.
V=[];VY=[];VN=[]
for r in training:
 n,Y=r['n'],r['Y'];V.append(ch.chebvander2d(scale(n,DOMAIN['n']),isospin_coordinate(Y),deg).reshape(-1));dy=[];dn=[]
 for i in range(7):
  for j in range(7):
   cc=np.zeros((7,7));cc[i,j]=1;dy.append(field(cc,n,Y,dy=1));dn.append(field(cc,n,Y,dn=1))
 VY.append(dy);VN.append(dn)
V=np.array(V);VY=np.array(VY);VN=np.array(VN);design=np.concatenate([weights['free']*V,weights['aY']*VY,weights['an']*VN]);target=np.concatenate([weights['free']*np.array([r['cold_field']for r in training]),weights['aY']*np.array([r['cold_aY']for r in training]),weights['an']*np.array([r['cold_an']for r in training])]);coef={};coef['cold']=np.linalg.lstsq(design,target,rcond=None)[0].reshape(7,7).tolist()
for key in ['T2','T4','T6']:coef[key]=np.linalg.lstsq(V,np.array([r[key+'_field']for r in training]),rcond=None)[0].reshape(7,7).tolist()
metadata=dict(fields=coef,degrees=deg,parameters=196,condition_number_C0=float(np.linalg.cond(design)),condition_number_other=float(np.linalg.cond(V)),training_physical_states=len(training)*3,form='a=939+C0(n,delta²)+T²C2+T⁴C4+T⁶C6',domain=DOMAIN,oldfailedpoint_role='training/development',source_sha256=hashes)
(P/'coefficients.json').write_text(json.dumps(metadata,indent=2)+'\n');new=evaluate(.1811,.12,.1817);diff=new['mu_n']-new['mu_p']-s.mu_n_mev+s.mu_p_mev;gate('new development balance gate',abs(math.expm1(diff/.12)),.01);metadata['development_oldpoint']=dict(prediction=new,delta_mu=diff,balance_error=abs(math.expm1(diff/.12)));metadata['checks']=checks;metadata['training']=training;metadata['runtime_seconds']=time.monotonic()-start;metadata['passed']=all(c['passed']for c in checks);metadata['end_source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()for p in oldfiles};assert metadata['end_source_sha256']==hashes
phase='complete';(P/'development_results.json').write_text(json.dumps(metadata,indent=2)+'\n');freeze={name:hashlib.sha256((P/name).read_bytes()).hexdigest()for name in ['potential.py','coefficients.json','PROTOCOL.json']};(P/'PRE_HOLDOUT_FREEZE.json').write_text(json.dumps(dict(files=freeze,phase='coefficients,basis,domain,gates frozen BEFORE newholdout evaluation',training_used_only=True),indent=2)+'\n');print(len(checks),metadata['passed'],diff,metadata['development_oldpoint']['balance_error'],metadata['runtime_seconds'],metadata['condition_number_C0'],metadata['condition_number_other'])
