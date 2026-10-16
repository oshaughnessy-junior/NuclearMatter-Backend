"""Independent re-solved physical-state finite differences, before candidate import."""
from pathlib import Path
import json,importlib.util,math,time
import numpy as np
P=Path(__file__).resolve().parent;sp=importlib.util.spec_from_file_location('pure_fd_momentum_rmf',P/'oracle.py');O=importlib.util.module_from_spec(sp);sp.loader.exec_module(O);L=json.loads((P/'PREPARATION_LOCK.json').read_text());assert not json.loads((P/'pure_results.json').read_text())['failed'];base=json.loads((P/'oracle_state_0.json').read_text());n,T,Y=L['heldouts_nB_T_Yp'][0];nn=n*(1-Y);pp=n*Y;steps=L['FD_steps'];checks=[];raw=[];start=time.monotonic()
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
def solve(nn,pp,T):
 z=O.state(nn+pp,T,pp/(nn+pp));ck('perturbed forward netroots',max(abs(x)for x in z['number_residuals']),2e-9);ck('perturbed forwardscalar',z['scalar_residual']/z['nB'],2e-9);return z
for rh in steps:
 hT=rh*T;up=solve(nn,pp,T+hT);dn=solve(nn,pp,T-hT);muT=(np.array([up['mu_n'],up['mu_p']])-np.array([dn['mu_n'],dn['mu_p']]))/(2*hT)
 cv=(up['energy']-dn['energy'])/(2*hT);entropy=-(up['free_energy']-dn['free_energy'])/(2*hT);pdT=(up['pressure']-dn['pressure'])/(2*hT);H=np.zeros((2,2));HF=np.zeros((2,2));fgrad=[]
 hs=[rh*nn,rh*pp]
 for j,h in enumerate(hs):
  pair=np.array([nn,pp]);p1=pair.copy();p0=pair.copy();p1[j]+=h;p0[j]-=h;u=solve(*p1,T);d=solve(*p0,T);H[:,j]=(np.array([u['mu_n'],u['mu_p']])-np.array([d['mu_n'],d['mu_p']]))/(2*h);HF[j,j]=(u['free_energy']+d['free_energy']-2*base['free_energy'])/h**2;fgrad.append((u['free_energy']-d['free_energy'])/(2*h))
 ff=[]
 for sn,sp_ in[(1,1),(1,-1),(-1,1),(-1,-1)]:ff.append(solve(nn+sn*hs[0],pp+sp_*hs[1],T)['free_energy'])
 HF[0,1]=HF[1,0]=(ff[0]-ff[1]-ff[2]+ff[3])/(4*hs[0]*hs[1]);raw.append(dict(relative_step=rh,Cv=cv,entropy=entropy,H=H.tolist(),H_from_free=HF.tolist(),muT=muT.tolist(),fgrad=fgrad,pressureT=pdT,gibbsduhemT=base['entropy']+nn*muT[0]+pp*muT[1]))
 (P/'pure_fd_partial.json').write_text(json.dumps(dict(raw=raw,checks=checks,seconds=time.monotonic()-start),indent=2)+'\n')
a,b=raw[-2:];rich=lambda key:(4*np.array(b[key])-np.array(a[key]))/3
ck('pure RichardsonCv',float(rich('Cv'))/base['Cv']-1,3e-6);ck('pure Richardsons',float(rich('entropy'))/base['entropy']-1,3e-6)
HH=np.array(base['H']);Hrich=rich('H');HFrich=rich('H_from_free');ck('pure RichardsonH',np.linalg.norm(Hrich-HH)/np.linalg.norm(HH),2e-6);ck('pure RichardsonfreeH',np.linalg.norm(HFrich-HH)/np.linalg.norm(HH),1e-4);ck('pure RichardsonMaxwellsymmetry',np.linalg.norm(Hrich-Hrich.T)/np.linalg.norm(HH),2e-6);ck('pure RichardsonGibbsDuhemT',(float(rich('pressureT'))-float(rich('gibbsduhemT')))/base['entropy'],3e-6)
refined=O.state(n,T,Y,tail=115,intervals=91)
for k in ['energy','pressure','entropy','free_energy','Cv','D','phi','mD','mu_n','mu_p']:ck('pure tail90to115 grid73to91 '+k,refined[k]/base[k]-1,2e-9)
ck('pure tail90to115 H',np.linalg.norm(np.array(refined['H'])-HH)/np.linalg.norm(HH),2e-9)
out=dict(candidate_imported=False,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,raw=raw,refined=refined,Richardson=dict(Cv=float(rich('Cv')),entropy=float(rich('entropy')),H=Hrich.tolist(),H_from_free=HFrich.tolist()),seconds=time.monotonic()-start);(P/'pure_fd_results.json').write_text(json.dumps(out,indent=2)+'\n');print('PUREFD',out['passed'],'FAIL',len(out['failed']),'seconds',out['seconds'])
