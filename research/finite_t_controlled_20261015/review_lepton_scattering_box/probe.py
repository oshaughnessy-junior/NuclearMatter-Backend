"""Fresh finite-grid physics and independent dense DOP853 evolution."""
from pathlib import Path
import sys,importlib.util,math,json,hashlib,time
import numpy as np
from scipy.integrate import quad,solve_ivp
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];OWN=Path(__file__).parent;LEAF=ROOT/'lepton_scattering_box';sys.dont_write_bytecode=True
start=time.monotonic();checks=[]
def ck(n,error,tol):checks.append(dict(name=n,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
dirs=['lepton_scattering_box','neutrino_lepton_scattering','leptonic_pair_annihilation','relativistic_capture_oracle','thermodynamics'];paths=[p for d in dirs for p in(ROOT/d).rglob('*')if p.is_file()and'__pycache__'not in str(p)]
def hashes():return{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
before=hashes();sp=importlib.util.spec_from_file_location('review_frozen_scattering_box',LEAF/'box.py');B=importlib.util.module_from_spec(sp);sys.modules[sp.name]=B;sp.loader.exec_module(B)
ck('commissioned box SHA',float(hashlib.sha256((LEAF/'box.py').read_bytes()).hexdigest()!='7c724c889678581ecec4c5a645a5e99e940f4ae8142fe39278ff4870342ea77f'),0)
ck('commissioned manifest SHA',float(hashlib.sha256((LEAF/'MANIFEST.json').read_bytes()).hexdigest()!='5770812ee3fbefb9b7fbd08aafbce6a6eaaf1d316a6d0ab79d554ca4eebb5317'),0)
material=B.bath();ck('independent bath population gate',material['residual'],2e-9)
# Own analytic azimuth and transverse momentum delta shell, no author's D.
M=.51099895069;GF=1.1663785e-11;HBAR=6.582119569e-22;T=2.;mu=material['mu_e'];cv=.5+2*.23129;ca=.5
# q-outer permits exact angular measure with no sampled phi.
def Dq(E,F,Q,pos):
 om=E-F;a=Q*Q-om*om;emin=(-om+Q*math.sqrt(1+4*M*M/a))/2;mut=-mu if pos else mu;upper=max(emin,mut,mut-om)+85*T
 kz=(E*E+Q*Q-F*F)/(2*Q);kx2=max(0.,E*E-kz*kz);h=a/2;L,R=(cv+ca)**2,(cv-ca)**2
 if pos:L,R=R,L
 pz=lambda e:(2*e*om-a)/(2*Q)
 rmax=math.sqrt(max(0.,upper*upper-M*M-pz(upper)**2))
 def integrand(r):
  root=math.sqrt(1+4*(M*M+r*r)/a);et=(-om+Q*root)/2;der=2*Q*r/(a*root);u=E*et-kz*pz(et);v2=kx2*r*r/2
  amplitude=L*(u*u+v2)+R*((u-h)**2+v2)-M*M*(cv*cv-ca*ca)*h
  return der*amplitude*expit((mut-et)/T)*expit((et+om-mut)/T)
 val=quad(integrand,0,rmax,epsabs=1e-8,epsrel=3e-10,limit=120)[0]
 return F/(64*math.pi**3*E*Q*HBAR)*16*GF*GF*val
def edge(E,F):
 return sum(quad(lambda Q:Q/(E*F)*Dq(E,F,Q,pos),abs(E-F),E+F,epsabs=2e-11,epsrel=8e-10,limit=120)[0]for pos in[False,True])
g=B.grid(5,21.,angle=40,inner=32,tail=80.);E=np.array(g['E']);w=np.array(g['w']);W=np.array(g['W']);K=np.array(g['K']);A=W[:,None]*w[None,:]*K
ck('one helicity integrated bin DOS',np.max(abs(W/(w*E*E/(2*math.pi**2*197.3269804**3))-1)),3e-15)
ck('no repeated outgoing DOS',np.max(abs((np.array(g['A'])-A))/np.maximum(A,1e-300)),3e-15)
edges=[]
for i,j in [(0,2),(1,3),(3,4)]:
 forward=edge(float(E[i]),float(E[j]));reverse=edge(float(E[j]),float(E[i]));ck('fresh independent physical forward edge '+str((i,j)),K[i,j]/forward-1,3e-7);ck('fresh independent physical reverse edge '+str((i,j)),K[j,i]/reverse-1,3e-7);ck('fresh independent reciprocal DOS '+str((i,j)),reverse/forward/((E[i]/E[j])**2*math.exp((E[j]-E[i])/T))-1,3e-9);edges.append(dict(i=i,j=j,E=E[i],F=E[j],forward=forward,reverse=reverse,author_forward=K[i,j],author_reverse=K[j,i]))
# Independent row-column derivative, ordinary occupation products, no KMS replacement.
def rates(f):
 F=A*f[:,None]*(1-f[None,:]);dn=F.sum(axis=0)-F.sum(axis=1);heat=-E@dn
 macro=-np.log(f/(1-f))@dn+heat/T
 microscopic=0.
 for i in range(len(f)):
  for j in range(i+1,len(f)):
   microscopic+=(F[i,j]-F[j,i])*math.log(F[i,j]/F[j,i])
 return dn,heat,macro,microscopic,F
for munu in [-1.8,.4]:
 f=expit((munu-E)/T);dn,heat,s,sm,F=rates(f);ck('actual independent FD stationarity '+str(munu),np.max(abs(dn))/F.sum(),3e-12)
# Fresh non-FD distribution tests both holes, not temperature closure or forced FD.
f=expit((-.3-E)/2.6)*(1+.09*np.sin(E/4));dn,heat,s,sm,F=rates(f);author=B.collision(g,f)
ck('fresh nonFD dense vs paired derivative',np.max(abs(dn-np.array(author['dn'])))/F.sum(),2e-13)
ck('fresh nonFD independent physical entropy identity',(s-sm)/F.sum(),3e-12);ck('fresh nonFD positive total entropy',min(sm,0)/F.sum(),1e-14);ck('fresh nonFD bath heat opposite radiation',(heat+author['radiation_energy'])/max(abs(heat),1e-300),3e-13)
# Different ODE algorithm in particle populations, with independent heat/entropy quadrature.
def rhs(t,y):
 f=y[:len(E)]/W;dn,heat,macro,_,_=rates(f);return np.r_[dn,heat,macro]
def entropy(f):return -W@(f*np.log(f)+(1-f)*np.log1p(-f))
runs=[]
for Tr in [1.6,2.8]:
 f0=expit((-.55-E)/Tr);y0=np.r_[W*f0,0.,0.];sol=solve_ivp(rhs,[0,.06],y0,method='DOP853',rtol=2e-11,atol=1e-22,dense_output=True,max_step=.008);end=sol.y[:,-1];ff=end[:len(E)]/W
 ck('DOP853 successful '+str(Tr),float(not sol.success),0);ck('DOP853 particle conservation '+str(Tr),(sum(end[:len(E)])-W@f0)/(W@f0),3e-12);ck('DOP853 energy thermostat '+str(Tr),(E@(end[:len(E)]-W*f0)+end[-2])/abs(end[-2]),3e-11)
 ds=entropy(ff)-entropy(f0)+end[-2]/T;ck('independent entropy integral vs endpoint '+str(Tr),(ds-end[-1])/abs(ds),3e-9);ck('thermostat expected heat sign '+str(Tr),float((end[-2]>0)!=(Tr>T)),0)
 hh=[B.evolve(g,Tr,-.55,duration=.06,steps=n)for n in[8,16,32]];norm=lambda a:W@abs(np.array(a['final'])-ff)/(W@f0);err=[float(norm(a))for a in hh]
 for n,h in zip([8,16,32],hh):ck('Heun relative error bounded '+str((Tr,n)),W@abs(np.array(h['final'])-ff)/(W@abs(ff-f0)),.003)
 for a,b,n in zip(err[:-1],err[1:],[8,16]):ck('Heun independently measured second order '+str((Tr,n)),math.log2(a/b)-2,.16)
 ck('Heun32 heat against independent ODE '+str(Tr),hh[-1]['matter_heat']/end[-2]-1,.0003)
 runs.append(dict(Tr=Tr,DOP853_final=ff.tolist(),DOP853_heat=end[-2],DOP853_entropy=end[-1],DOP853_endpoint_entropy=ds,weighted_occupation_errors=err,Heun=hh,nfev=sol.nfev))
# Saved finite-boundary evidence: core is held fixed, missing cross-boundary flows are real.
sens=json.loads((LEAF/'sensitivity_results.json').read_text());tail=[x for x in sens['cases']if x['label']=='upper_extension'and x['Tr']==3. and x['tail_order']==6][0];tg=tail['grid'];Et=np.array(tg['E']);Wt=np.array(tg['W']);ft=expit((-.7-Et)/3.);At=np.array(tg['A']);flows=At*ft[:,None]*(1-ft[None,:]);forward=float(flows[:8,8:].sum());reverse=float(flows[8:,:8].sum())
ck('physical saved omitted forward flux',forward/tail['cross_boundary_forward']-1,3e-15);ck('physical saved omitted reverse flux',reverse/tail['cross_boundary_backward']-1,3e-15);ck('cutoff noneq exchanges do not cancel',float(abs(reverse-forward)<.2*max(reverse,forward)),0)
# Preserve six known failed gates across the actual raw reports.
failed=[]
for name in ['sensitivity_results.json','followup_results.json','grid_edge_results.json']:
 r=json.loads((LEAF/name).read_text());failed.extend([dict(file=name,check=x)for x in r['checks']if not x['passed']])
ck('six earlier failed gates remain preserved',len(failed)-6,0)
ck('all frozen source/raw bytes unchanged',float(before!=hashes()),0)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],material=material,grid=g,edges=edges,new_nonFD=dict(f=f.tolist(),dn=dn.tolist(),heat=heat,macro=s,micro=sm),runs=runs,omitted_boundary=dict(forward=forward,reverse=reverse),preserved_author_failures=failed,frozen_before=before,frozen_after=hashes(),seconds=time.monotonic()-start)
(OWN/'results.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(passed=out['passed'],failed=out['failed'],seconds=out['seconds'],runs=[dict(Tr=x['Tr'],heat=x['DOP853_heat'],errors=x['weighted_occupation_errors'])for x in runs]),indent=2))
