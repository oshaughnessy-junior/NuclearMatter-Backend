"""Independent static Dirac spin trace / rapidity phase and fresh EOS-root assessment."""
from pathlib import Path
import sys,math,json,hashlib,importlib.util,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'allowed_rate_assessment';sys.dont_write_bytecode=True
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
F=load('allowed_review_frozen_adapter',A/'assess.py');R=load('allowed_review_independent_EOS',ROOT/'review_transparent_balance_grid/probe.py');E=F.C.E
checks=[];rows=[];start=time.monotonic();before={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()}
def ck(n,e,t=3e-10):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
GF=E.GF;VUD=E.VUD;gA=1.2723;me=F.C.ME;mn=F.C.MN;mp=F.C.MP;Q=mn-mp;hbar=6.582119569e-22;A0=GF*GF*VUD*VUD*(1+3*gA*gA)/(2*math.pi**3*hbar)
# Rest baryon projectors derive LO static spin sum independently of spectral code.
def sl(p,m=0):return np.einsum('u,u,uab->ab',p,E.MET,E.G)+m*E.I
V=E.G-gA*(E.G@E.G5);AV=E.adj(V[None,...])[0];L=E.G@(E.I-E.G5);AL=E.adj(L[None,...])[0];pa=np.array([mn,0,0,0]);pb=np.array([mp,0,0,0]);H=np.einsum('ab,ubc,cd,vda->uv',sl(pb,mp),V,sl(pa,mn),AV)
for cos in [-.79,.13,.83]:
 el=2.31;en=4.17;pl=math.sqrt(el*el-me*me);l=np.array([el,pl*math.sqrt(1-cos*cos),0,pl*cos]);k=np.array([en,0,0,en]);LT=np.einsum('ab,ubc,cd,vda->uv',sl(l,me),L,sl(k),AL);amp=GF*GF*VUD*VUD/2*np.einsum('uv,uv,u,v->',H,LT,E.MET,E.MET).real
 want=32*GF*GF*VUD*VUD*mn*mp*el*en*((1+3*gA*gA)+(1-gA*gA)*pl/el*cos);ck('static unaveraged Dirac spin trace '+str(cos),amp/want-1,3e-13)
ck('absolute per-target allowed prefactor',F.S.RATE_PREFACTOR/A0-1,3e-15)
def moment(T,mu,ch,N=96):
 if ch=='EC':lo=max(me,Q);hi=max(lo,mu)+100*T
 elif ch=='PC':lo=me;hi=max(me,-mu)+100*T
 else:lo=me;hi=Q
 a=math.acosh(lo/me);b=math.acosh(hi/me);x,w=leggauss(N);y=a+(x+1)*(b-a)/2;Eel=me*np.cosh(y);p=me*np.sinh(y);enu=Eel-Q if ch=='EC'else Eel+Q if ch=='PC'else Q-Eel;f=expit((mu-Eel)/T)if ch=='EC'else expit((-mu-Eel)/T)if ch=='PC'else expit((Eel-mu)/T);base=w*(b-a)/2*Eel*p*p*enu**2*f;return A0*np.array([sum(base),sum(base*enu)])
raw=json.loads((A/'results.json').read_text());roots=[];model=[]
for row in raw['rows']:
 T,n,y=row['T'],row['nB'],row['detailed_kinetic_Ye'];b=R.eos(T,n,y);ck('same independently reconstructed mue '+row['label'],b['mu_e']-row['actual_bath']['mu_e'],3e-11)
 compare={}
 for ch in ['EC','PC','beta']:
  target=n*y if ch=='EC'else n*(1-y);a=moment(T,b['mu_e'],ch);fine=moment(T,b['mu_e'],ch,144)
  for k in [0,1]:
   ck('fresh rapidity refinement '+str((T,n,ch,k)),a[k]/fine[k]-1,3e-10);ck('fresh absolute assessment moment '+str((T,n,ch,k)),target*fine[k]/row['channels'][ch]['allowed'][k]-1,3e-10)
  detailed=np.array(row['channels'][ch]['detailed']);er=target*fine/detailed-1;compare[ch]=er.tolist()
  for k,e in enumerate(er):model.append(dict(label=row['label'],channel=ch,moment=k,error=float(e),gate=.01,passed=bool(abs(e)<=.01)))
 trials=[]
 def drift(yy):
  bb=R.eos(T,n,yy);v={ch:(n*yy if ch=='EC'else n*(1-yy))*moment(T,bb['mu_e'],ch)for ch in ['EC','PC','beta']};net=v['beta'][0]+v['PC'][0]-v['EC'][0];trials.append(dict(Ye=yy,mu_e=bb['mu_e'],net=net));return net
 root=brentq(drift,.05,.9,xtol=2e-12);bb=R.eos(T,n,root);v={ch:(n*root if ch=='EC'else n*(1-root))*moment(T,bb['mu_e'],ch,144)for ch in ['EC','PC','beta']};net=v['beta'][0]+v['PC'][0]-v['EC'][0];gross=sum(z[0]for z in v.values())
 ck('fresh allowed root vs assessment '+row['label'],root-row['allowed_kinetic_Ye'],3e-10);ck('fresh root gross residual '+row['label'],net/gross,1e-9)
 model.append(dict(label=row['label'],channel='root',error=root-y,gate=.001,passed=bool(abs(root-y)<=.001)));roots.append(dict(label=row['label'],root=root,delta=root-y,affinity=bb['mu_n']-bb['mu_p']-bb['mu_e'],gross=gross,net=net,trials=trials));rows.append(dict(label=row['label'],relative_errors=compare))
# Reproduce failure accounting exactly, including beta passing the illustrative1%.
ck('20 retained model failures',len([c for c in model if not c['passed']])-20,0)
for ours,old in zip(model,raw['model_assessment_checks']):
 ck('model gate classification unchanged '+str((ours['label'],ours['channel'],ours.get('moment'))),0 if ours['passed']==old['passed']else 1,0)
# Vacuum allowed endpoint is a pure proper-model anchor, not lifetime calibration.
w=Q/me;fermi=(math.sqrt(w*w-1)*(2*w**4-9*w*w-8)+15*w*math.acosh(w))/60;analytic=A0*me**5*fermi;v=moment(.02,-30,'beta',144);ck('independent vacuum beta phase width',v[0]/analytic-1,3e-12)
for rel,h in raw['pins'].items():ck('frozen assessment source/data '+rel,0 if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h else 1,0)
after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()};ck('author bytes unchanged',int(before!=after),0)
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],model_checks=model,model_failures=[c for c in model if not c['passed']],roots=roots,rows=rows,seconds=time.monotonic()-start,before=before,after=after,absolute_allowed_width=analytic);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'],'modelFailures',len(out['model_failures']),'seconds',out['seconds'])
