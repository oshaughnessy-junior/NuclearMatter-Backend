from pathlib import Path
import sys,json,math,importlib.util
import numpy as np
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'hot_beta_moments'));import hot_moments as A
sp=importlib.util.spec_from_file_location('flat_dalitz_inverse',R/'review_crossed_moments/reference.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D)
b,_=A.K.matched_npe(2.,1e-7,.3);eq=b.mu_n-b.mu_p-b.mu_e;r=D.integrate(b,eq+.8*b.T,order=28,radial=16,orient=12,phi=16,budget=110);target=[x['result']for x in json.loads((R/'hot_beta_moments/results.json').read_text())['rows']if x.get('label')=='Laguerre'][-1]['scenarios']['inverse_dominated'];checks=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
gate('new inverse forward root',r['density']/7e-8-1,3e-9);s=r['scenarios']['FD']
for branch,aname in [('F','forward'),('B','backward')]:
 for i,key in enumerate(D.NAMES):gate('independent inverse positive '+branch+key,target[aname][key]['value']/s[branch][i]-1,3e-6)
for i,key in enumerate(D.NAMES):gate('independent signed inverse '+key,target['net'][key]/s['net'][i]-1,3e-6)
gate('inverse net sign',int(s['net'][0]>=0),0);gate('inverse entropy positive',int(s['entropy']<=0),0);gate('negative affinity entropy',(s['entropy']+.8*s['net'][0])/(s['F'][0]+s['B'][0]),3e-11);gate('author entropy absolute',target['entropy_production']/s['entropy']-1,3e-6)
(P/'inverse_results.json').write_text(json.dumps(dict(checks=checks,passed=all(x['passed']for x in checks),result=r,affinity=-.8*b.T),default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item(),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks),r['seconds'])
