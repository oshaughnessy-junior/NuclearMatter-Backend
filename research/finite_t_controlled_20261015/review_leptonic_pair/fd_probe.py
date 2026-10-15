from pathlib import Path
import sys,json,math,time
import numpy as np
import fd_reference as F
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'leptonic_pair_annihilation'));import pair as A
rows=[];checks=[];start=time.monotonic();T=2.;mu=.17446333200826397;c=A.Couplings('e');scenarios=dict(empty=None,LTE=(.5,-.5),positive=(.5,-2.1),negative=(.5,1.1));names=['number','nu_energy','antinu_energy','electron_energy','positron_energy']
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save():
 (P/'fd_results.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=all(c['passed']for c in checks),seconds=time.monotonic()-start),default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item(),indent=2)+'\n')
for controls in [dict(order=28,angle=8,phi=8,tail=40.),dict(order=40,angle=12,phi=12,tail=60.)]:
 r=F.integrate(T,mu,c.cv,c.ca,scenarios,**controls);rows.append(dict(kind='independent',result=r));save()
 for name,s in r['scenarios'].items():
  net=s['net'];scale=s['F'][1]+s['F'][2]+s['B'][1]+s['B'][2];gate('independent energy '+name,(net[1]+net[2]-net[3]-net[4])/scale,3e-11)
  if name!='empty':gate('both partner KMS '+name,s['log_ratio_error'],3e-11);gate('entropy nonnegative '+name,int(s['entropy']< -1e-23),0)
refs=[x['result']for x in rows]
for name in scenarios:
 for branch in ['F','B']:
  for i,key in enumerate(names):
   lo,hi=refs[0]['scenarios'][name][branch][i],refs[1]['scenarios'][name][branch][i]
   if hi:gate('independent FD refine '+name+branch+key,lo/hi-1,2e-6)
# Fresh author computations, never importing/running its driver module.
for name in ['LTE','positive','negative']:
 un,ua=scenarios[name];run=A.thermal(T,mu,c,radial=20,incident=10,outgoing=10,azimuth=12,tail=60.,radiation=A.Radiation(T,un,ua));rows.append(dict(kind='author',name=name,result=run));save();ref=refs[-1]['scenarios'][name]
 for branch,ab in [('F','forward'),('B','backward')]:
  for i,key in enumerate(names):gate('independent FD absolute '+name+branch+key,run[ab][key]['value']/ref[branch][i]-1,2e-6)
 scale=ref['F'][0]+ref['B'][0]
 if name=='LTE':gate('reference LTE roundoff',ref['net'][0]/scale,3e-11)
 else:
  for i,key in enumerate(names):gate('signed net absolute '+name+key,run['net'][key]/ref['net'][i]-1,2e-6)
  gate('entropy absolute '+name,run['entropy']/ref['entropy']-1,2e-6);gate('entropy chemical identity '+name,(ref['entropy']+(un+ua)/T*ref['net'][0])/scale,3e-11)
save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
