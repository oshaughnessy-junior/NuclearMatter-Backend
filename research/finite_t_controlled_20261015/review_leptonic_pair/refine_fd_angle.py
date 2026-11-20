from pathlib import Path
import json,time,math
import numpy as np
import fd_reference as F
P=Path(__file__).resolve().parent;d=json.loads((P/'fd_results.json').read_text());lo=[x['result']for x in d['rows']if x['kind']=='independent'][-1];sc=dict(empty=None,LTE=(.5,-.5),positive=(.5,-2.1),negative=(.5,1.1));v=.5+2*.23129;a=.5;r=F.integrate(2.,.17446333200826397,v,a,sc,order=40,angle=16,phi=16,tail=60.);checks=[];names=['number','nu_energy','antinu_energy','electron_energy','positron_energy']
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for name,s in r['scenarios'].items():
 for branch in ['F','B']:
  for i,key in enumerate(names):
   if s[branch][i]:gate('isolated angle12to16 '+name+branch+key,lo['scenarios'][name][branch][i]/s[branch][i]-1,2e-6)
 if name=='empty':continue
 author=next(x['result']for x in d['rows']if x['kind']=='author'and x['name']==name)
 for branch,ab in [('F','forward'),('B','backward')]:
  for i,key in enumerate(names):gate('refined independent absolute '+name+branch+key,author[ab][key]['value']/s[branch][i]-1,2e-6)
(P/'angle_refinement.json').write_text(json.dumps(dict(checks=checks,result=r,passed=all(c['passed']for c in checks)),default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item(),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks),r['seconds']);print('max',max(abs(c['error'])for c in checks if'refined independent absolute'in c['name']))
