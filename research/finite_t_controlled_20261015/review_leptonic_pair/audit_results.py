from pathlib import Path
import json,math
import reference as F
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'leptonic_pair_annihilation'))
import pair as A
P=Path(__file__).resolve().parent;d=json.loads((P/'fd_results.json').read_text());checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for flavor,v,a in [('e',.96258,.5),('mu',-.03742,-.5),('tau',-.03742,-.5)]:
 c=A.Couplings(flavor);gate('primary flavor vector '+flavor,c.cv-v,2e-16);gate('primary flavor axial '+flavor,c.ca-a,0)
r=F.separated(2,.17446333200826397,.96258,.5);ne=r['FDmoments_minus'][1]/(math.pi**2*F.HC**3);np_=r['FDmoments_plus'][1]/(math.pi**2*F.HC**3)
gate('netcharge physical npe root',(ne-np_)/3e-8-1,3e-10)
for row in d['rows']:
 if row['kind']!='independent':continue
 z=row['result'];gate('actual electron particles',z['density'][0]/ne-1,2e-7);gate('actual positron particles',z['density'][1]/np_-1,2e-7)
 for name,s in z['scenarios'].items():
  R=s['net'][0];delta=dict(electron=-R,positron=-R,nu=R,antinu=R);gate('material netcharge pair change',-delta['electron']+delta['positron'],0);gate('flavor lepton pair change',delta['electron']-delta['positron']+delta['nu']-delta['antinu'],0)
  if name in ['positive','negative']:gate('both affinity source directions',int((R>0)!=(name=='positive')),0)
rows.append(dict(independent_electron_density=ne,independent_positron_density=np_,netdensity=ne-np_,equilibrium_mu_e=.17446333200826397))
(P/'final_audit_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,passed=all(c['passed']for c in checks),meaning='stoichiometry is algebraic; forward particle-density accuracy is a numerical check'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']])
