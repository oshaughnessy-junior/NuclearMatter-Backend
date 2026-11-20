from pathlib import Path
import json,math,time
import scatter as S
P=Path(__file__).resolve().parent;rows=[];checks=[];start=time.monotonic()
def save():
 (P/'refinement.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=all(x['passed']for x in checks),seconds=time.monotonic()-start),indent=2)+'\n')
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for flavor,pos,anti,radmu in [('e',False,False,None),('e',True,False,None),('mu',False,False,None),('mu',True,False,None),('e',False,True,None),('e',False,False,0.)]:
 runs=[]
 for n,ang,phi in [(24,24,32),(32,40,48)]:
  r=S.thermal(2.,.17446333200826397,10.,S.P.Couplings(flavor),positron=pos,anti=anti,radial=n,incident=ang,outgoing=ang,phi=phi,tail=60.,radiation_mu=radmu);rows.append(r);runs.append(r);save()
 for key in ['rate','transport_rate','neutrino_energy_change','outgoing_energy_moment']:gate('refine24to40 '+str((flavor,pos,anti,radmu,key)),runs[0][key]/runs[1][key]-1,3e-6)
 save()
save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
