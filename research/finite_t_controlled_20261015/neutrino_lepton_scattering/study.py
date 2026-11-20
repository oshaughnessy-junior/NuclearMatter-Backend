from pathlib import Path
import json,math,time
from scipy.integrate import quad
from scipy.special import expit
import scatter as S
P=Path(__file__).resolve().parent;rows=[];checks=[];start=time.monotonic()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save():
 (P/'results.json').write_text(json.dumps(dict(rows=rows,checks=checks,passed=all(c['passed']for c in checks),seconds=time.monotonic()-start),indent=2)+'\n')
T=2.;mu=.17446333200826397
for flavor,pos,anti,radmu in [('e',False,False,None),('e',True,False,None),('mu',False,False,None),('mu',True,False,None),('e',False,True,None),('e',False,False,0.)]:
 targetmu=-mu if pos else mu;density=quad(lambda p:p*p*expit((targetmu-math.hypot(S.ME,p))/T),0,200,epsabs=1e-18,epsrel=3e-12)[0]/(math.pi**2*S.HC**3);runs=[]
 for order,ang,phi,tail in [(16,10,12,40.),(24,16,20,60.)]:
  r=S.thermal(T,mu,10.,S.P.Couplings(flavor),positron=pos,anti=anti,radial=order,incident=ang,outgoing=ang,phi=phi,tail=tail,radiation_mu=radmu);runs.append(r);rows.append(r);gate('actualFDtargetdensity',r['target_density']/density-1,3e-9);gate('energy ledger',r['energy_ledger']/max(abs(r['outgoing_energy_moment']),10*r['rate']),3e-12);gate('eventfourmomentum',r['max_four_residual'],3e-10);save()
 for key in ['rate','transport_rate','neutrino_energy_change','outgoing_energy_moment']:gate('controlrefine '+str((flavor,pos,anti,radmu,key)),runs[0][key]/runs[1][key]-1,3e-6)
 save()
# Two new exact energy-redistribution points and direct reversed occupations.
diffs=[]
for ein,eout,c in [(10.,6.3,-.23),(10.,13.7,.41)]:
 for pos in [False,True]:
  fine=[]
  for order,tail in [(16,50.),(28,80.)]:fine.append(S.differential(T,mu,ein,eout,c,S.P.Couplings('e'),positron=pos,order=order,tail=tail))
  reverse=S.differential(T,mu,eout,ein,c,S.P.Couplings('e'),positron=pos,order=28,tail=80.)
  gate('redistributionenergyquadrefine',fine[0]['value']/fine[1]['value']-1,3e-8);gate('directoccupationKMS',fine[1]['reverse_same_phase']/fine[1]['value']*math.exp((ein-eout)/T)-1,3e-11);gate('reversedDOSratio',reverse['value']/fine[1]['value']*(eout/ein)**2*math.exp((ein-eout)/T)-1,3e-10);diffs.append(dict(forward=fine,reverse=reverse,positron=pos))
(P/'differential_results.json').write_text(json.dumps(dict(rows=diffs,checks=checks[-12:],passed=all(c['passed']for c in checks[-12:])),indent=2)+'\n');save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start);print([c for c in checks if not c['passed']])
