from pathlib import Path
import json,math,hashlib
import scatter as S
P=Path(__file__).resolve().parent;checks=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
c=S.P.Couplings('e');r=json.loads((P/'refinement.json').read_text());k=json.loads((P/'kernel_integral_results.json').read_text());cm=r['rows'][1]
for key,val in k['rows'][-1]['values'].items():gate('absolute shellkernel vs CMevents '+key,val/cm[key]-1,3e-6)
# Isolated BOTH material-target and finalnu blocking already held physical; only
# extend radial tail at unchanged selected angular/radial rules.
tail=S.thermal(2.,.17446333200826397,10.,c,radial=32,incident=40,outgoing=40,phi=48,tail=80.)
for key in ['rate','transport_rate','neutrino_energy_change','outgoing_energy_moment']:gate('isolated thermal tail60to80 '+key,cm[key]/tail[key]-1,3e-6)
# Recheck actual FD forward root after control-only guards, finite signs/units.
gate('electron particle root',tail['target_density']/2.034286367843283e-7-1,3e-9)
for name,fn in [('nanEnu',lambda:S.thermal(2,0,math.nan,c)),('boolangle',lambda:S.thermal(2,0,10,c,incident=True)),('extremeFD',lambda:S.thermal(2,0,10,c,radiation_mu=40)),('boolcharge',lambda:S.thermal(2,0,10,c,positron=1)),('badT',lambda:S.thermal(.1,0,10,c)),('nanConditional',lambda:S.differential(2,0,10,math.nan,.3,c)),('collinear',lambda:S.differential(2,0,10,10,1,c)),('badtail',lambda:S.differential(2,0,10,8,.3,c,tail=0)),('boolconditionalorder',lambda:S.differential(2,0,10,8,.3,c,order=True))]:
 try:fn();gate('reject '+name,1,0)
 except ValueError:gate('reject '+name,0,0)
# Valid pre-guard physical differential rule reproduces selected output exactly.
prior=json.loads((P/'differential_results.json').read_text())['rows'][0]['forward'][-1];again=S.differential(2.,.17446333200826397,10,6.3,-.23,c,order=28,tail=80.)
gate('controlguard valid differential unchanged',again['value']/prior['value']-1,0)
(P/'final_checks.json').write_text(json.dumps(dict(checks=checks,passed=all(c['passed']for c in checks),isolated_tail=tail,absolute_kernel=k['rows'][-1],scope='one selectedelectronνE10 cross-method total/transport/energy+tails; noallflavorcontinuumcert'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks));print([c for c in checks if not c['passed']])
