from pathlib import Path
import json,hashlib,math
P=Path(__file__).resolve().parent;A=P.parent/'heavy_baryon_limit_assessment';d=json.loads((P/'results.json').read_text());checks=[]
def ck(n,e,t):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
a=json.loads((A/'lambda1_scalar_profile0.json').read_text());b=json.loads((A/'lambda1_spinor_profile0.json').read_text());base=json.loads((A/'BASELINE_FORWARD.json').read_text());ck('savedforwardstagepassed',0 if base['status']=='BASELINE_FORWARD_PASS'and not base['failed']else 1,0)
for ch in['EC','PC']:
 if ch=='EC':x=a['raw']['EC']['scenarios']['empty']['net'];y=b['raw']['EC']['scenarios']['empty']['net'];names=list(x)
 else:x=dict(enumerate(a['raw']['PC']['sums'][0][0]));y=dict(enumerate(b['raw']['PC']['sums'][0][0]));names=list(x)
 for name in names:ck('savedlamb1absolute spinorphase '+str((ch,name)),x[name]/y[name]-1,3e-6)
 for k in range(2):ck('independentrapidity allowedanchor '+str((ch,k)),d['allowed'][ch][k]/base['allowed'][ch][k]-1,3e-12)
for n,h in json.loads((A/'MANIFEST.json').read_text())['files'].items():ck('frozenmanifestartifact '+n,0 if hashlib.sha256((A/n).read_bytes()).hexdigest()==h else 1,0)
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],scope='Savedbaselinefullspinorphase andartifact audit, NOTnewfullspinorphasererun. IndepfreshWeyl/currentandnewprofile evidence separate inresults.json.')
(P/'anchor_audit_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
