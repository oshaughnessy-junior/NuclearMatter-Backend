"""Independent pair-density identity and preserved v1/source checks."""
from pathlib import Path
import json,hashlib,math
from scipy.integrate import quad
from scipy.special import expit
P=Path(__file__).parent;ROOT=P.parent;checks=[]
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=abs(e)<=g))
x=json.loads((P/'training.json').read_text());dens=[quad(lambda p:p*p*expit((s*.17446333200826397-math.hypot(p,.51099895069))/2),0,250,epsabs=1e-12,epsrel=2e-13)[0]/(math.pi**2*197.3269804**3)for s in[1,-1]]
for row in x['rows']:
 for a,b in zip(row['fine']['targets'],dens):ck('independent actual pair density E'+str(row['E'])+' pos'+str(a['positron']),a['target_density']/b-1,2e-9)
ck('net neutrality same underlying npe material', (dens[0]-dens[1])/3e-8-1,2e-9)
for key,sha in json.loads((P/'model.json').read_text())['source_pins'].items():ck('physical source frozen '+key,0 if hashlib.sha256((ROOT/key).read_bytes()).hexdigest()==sha else 1,0)
old=ROOT/'lepton_scattering_fits';manifest=json.loads((old/'MANIFEST.json').read_text());print(type(manifest),list(manifest)[:5])
# V1 remains preserved; its exact frozen manifest is not modified or reclassified.
files=manifest.get('files',{})
for name,sha in files.items():
 if isinstance(sha,str):
  target=old/name
  ck('preserved v1 '+name,0 if target.is_file()and hashlib.sha256(target.read_bytes()).hexdigest()==sha else 1,0)
result=dict(checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],independent_pair_densities_fm3=dens,net_charge_fm3=dens[0]-dens[1]);(P/'final_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=result['failed']),indent=2))
