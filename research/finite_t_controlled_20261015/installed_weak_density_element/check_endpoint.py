"""Fresh actual changed-state source profiles, tail and independent EOS inversion checks."""
from pathlib import Path
import sys,json,math
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P));import element as E
z=json.loads((P/'trajectory_8.json').read_text());final=z['final'];s=E.M.state(final['T'],final['nB'],final['Ye'],photons=True);profile=sys.argv[1]
if profile not in E.L['endpoint_profiles']:raise ValueError('Declaredprofilesonly')
source=E.sources(s,profile);target=z['conserved']['u'];tr=brentq(lambda T:E.M.state(T,final['nB'],final['Ye'],photons=True).specific_energy-target,1.9,2.1,xtol=1e-12,rtol=1e-13);checks=[]
def ck(name,error,gate):checks.append(dict(name=name,error=float(error),gate=gate,passed=bool(abs(error)<=gate)))
ck('independent bracket root T',tr-final['T'],2e-8)
for name,error in source['density_residuals'].items():ck('changedstate incoming density '+name,error,E.L['gates']['source_density_relative'])
ck('changedstate species energy',source['energy_residual'],2e-9);ck('changedstate directenergy',source['direct_energy_residual'],2e-9)
out=dict(profile=profile,state=s.asdict(),source=source,brent_temperature=tr,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']]);(P/f'endpoint_{profile}.json').write_text(json.dumps(out,indent=2)+'\n');print('endpoint',profile,out['passed'],out['failed'])
