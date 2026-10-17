"""Selected finite-T CP energy exchange and isolated two-partner inverse tail."""
from pathlib import Path
import math,json
import pair as P
H=Path(__file__).parent;checks=[];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
old=json.loads((H/'thermal_results.json').read_text());e=next(z['result']for z in old['rows']if z.get('label')=='e'and z['result']['controls']['radial']==20)
r=P.thermal(2.,-.17446333200826397,P.Couplings('e'),radial=20,incident=8,outgoing=8,azimuth=8,tail=40.)
rows.append(dict(CP_empty=r))
for a,b in [('number','number'),('nu_energy','antinu_energy'),('antinu_energy','nu_energy'),('electron_energy','positron_energy'),('positron_energy','electron_energy')]:gate('CP exchange '+a,r['totals'][a]['value']/e['totals'][b]['value']-1,3e-11)
prior=json.loads((H/'pre_cleanup_fd_results.json').read_text());f=next(z['result']for z in prior['rows']if z.get('label')=='forward'and z['result']['controls']['radial']==24)
r=P.thermal(2.,.17446333200826397,P.Couplings('e'),radial=24,incident=12,outgoing=12,azimuth=12,tail=70.,radiation=P.Radiation(2.,.5,-2.1))
rows.append(dict(isolated_FD_tail70=r))
for branch in ['forward','backward']:
 for name in f[branch]:gate('isolated bothpartner tail '+branch+name,r[branch][name]['value']/f[branch][name]['value']-1,3e-6)
(H/'tail_cp_results.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
