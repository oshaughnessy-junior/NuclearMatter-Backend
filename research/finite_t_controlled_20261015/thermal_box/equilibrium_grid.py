from pathlib import Path
import json,time,math
import numpy as np
import box as b
P=Path(__file__).resolve().parent;old=json.loads((P/'initial83pass_before_rate_guards.json').read_text());start=time.monotonic();nB=old['nB'];v=old['initial']['totals'];g=b.grid([0.,2.,4.,6.,8.,11.,14.,19.,24.]);eq=b.equilibrium(nB,g,v['energy_MeV_per_baryon'],v['electron_lepton_per_baryon']);r=b.rates(eq['T'],eq['Ye'],nB,g);C=r['j']*(1-eq['f'])-r['chi']*eq['f'];checks=[]
for field in ['energy_MeV_per_baryon','electron_lepton_per_baryon']:checks.append(dict(name='8binLTE '+field,error=abs(eq['totals'][field]-v[field]),passed=abs(eq['totals'][field]-v[field])<3e-10))
error=float(np.max(abs(C)/(r['j']+r['chi'])));checks.append(dict(name='8binLTE actual physical stationarity',error=error,passed=error<3e-12))
report=dict(passed=all(c['passed']for c in checks),checks=checks,grid=g,LTE=eq,rates=r,delta_8_minus_4=dict(T=eq['T']-old['LTE']['T'],Ye=eq['Ye']-old['LTE']['Ye']),runtime_seconds=time.monotonic()-start,scope='Finite-grid LTE4versus8 bins at identical total E/Le; records grid sensitivity only, no continuum or energy-grid-converged trajectory claim')
(P/'equilibrium_grid.json').write_text(json.dumps(report,default=lambda x:x.tolist()if isinstance(x,np.ndarray)else x.item(),indent=2)+'\n');print(len(checks),report['passed'],report['delta_8_minus_4'],report['runtime_seconds'])
