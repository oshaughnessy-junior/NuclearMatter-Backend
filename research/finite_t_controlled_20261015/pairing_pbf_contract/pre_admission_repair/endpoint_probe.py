"""Selected Tc limit ladder, with the literal legacy clamp retained."""
from pathlib import Path
import json,math
from contract import LEGACY,current_coefficients
P=Path(__file__).resolve().parent;Tc=1e9;mst=.78;kf=1.9;c=current_coefficients(mst,kf)
base=1.170e21*mst*mst*c['vF_over_c_legacy']*3
slopes={'1S0':base*c['singlet_legacy_axial_only']*.602*(1.456-.157+1.764)**2,'3P2B':base*.5*1.26**2*1.204*(float(LEGACY['_C3P2'][2])+.2409)**2*(.7893+1.764)**2}
rows=[];checks=[]
for delta in [1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-10,1e-14,0.]:
 T=Tc*(1-delta);row=dict(delta=delta,T=T,actual_tau=float(LEGACY['below'](T,Tc,strict=False)[1]))
 for name,func in [('1S0',LEGACY['pbf_1s0']),('3P2B',LEGACY['pbf_n3p2'])]:
  q=float(func(T,Tc,mst,kf));row[name]=dict(Q=q,leading_linear_Tc_prediction=slopes[name]*delta,relative_to_linear=q/(slopes[name]*delta)-1 if delta else None)
  if delta in [1e-5,1e-6,1e-7]:
   error=q/(slopes[name]*delta)-1;checks.append(dict(name=name+' pre-clamp Tc asymptote '+str(delta),error=error,tolerance=.001,passed=abs(error)<.001))
 rows.append(row)
out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],slopes=slopes,rows=rows,qualification='Linear limit only above tau clamp distance; source deliberately saturates tau at1−1e−12 including Tc, not a literal exact-zero implementation.')
(P/'endpoint_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed']}))
