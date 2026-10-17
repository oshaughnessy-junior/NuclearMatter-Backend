from pathlib import Path
import sys,json,math,hashlib
P=Path(__file__).resolve().parent;A=P.parent/'production_baryon_port';sys.dont_write_bytecode=True;sys.path.insert(0,str(A/'test_mirror/src'))
from nmbackend.reference import finite_t_baryons as B
assert hashlib.sha256(Path(B.__file__).read_bytes()).hexdigest()=='f4c007f400df346811d4ce759ee8d96d6b74170c7ef1d0c4f5bc063f8018f65b'
ref=json.loads((P/'results.json').read_text());assert not ref['failed'];checks=[];rows=[]
for idx in[0,2,5]:
 st=ref['rows'][idx]['state'];T,mu,m=[st[k]for k in['temperature','chemical_potential','mass']];slopes=[]
 for frac in[.001,.0005]:
  h=T*frac;lo=B.state_mu(T,mu-h,m);hi=B.state_mu(T,mu+h,m);slopes.append(dict(smu=(hi.entropy_density-lo.entropy_density)/(2*h),kmu=(hi.kinetic_energy_density-lo.kinetic_energy_density)/(2*h)))
 v={k:(4*slopes[1][k]-slopes[0][k])/3 for k in slopes[0]};target=dict(smu=st['dn_dtemperature_fixed_mu'],kmu=(mu-m)*st['dn_dmu_fixed_temperature']+T*st['dn_dtemperature_fixed_mu'])
 for k in v:checks.append(dict(name='independentresponseMaxwell '+str((idx,k)),error=v[k]/target[k]-1,gate=2e-6,passed=bool(abs(v[k]/target[k]-1)<=2e-6)))
 rows.append(dict(index=idx,ladder=slopes,richardson=v,targets=target))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows);(P/'maxwell_results.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
