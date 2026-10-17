from pathlib import Path
import sys,json,time,math
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'fd_fluid_element'))
import element as a
start=time.monotonic();m=a.physics.matter_state(2.003,7.03e-5,2.97e-5);rows=[];checks=[]
for flavor in ['nue','antinue']:
 coarse=a.physics.coefficients(m,flavor,8.,order=48,tail=40.);fine=a.physics.coefficients(m,flavor,8.,order=64,tail=50.)
 for key in ['j_s','chi_s']:
  err=abs(coarse[key]/fine[key]-1);checks.append(dict(name=f'heldout E8 {flavor} {key}48->64tail50',error=err,tolerance=3e-6,passed=bool(math.isfinite(err)and err<3e-6)))
 rows.append(dict(flavor=flavor,E=8.,coarse=coarse,fine=fine))
r=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,state=dict(T=2.003,nB=1e-4,Ye=.297),runtime_seconds=time.monotonic()-start,scope='Selected differential coefficients only; no new full-energy integral qualification')
Path(__file__).with_name('selected_source_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(dict(passed=r['passed'],checks=len(checks),runtime=r['runtime_seconds'])))
