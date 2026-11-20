"""Selected PC spectrum controls; preserve coarse failures separately."""
from pathlib import Path
import importlib.util,sys,json,math,time
from scipy.special import expit
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('hotPC_spectrum_study_function',P/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
def run():
 start=time.monotonic();checks=[];rows=[];b,meta=S.matched()
 pilot=json.loads((P/'point_results.json').read_text())
 if pilot['failed']:raise RuntimeError('Absolute phase/current gate not qualified')
 for E in [2.,5.,10.]:
  vals=[]
  for rule in [(6,8,8,4,40.,24.),(8,12,12,4,50.,30.),(12,20,16,8,70.,40.)]:
   n,a,e,p,t,ot=rule;r=S.integrate(E,lepton_order=n,angle_order=a,energy_order=e,phi_order=p,tail=t,outer_tail=ot);vals.append(r)
   (P/'partial_results.json').write_text(json.dumps(dict(E=E,rules=vals,completed=rows),indent=2)+'\n')
  for k in ['j','chi']:
   for i in [0,1]:
    err=vals[i][k]['value']/vals[i+1][k]['value']-1;checks.append(dict(name=f'{E} rule{i}to{i+1} {k}',error=err,tolerance=3e-6,passed=abs(err)<=3e-6,kind='coarse'if i==0 else'qualification'))
  r=vals[-1];eq=r['equilibrium_mu_antinue'];f=float(expit((eq-E)/2));j,chi=r['j']['value'],r['chi']['value'];checks.append(dict(name=f'{E} separate KMS',error=r['log_kms_actual']-r['log_kms_expected'],tolerance=3e-12,passed=abs(r['log_kms_actual']-r['log_kms_expected'])<=3e-12));checks.append(dict(name=f'{E} trueabs vsrelax',passed=chi>0 and r['relaxation']==j+chi and r['relaxation']>chi));err=(j*(1-f)-chi*f)/(j*(1-f)+chi*f);checks.append(dict(name=f'{E} actualLTE',error=err,tolerance=3e-12,passed=abs(err)<=3e-12))
  rows.append(dict(E=E,rules=vals))
 # No stationary-proton cutoff: selected actual moving-target support below1.806MeV.
 d=S.differential(b,1.7,.5987748936352157,-.1995027501115606,energy_order=24,phi_order=8,tail=80.)
 checks.append(dict(name='belowrestthreshold actualthermal differential',passed=d['j']['value']>0 and d['chi']['value']>0))
 reference=S.P.integral(vars(b),order_n=32,order_e=52,order_angle=12,tail=60.);(P/'FRESH_EVENT_REFERENCE.json').write_text(json.dumps(reference,indent=2)+'\n')
 checks.append(dict(name='freshCMFDneutrondensity',error=reference['neutron_density']/7e-8-1,tolerance=3e-8,passed=abs(reference['neutron_density']/7e-8-1)<=3e-8))
 out=dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],below_rest_threshold=d,seconds=time.monotonic()-start);(P/'results.json').write_text(json.dumps(out,indent=2,default=lambda v:v.item())+'\n');print({k:out[k]for k in ['passed','failed','seconds']})
if __name__=='__main__':run()
