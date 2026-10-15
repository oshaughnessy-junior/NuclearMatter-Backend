"""Fine-rule qualification, explicitly excluding preserved coarse FD failures."""
from pathlib import Path
import sys,json,time,math,hashlib
sys.dont_write_bytecode=True
from muon_events import *
H=Path(__file__).parent;start=time.monotonic();initial=json.loads((H/'initial_refinement_results.json').read_text());checks=[c for c in initial['checks']if not c['name'].startswith(('FDpartners refinement0','FDpartners refinement1'))];diagnostics=[c for c in initial['checks']if c not in checks];rows=[]
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
def save(phase):
 (H/'qualified_results.json').write_text(json.dumps(dict(phase=phase,passed=phase=='complete'and all(c['passed']for c in checks),qualified_checks=checks,excluded_coarse_diagnostics=diagnostics,rows=rows,runtime_seconds=time.monotonic()-start,initial_report_sha256=hashlib.sha256((H/'initial_refinement_results.json').read_bytes()).hexdigest()),indent=2)+'\n')
m=eos.matter_state(2.,1e-4,.29,.01);b=Bath(2.,m.muon.chemical_potential,m.electron.chemical_potential);rad=Radiation(2.,15.,b.mu_mu-b.mu_e-15.-1.3)
reference={r['label']:r['result']for r in initial['rows']if'label'in r}
for label,radiation,blocking in [('unblocked',None,False),('transparent',None,True),('FDpartners',rad,True)]:
 old=reference[label+'_extended'];new=integrate(b,radiation,electron_blocking=blocking,parent_order=32,pair_order=40,pair_angle=24,orientation_order=20,azimuth_order=32,tail=50.)
 rows.append(dict(label=label+'_independent_tail50',result=new));save(label+' tail checkpoint')
 for branch in ['forward','backward']:
  for key in ['number','electron_energy','antinue_energy','numu_energy','parent_energy']:
   a=new[branch][key]['value'];z=old[branch][key]['value'];gate(label+' isolated tail50vs70 '+branch+' '+key,a/z-1 if z else a,3e-9)
# Finer FD benchmark; original frozen coarse failures are not rerun as PASS.
finer=integrate(b,rad,parent_order=36,pair_order=48,pair_angle=28,orientation_order=24,azimuth_order=40,tail=70.)
rows.append(dict(label='FDpartners_finer',result=finer));save('finer FD checkpoint')
alt=integrate(b,rad,phase='dalitz',parent_order=32,pair_order=40,pair_angle=24,orientation_order=20,azimuth_order=32,tail=70.)
rows.append(dict(label='FDpartners_alternate40',result=alt));save('independent phase checkpoint')
for label,value in [('pair40',reference['FDpartners_extended']),('dalitz40',alt)]:
 for branch in ['forward','backward']:
  for key in ['number','electron_energy','antinue_energy','numu_energy','parent_energy']:
   gate('qualified '+label+' vs pair48 '+branch+' '+key,value[branch][key]['value']/finer[branch][key]['value']-1,3e-6)
# Explicit material-fraction source convention at fixed declared nB.
rate=finer['net']['number'];nB=1e-4
rows.append(dict(composition_sources=dict(dYe_dt=rate/nB,dYmu_dt=-rate/nB,dYp_dt=0.,units='s^-1',nB=nB)))
save('complete');print(json.dumps(dict(passed=all(c['passed']for c in checks),checks=len(checks),excluded=len(diagnostics),excluded_failures=sum(not c['passed']for c in diagnostics),new_failures=[c for c in checks if not c['passed']],runtime=time.monotonic()-start)))
