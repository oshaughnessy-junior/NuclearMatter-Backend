"""Private bounded transparent Euler benchmark. No public integrator or cache."""
from pathlib import Path
from dataclasses import asdict
import json,hashlib,sys,time,signal,math
from nmbackend.reference import free_matter as F
from nmbackend.reference import weak_reference as W
from nmbackend.reference import _free_matter_sources as S
P=Path(__file__).resolve().parent;B=P.parent/'free_source_binder'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name,v):(P/name).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(name):return json.loads((P/name).read_text())
def verify(lock):
 assert sha(B/'MANIFEST.json')==lock['binder_manifest_sha256']
 for f,h in load('binder_source_pin.json').items():assert sha(B/f)==h,f
 assert sha(P/'element.py')==lock['driver_sha256']
 assert asdict(F.SourceControls.fine())==lock['controls']
def state(T,y):return F.state_density(T,1e-7,y,photons='absent')
def initialise():
 assert not (P/'PRELOCK.json').exists(),'Never overwrite prelock'
 dump('binder_source_pin.json',json.loads((B/'MANIFEST.json').read_text()))
 dump('PRELOCK.json',dict(model=F.MODEL,initial=dict(T=2.,nB=1e-7,Ye=.30,photons='absent',radiation='empty'),duration_s=2e-4,rungs=[2,4],method='Explicit Euler in actual (T,Ye), fixed density; no trajectory accuracy claim from two rungs',max_fresh_source_groups=6,planned_unique_groups=5,per_group_seconds=88,controls=asdict(F.SourceControls.fine()),driver_sha256=sha(P/'element.py'),binder_manifest_sha256=sha(B/'MANIFEST.json'),gates=dict(energy_defect_over_gross_escaped=.02,lepton_defect_over_gross_emitted=2e-10,number_closure_relative=2e-9,instant_source_energy_scaled=2e-8,entropy_chain_defect_over_integrated_abs=.02),time_error_policy='Report h versus h/2 differences and ledger errors, without extrapolated certificate; preserve all failures'))
def call(s,group):
 t=time.monotonic();source=F.source_moments(s,radiation=W.Radiation.empty(),controls=F.SourceControls.fine());rhs=F.instantaneous_rhs(s,source,baryon_density_dot_fm3_s=0.)
 record=dict(group=group,state=asdict(s),state_identity=s.identity,source_identity=source.identity,source=S._payload(source),rhs=asdict(rhs),seconds=time.monotonic()-t,execution=dict(pid=__import__('os').getpid(),python=sys.version,driver_sha256=sha(P/'element.py'),binder_manifest_sha256=sha(B/'MANIFEST.json'),module_origins={m.__name__:m.__file__ for m in [F,W,S]}))
 dump('group_'+str(group)+'.json',record);return record

def advance(progress,rec):
 steps=progress['steps'];h=2e-4/steps
 s=state(progress['T'],progress['Ye']);assert s.identity==rec['state_identity'],'Cannot reuse different state source'
 assert progress['next_stage']<steps
 r=rec['rhs'];src=rec['source'];T=s.temperature_mev+h*r['temperature_dot_mev_s'];y=s.Ye+h*r['Ye_dot_s'];end=state(T,y)
 progress['history'].append(dict(stage=progress['next_stage'],h=h,group=rec['group'],state_identity=s.identity,source_identity=rec['source_identity'],T_before=s.temperature_mev,Ye_before=s.Ye,T_after=T,Ye_after=y))
 progress.update(T=T,Ye=y,next_stage=progress['next_stage']+1)
 for key,increment in [('escaped_energy',h*src['neutrino_energy_escape']['value']),('escaped_nue',h*src['nue_number_source']['value']),('escaped_anti',h*src['antinue_number_source']['value']),('entropy_integral',h*r['material_entropy_per_baryon_dot_s']),('entropy_abs_integral',h*abs(r['material_entropy_per_baryon_dot_s']))]:progress[key]+=increment
 progress['endpoint']=asdict(end)
 return progress

def run(group):
 signal.alarm(88);lock=load('PRELOCK.json');verify(lock)
 assert not (P/('group_'+str(group)+'.json')).exists(),'No source call overwrite/retry'
 if group==0:
  s=state(2.,.3);rec=call(s,0)
  for steps in [2,4]:
   p=dict(steps=steps,next_stage=0,T=2.,Ye=.3,initial=asdict(s),history=[],escaped_energy=0.,escaped_nue=0.,escaped_anti=0.,entropy_integral=0.,entropy_abs_integral=0.)
   dump('rung_'+str(steps)+'.json',advance(p,rec))
 else:
  steps=2 if group==1 else 4;p=load('rung_'+str(steps)+'.json')
  expected=1 if group==1 else group-1
  assert p['next_stage']==expected
  rec=call(state(p['T'],p['Ye']),group);dump('rung_'+str(steps)+'.json',advance(p,rec))
 verify(lock);signal.alarm(0);print(json.dumps(dict(group=group,seconds=rec['seconds'],T=rec['state']['temperature_mev'],Ye=rec['state']['Ye'],R=rec['source']['proton_number_source']['value'],Q=rec['source']['neutrino_energy_escape']['value'])))

def report():
 lock=load('PRELOCK.json');verify(lock);checks=[];rows=[]
 def ck(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and abs(error)<=tol)))
 for steps in [2,4]:
  p=load('rung_'+str(steps)+'.json');assert p['next_stage']==steps
  a,b=p['initial'],p['endpoint'];n=1e-7;q=p['escaped_energy'];le=p['escaped_nue']+p['escaped_anti'];material_delta=b['energy_density']-a['energy_density'];defect=material_delta+q
  ck('rest-inclusive energy/gross escape '+str(steps),defect/q,lock['gates']['energy_defect_over_gross_escaped'])
  ck('electron Le/gross emitted '+str(steps),(n*(p['Ye']-.3)+p['escaped_nue']-p['escaped_anti'])/le,lock['gates']['lepton_defect_over_gross_emitted'])
  for species,target in [('neutron',n*(1-p['Ye'])),('proton',n*p['Ye'])]:ck('endpoint particle number '+str((steps,species)),b[species]['particle_density']/target-1,2e-9)
  ck('endpoint charge '+str(steps),b['electron']['net_density']/b['proton']['particle_density']-1,2e-9)
  ds=b['entropy_density']/n-a['entropy_density']/n
  ck('material entropy chain '+str(steps),(ds-p['entropy_integral'])/p['entropy_abs_integral'],.02)
  thermal_initial=a['energy_density']-n*((1-.3)*F.MN+.3*(F.MP+F.ME))
  rows.append(dict(steps=steps,endpoint_T=p['T'],endpoint_Ye=p['Ye'],escaped_energy=q,material_energy_change=material_delta,energy_defect=defect,energy_defect_over_thermal_initial=defect/thermal_initial,energy_defect_over_escape=defect/q,thermal_initial=thermal_initial,material_entropy_change_per_baryon=ds,material_entropy_integral=p['entropy_integral'],escaped_nue=p['escaped_nue'],escaped_anti=p['escaped_anti']))
 for g in range(5):
  r=load('group_'+str(g)+'.json');ck('source energy ledger '+str(g),r['source']['ledgers']['material_energy_scaled'],2e-8)
 out=dict(checks=checks,passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,step_difference=dict(T=rows[0]['endpoint_T']-rows[1]['endpoint_T'],Ye=rows[0]['endpoint_Ye']-rows[1]['endpoint_Ye']),fresh_source_calls=5,initial_source_reused='Same executed group0 source, checked canonical identity, shared ONLY between initial stages',qualification='Two Euler rungs measure temporal sensitivity, not a convergence order/domain certificate. Complete3CC free model only, no long trajectory/hydro claim.')
 dump('RESULTS.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':
 if sys.argv[1]=='lock':initialise()
 elif sys.argv[1]=='report':report()
 else:run(int(sys.argv[1]))
