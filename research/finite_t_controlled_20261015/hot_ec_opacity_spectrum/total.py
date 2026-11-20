"""Exact prescribed node chunks for bounded one-helicity spectral reconstruction.

This is evidence assembly, not a physics-rate cache or API approximation.
"""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit
P=Path(__file__).parent;ROOT=P.parent
sp=importlib.util.spec_from_file_location('hot_EC_spectral_total_functions',P/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def contract(which):
 order,upper,controls=(24,50.,[8,12,12,4,50.,30.])if which=='coarse'else(32,60.,[12,20,16,4,70.,40.]);x,w=leggauss(order);nodes=(x+1)*upper/2;weights=w*upper/2
 c=dict(rule=which,nu_order=order,upper=upper,controls=controls,nodes=nodes.tolist(),weights=weights.tolist(),spectrum_sha=sha(P/'spectrum.py'),physical_sources=S.PINS,reference_total_sha=sha(ROOT/'hot_electron_capture_moments/results.json'),accepted_review_sha=sha(ROOT/'review_hot_electron_capture/REVIEW.md'),state_sha=sha(ROOT/'hot_electron_capture_moments/MATCHED_STATE.json'))
 return c

def collect(which,part):
 file=P/f'total_{which}_part{part}.json'
 if file.exists():raise RuntimeError('Refusing to overwrite frozen node evidence')
 c=contract(which);N=c['nu_order'];ids=list(range(N))if which=='coarse'else list(range(part*16,(part+1)*16));rows=[];start=time.monotonic();n,a,e,ph,t,ot=c['controls']
 if which=='coarse'and part!=0 or which=='fine'and part not in[0,1]:raise ValueError('Declared exact node chunk only')
 for i in ids:
  if time.monotonic()-start>105:raise TimeoutError('Node-chunk bounded105s; partial remains unqualified')
  r=S.integrate(float(c['nodes'][i]),lepton_order=n,angle_order=a,energy_order=e,phi_order=ph,tail=t,outer_tail=ot,budget_seconds=20.);rows.append(dict(index=i,E=c['nodes'][i],weight=c['weights'][i],spectrum=r));(P/f'total_{which}_part{part}_partial.json').write_text(json.dumps(dict(contract=c,rows=rows,qualified=False),indent=2));print(which,part,i,r['seconds'],flush=True)
 if contract(which)!=c:raise RuntimeError('Evidence provenance changed mid-experiment')
 out=dict(contract=c,rows=rows,seconds=time.monotonic()-start,qualified=True);file.write_text(json.dumps(out,indent=2)+'\n')

def validate_node(c,r):
 i=r['index'];v=r['spectrum']
 n,a,e,ph,t,ot=c['controls'];expected=dict(lepton_order=n,angle_order=a,energy_order=e,phi_order=ph,tail=t,outer_tail=ot)
 if type(i)is not int or not 0<=i<c['nu_order']or r['E']!=c['nodes'][i]or r['weight']!=c['weights'][i]or v['Enu']!=r['E']or v['controls']!=expected or v['bath']!=S.FIXED_BATH or v['matched_state']!=json.loads((ROOT/'hot_electron_capture_moments/MATCHED_STATE.json').read_text())or v['source_pins']!=S.PINS or v['implementation_sha256']!=c['spectrum_sha']:raise RuntimeError('Node/state/control/source mismatch')
 if v['units']!='j and true chi: s^-1 per one active nue occupation; relaxation=j+chi'or v['channel']!='p+e- <-> n+nue ONLY; no joined inventory':raise RuntimeError('Units/channel mismatch')
 for key in['j','chi']:
  z=v[key]
  if type(z['value'])not in(int,float)or not math.isfinite(z['value'])or z['value']<=0 or type(z['log'])not in(int,float)or not math.isfinite(z['log'])or z['representable']is not True or z['reason']is not None or abs(math.log(z['value'])-z['log'])>1e-12:raise RuntimeError('Selected positive spectral coefficient representation mismatch')

def combine():
 groups={};checks=[]
 def ck(name,er,gate=3e-6):checks.append(dict(name=name,error=float(er),gate=gate,passed=bool(math.isfinite(er)and abs(er)<=gate)))
 for which in['coarse','fine']:
  c=contract(which);chunks=[json.loads((P/f'total_{which}_part{i}.json').read_text())for i in(range(1)if which=='coarse'else range(2))];rows=[r for q in chunks for r in q['rows']]
  if any(q['contract']!=c or not q['qualified']for q in chunks)or [r['index']for r in rows]!=list(range(c['nu_order'])):raise RuntimeError('Exact complete provenance/unique-node coverage required')
  sums={s:{k:np.zeros(2)for k in['F','B']}for s in['empty','LTE','noneq']};mu=rows[0]['spectrum']['equilibrium_mu_nue'];source=json.loads((ROOT/'hot_electron_capture_moments/results.json').read_text())['rules'][-1]
  for r in rows:
   i=r['index'];v=r['spectrum'];E=r['E'];weight=r['weight']
   validate_node(c,r)
   ck(which+' direct spectral KMS '+str(i),v['log_kms_actual']-v['log_kms_expected'],3e-12);dos=E*E/(2*math.pi**2*S.HC**3);vec=weight*dos*np.array([1.,E])
   for scenario,mur in[('empty',None),('LTE',mu),('noneq',mu-1.2)]:
    f=0. if mur is None else expit((mur-E)/2);sums[scenario]['F']+=vec*v['j']['value']*(1-f);sums[scenario]['B']+=vec*v['chi']['value']*f
  for s in sums:
   ref=source['scenarios'][s]
   for direction,key in[('F','forward'),('B','backward')]:
    for j,observable in enumerate(['number','neutrino_energy']):
     want=ref[key][observable]['value'];got=sums[s][direction][j];ck(which+' vs accepted CM '+s+direction+observable,got/want-1 if want else got,3e-6 if want else 0)
  groups[which]=dict(contract=c,total={s:{k:v.tolist()for k,v in d.items()}for s,d in sums.items()},seconds=sum(q['seconds']for q in chunks))
 for scenario in groups['fine']['total']:
  for direction in['F','B']:
   for i,key in enumerate(['number','neutrino_energy']):
    want=groups['fine']['total'][scenario][direction][i];got=groups['coarse']['total'][scenario][direction][i];ck('global refinement '+scenario+direction+key,got/want-1 if want else got,3e-6 if want else 0)
 result=dict(groups=groups,checks=checks,passed=sum(v['passed']for v in checks),failed=[v for v in checks if not v['passed']],one_helicity_DOS='E^2/(2pi^2HC^3), fm^-3 MeV^-1',scope='selected EC spectral moment reconstruction only; no new source join or state-domain certificate');(P/'total_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=result['failed'],totals=groups['fine']['total']),indent=2))
if __name__=='__main__':
 if sys.argv[1]=='combine':combine()
 else:collect(sys.argv[1],int(sys.argv[2]))
