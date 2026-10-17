"""Bounded physical PC spectral node collection; no interpolation/rate cache API."""
from pathlib import Path
import importlib.util,sys,json,hashlib,time,math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp,log_expit
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sp=importlib.util.spec_from_file_location('hotPC_total_pure_function',P/'spectrum.py');S=importlib.util.module_from_spec(sp);sys.modules[sp.name]=S;sp.loader.exec_module(S)
EDGES=[0.,1.5,1.8060662080187608,2.2,4.,8.,16.,32.,60.]
def contract(rule):
 if rule not in['coarse','middle','fine']:raise ValueError('Declaredrule')
 n={'coarse':4,'middle':6,'fine':8}[rule];x,w=leggauss(n);E=[];W=[]
 for lo,hi in zip(EDGES[:-1],EDGES[1:]):E.extend(lo+(x+1)*(hi-lo)/2);W.extend(w*(hi-lo)/2)
 controls=[8,12,12,4,60.,35.]if rule=='coarse'else[10,16,16,4,75.,45.]
 return dict(rule=rule,incident_panel_edges=EDGES,order_per_panel=n,nodes=list(map(float,E)),weights=list(map(float,W)),controls=controls,spectrum_sha=hashlib.sha256((P/'spectrum.py').read_bytes()).hexdigest(),sources=S.PINS,eventreference_sha=hashlib.sha256((P/'FRESH_EVENT_REFERENCE.json').read_bytes()).hexdigest())
def collect(rule,part):
 c=contract(rule);N=len(c['nodes']);ids=list(range(N))if rule=='coarse'else list(range(part*16,(part+1)*16))
 if rule=='coarse'and part!=0 or rule!='coarse'and not 0<=part<(len(c['nodes'])//16):raise ValueError('exactboundedchunk')
 outpath=P/f'total_{rule}_{part}.json'
 if outpath.exists():raise RuntimeError('Frozencompletedphysicalnodes mustnotbeoverwritten')
 if any(not x['passed']for x in json.loads((P/'point_results.json').read_text())['checks']):raise RuntimeError('absoluteanchor gate')
 rows=[];start=time.monotonic();n,a,e,phi,t,ot=c['controls']
 for i in ids:
  if time.monotonic()-start>100:raise TimeoutError('PCspectralchunk100s budget; partial retained')
  r=S.integrate(c['nodes'][i],lepton_order=n,angle_order=a,energy_order=e,phi_order=phi,tail=t,outer_tail=ot,budget_seconds=15.);rows.append(dict(index=i,E=c['nodes'][i],weight=c['weights'][i],spectrum=r));(P/f'total_{rule}_{part}_partial.json').write_text(json.dumps(dict(contract=c,rows=rows,qualified=False),indent=2)+'\n')
 if contract(rule)!=c:raise RuntimeError('sourcecontrols changed')
 out=dict(contract=c,rows=rows,qualified=True,seconds=time.monotonic()-start);outpath.write_text(json.dumps(out,indent=2)+'\n');print(rule,part,'completed',len(rows),'seconds',out['seconds'])
def validate_row(c,row,bath,meta):
 idx=row['index'];v=row['spectrum']
 if type(idx)is not int or not 0<=idx<len(c['nodes']):raise RuntimeError('literal validphysicalnode index required')
 n,a,e,ph,t,ot=c['controls'];wanted=dict(lepton_order=n,angle_order=a,energy_order=e,phi_order=ph,tail=t,outer_tail=ot)
 if row['E']!=c['nodes'][idx]or row['weight']!=c['weights'][idx]or v['Enu']!=row['E']or v['controls']!=wanted or v['bath']!=vars(bath)or v['matched_state']!=meta or v['implementation_sha256']!=c['spectrum_sha']or v['source_pins']!=S.PINS:raise RuntimeError('node physicalmetadata mismatch')
 for key in ['j','chi']:
  coeff=v[key];log=coeff['log']
  if log is not None and(type(log)not in(int,float)or not math.isfinite(log)):raise RuntimeError('invalid physical log')
  if coeff!=S.represented(log):raise RuntimeError('positive/log representation mismatch')
 eq=bath.mu_n-bath.mu_p-bath.mu_e
 expect=v['j']['log']-v['chi']['log']if v['j']['log']is not None and v['chi']['log']is not None else None
 relaxation=v['j']['value']+v['chi']['value']if v['j']['representable']and v['chi']['representable']else None
 if v['equilibrium_mu_antinue']!=eq or v['log_kms_expected']!=-(row['E']-eq)/bath.T or v['log_kms_actual']!=expect or v['relaxation']!=relaxation:raise RuntimeError('derivedoperator metadata mismatch')
 if v['units']!='j and true chi: s^-1 per one active antinue occupation; relaxation=j+chi'or v['channel']!='n+e+ <-> p+antinue ONLY; no joined inventory':raise RuntimeError('units/channel mismatch')

def combine():
 groups={};checks=[]
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 ref=json.loads((P/'FRESH_EVENT_REFERENCE.json').read_text());b,meta=S.matched();mu=b.mu_n-b.mu_p-b.mu_e
 for rule in ['coarse','middle','fine']:
  c=contract(rule);chunks=[json.loads((P/f'total_{rule}_{p}.json').read_text())for p in([0]if rule=='coarse'else range(len(c['nodes'])//16))];rows=[r for ch in chunks for r in ch['rows']]
  if any(ch['qualified']is not True or ch['contract']!=c for ch in chunks)or [r['index']for r in rows]!=list(range(len(c['nodes']))):raise RuntimeError('exactcompleteunique physicalnode coverage/provenance required')
  sums={}
  for si,(name,mur)in enumerate([('empty',None),('LTE',mu),('noneq',mu-.8)]):
   logs={'F':[[],[]],'B':[[],[]]}
   for row in rows:
    validate_row(c,row,b,meta);v=row['spectrum'];E=row['E'];idx=row['index'];n,a,e,phi,t,ot=c['controls'];controls=dict(lepton_order=n,angle_order=a,energy_order=e,phi_order=phi,tail=t,outer_tail=ot)
    if E!=c['nodes'][idx]or row['weight']!=c['weights'][idx]or v['bath']!=vars(b)or v['matched_state']!=meta or v['source_pins']!=S.PINS or v['implementation_sha256']!=c['spectrum_sha']or v['controls']!=controls:raise RuntimeError('nodephysicalstate/source/control mismatch')
    if v['j']['log']is not None and v['chi']['log']is not None:gate(f'{rule} separateKMS {idx} {name}',v['log_kms_actual']-v['log_kms_expected'],3e-12)
    base=math.log(row['weight'])+2*math.log(E)-math.log(2*math.pi**2*S.HC**3)
    for key,logocc in [('F',0. if mur is None else float(log_expit((E-mur)/2))),('B',None if mur is None else float(log_expit((mur-E)/2)))]:
     coeff=v['j']if key=='F'else v['chi']
     if logocc is None or coeff['log']is None:continue
     for j in [0,1]:logs[key][j].append(base+coeff['log']+logocc+j*math.log(E))
   values={key:[float(math.exp(logsumexp(v)))if v else 0. for v in pair]for key,pair in logs.items()}
   for direction,di in [('F',0),('B',1)]:
    for j,observable in enumerate(['number','neutrino_energy']):
     want=ref['sums'][si][di][j];got=values[direction][j];gate(f'{rule} actualCM {name}{direction}{observable}',got/want-1 if want else got,3e-6 if want else 0.)
   sums[name]=values
  groups[rule]=dict(total=sums,seconds=sum(ch['seconds']for ch in chunks),contract=c)
 for name in groups['fine']['total']:
  for direction in ['F','B']:
   for j in [0,1]:
    a=groups['coarse']['total'][name][direction][j];b=groups['fine']['total'][name][direction][j];gate(f'globalrefinement {name}{direction}{j}',a/b-1 if b else a,3e-6 if b else 0.)
 for name in groups['fine']['total']:
  for direction in ['F','B']:
   for j in [0,1]:
    a=groups['middle']['total'][name][direction][j];b=groups['fine']['total'][name][direction][j];gate(f'qualifiedincidentrefinement {name}{direction}{j}',a/b-1 if b else a,3e-6 if b else 0.)
 out=dict(groups=groups,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],DOS='ONEantinuhelicity E²/(2pi²HC³),fm^-3MeV^-1');(P/'total_results.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(passed=out['passed'],failed=out['failed'],fine=groups['fine']['total']))
if __name__=='__main__':
 if sys.argv[1]=='combine':combine()
 else:collect(sys.argv[1],int(sys.argv[2]))
