"""Declared physical beta spectrum quadrature and one-helicity moment audit."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import log_expit,logsumexp
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True
s=importlib.util.spec_from_file_location('hotbeta_total_actual_function',P/'spectrum.py');S=importlib.util.module_from_spec(s);sys.modules[s.name]=S;s.loader.exec_module(S)
EDGES=[0.,.2,.5,.7,S.ENDPOINT,.9,1.05,1.2,1.4]
def contract(rule):
 if rule not in['coarse','fine']:raise ValueError('declaredrule')
 order=6 if rule=='coarse'else 10;controls=[8,10,10,6,50.]if rule=='coarse'else[12,16,16,10,70.];x,w=leggauss(order);E=[];W=[]
 for lo,hi in zip(EDGES[:-1],EDGES[1:]):E.extend(lo+(x+1)*(hi-lo)/2);W.extend(w*(hi-lo)/2)
 return dict(rule=rule,incident_order_per_panel=order,edges=EDGES,nodes=list(map(float,E)),weights=list(map(float,W)),controls=controls,spectrum_sha=hashlib.sha256((P/'spectrum.py').read_bytes()).hexdigest(),sources=S.PINS,reference_hot_total_sha=S.HOT_RESULTS_SHA)
def validate(c,r,b,meta):
 i=r['index'];v=r['spectrum']
 if type(i)is not int or not 0<=i<len(c['nodes']):raise RuntimeError('literalnodeindex')
 p,co,e,phi,t=c['controls'];wanted=dict(parent_order=p,cone_order=co,electron_order=e,phi_order=phi,tail=t)
 if r['E']!=c['nodes'][i]or r['weight']!=c['weights'][i]or v['Enu']!=r['E']or v['bath']!=vars(b)or v['matched_state']!=meta or v['source_pins']!=S.PINS or v['implementation_sha256']!=c['spectrum_sha']or v['controls']!=wanted:raise RuntimeError('physicalnodeprovenance')
 for k in ['j','chi']:
  log=v[k]['log']
  # Every selectedfiniteTintegratedbeta node has strictlypositive support.
  # Ordinaryoutputunderflow allowed; falseempty-support evidence forbidden.
  if type(log)not in[int,float]or not math.isfinite(log)or v[k]!=S.represented(log):raise RuntimeError('selectedpositive log/ordinary physicalcoefficient mismatch')
 if v['units']!='j and truechi s^-1 peroneactive antinueoccupation; DOS E²/(2pi²HC³)'or v['channel']!='n <-> p+e-+antinue ONLY; noPCjoin':raise RuntimeError('channel/units')
 eq=b.mu_n-b.mu_p-b.mu_e
 if v['equilibrium_mu_antinue']!=eq or v['log_kms_expected']!=-(r['E']-eq)/b.T or v['log_kms_actual']!=v['j']['log']-v['chi']['log']:raise RuntimeError('derivedoperator metadata')
def collect(rule,part):
 c=contract(rule);N=len(c['nodes']);ids=list(range(N))if rule=='coarse'else list(range(part*40,(part+1)*40));path=P/f'total_{rule}_{part}.json'
 if path.exists():raise RuntimeError('completedphysicalevidence cannotbeoverwritten')
 if rule=='coarse'and part!=0 or rule=='fine'and part not in[0,1]:raise ValueError('boundedexactchunk')
 pilot=json.loads((P/'pilot_results.json').read_text())
 if pilot['failed']:raise RuntimeError('absolutephasecurrentanchors notqualified')
 start=time.monotonic();rows=[];b,meta=S.matched();p,co,e,phi,t=c['controls']
 for i in ids:
  if time.monotonic()-start>100:raise TimeoutError('physicalbeta spectralchunk100s budget')
  v=S.integrate(c['nodes'][i],parent_order=p,cone_order=co,electron_order=e,phi_order=phi,tail=t,budget_seconds=10.);r=dict(index=i,E=c['nodes'][i],weight=c['weights'][i],spectrum=v);validate(c,r,b,meta);rows.append(r);(P/f'total_{rule}_{part}_partial.json').write_text(json.dumps(dict(contract=c,rows=rows,qualified=False),indent=2)+'\n')
 if contract(rule)!=c:raise RuntimeError('midbatchsourcechange')
 path.write_text(json.dumps(dict(contract=c,rows=rows,qualified=True,seconds=time.monotonic()-start),indent=2)+'\n');print(rule,part,len(rows),time.monotonic()-start)
def combine():
 checks=[];groups={};b,meta=S.matched();mu=b.mu_n-b.mu_p-b.mu_e;ref=json.loads((R/'hot_beta_moments/results.json').read_text())['rows'][-1]['result']['scenarios']
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 for rule in ['coarse','fine']:
  c=contract(rule);chunks=[json.loads((P/f'total_{rule}_{i}.json').read_text())for i in([0]if rule=='coarse'else[0,1])];rows=[r for ch in chunks for r in ch['rows']]
  if any(ch['qualified']is not True or ch['contract']!=c for ch in chunks)or [r['index']for r in rows]!=list(range(len(c['nodes']))):raise RuntimeError('completeuniquephysicalcoverage')
  out={}
  for name,mur in [('empty',None),('LTE',mu),('emission_dominated',mu-1.6),('inverse_dominated',mu+1.6)]:
   logs={'F':[[],[]],'B':[[],[]]}
   for row in rows:
    validate(c,row,b,meta);v=row['spectrum'];E=row['E'];base=math.log(row['weight'])+2*math.log(E)-math.log(2*math.pi**2*S.HC**3);gate(f'{rule} separateKMS {row["index"]} {name}',v['log_kms_actual']-v['log_kms_expected'],3e-12)
    for key,occ in [('F',0. if mur is None else float(log_expit((E-mur)/2))),('B',None if mur is None else float(log_expit((mur-E)/2)))]:
     if occ is None:continue
     coeff=v['j']if key=='F'else v['chi']
     for j in[0,1]:logs[key][j].append(base+coeff['log']+occ+j*math.log(E))
   vals={k:[math.exp(float(logsumexp(v)))if v else 0. for v in pair]for k,pair in logs.items()}
   for direction,key in [('F','forward'),('B','backward')]:
    for i,observable in enumerate(['number','neutrino_energy']):
     want=ref[name][key][observable]['value'];got=vals[direction][i];gate(f'{rule} wholeevent {name}{direction}{observable}',got/want-1 if want else got,3e-6 if want else 0.)
   out[name]=vals
  groups[rule]=dict(total=out,seconds=sum(ch['seconds']for ch in chunks),contract=c)
 for name in groups['fine']['total']:
  for direction in ['F','B']:
   for j in[0,1]:
    a=groups['coarse']['total'][name][direction][j];b=groups['fine']['total'][name][direction][j];gate(f'globalrefinement {name}{direction}{j}',a/b-1 if b else a,3e-6 if b else 0.)
 out=dict(groups=groups,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],scope='selectedwhole beta/inverse3body only; noPCjoin');(P/'total_results.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(passed=out['passed'],failed=out['failed'],fine=groups['fine']['total']))
if __name__=='__main__':
 if sys.argv[1]=='combine':combine()
 else:collect(sys.argv[1],int(sys.argv[2]))
