"""Fresh original forward, changed endpoint quadrature/tails and closure ledgers."""
from pathlib import Path
import json,sys,importlib.util,hashlib,math
P=Path(__file__).parent;ROOT=P.parent;s=importlib.util.spec_from_file_location('three_source_endpoint_client',P/'client.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M);C=M.Client();checks=[]
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
x=json.loads((P/'results.json').read_text());final=x['rows'][-1]['final'];initial=C.evaluate(2.,.3,action='source');review=json.loads((ROOT/'review_transparent_capture_balance/CHECKPOINT.json').read_text());refpath='transparent_capture_balance/results.json';refbytes=(ROOT/refpath).read_bytes()
if hashlib.sha256(refbytes).hexdigest()!=review['frozen_dependencies'][refpath]:raise RuntimeError('Accepted original source reference changed')
ref=json.loads(refbytes);print('reference rowkeys',ref['rows'][0].keys(),flush=True)
# Rows save coarse/refined selected actual source evaluations.
rr=ref['rows'][0]
if 'coarse'in rr:refsource=rr['coarse']
elif 'base'in rr:refsource=rr['base']
else:refsource=rr
if refsource['Ye']!=.3:raise RuntimeError('Original source reference must exactly match Ye.3')
for k in['beta','PC','EC']:
 for obs,v in initial['sources']['channels'][k].items():ck('fresh initial forward '+k+obs,v/refsource['channels'][k][obs]-1,3e-6)
endpoint={}
for profile in['base','fine','tail']:
 r=C.evaluate(final['T'],final['Ye'],action='source',profile=profile);endpoint[profile]=r;print('endpoint',profile,r['sources']['R'],r['sources']['Q'],r['elapsed'],flush=True)
 for k,v in r['sources']['density_residuals'].items():ck('changed FD incoming density '+profile+k,v,3e-8)
for a,b in[('base','fine'),('fine','tail')]:
 for k in['beta','PC','EC']:
  for obs,v in endpoint[a]['sources']['channels'][k].items():ck('changed '+a+'to'+b+' '+k+obs,v/endpoint[b]['sources']['channels'][k][obs]-1,3e-6)
q0=initial['sources'];q1=endpoint['fine']['sources'];ck('actual source changes',0 if abs(q1['R']/q0['R']-1)>.01 and abs(q1['Q']/q0['Q']-1)>.005 else 1,0)
# Integrals reconstructed from literal numerical stage weights, independently of
# the trajectory's accumulator: no equality against telescoping u alone.
for traj in x['rows']:
 weights=[]
 for rec in traj['records']:
  dt=traj['duration']/traj['steps'];weights.append((dt if traj['method']=='Euler'else dt/2,rec['source']))
  if traj['method']=='Heun':weights.append((dt/2,rec['second_source']))
 I={k:{obs:sum(w*z['channels'][k][obs]for w,z in weights)for obs in['number','neutrino_energy']}for k in['beta','PC','EC']};Q=sum(v['neutrino_energy']for v in I.values());anti=I['beta']['number']+I['PC']['number'];nu=I['EC']['number'];tag=str((traj['method'],traj['steps']));ck('independent stage escape energy '+tag,Q/traj['escaped_energy']-1,3e-12);ck('independent stage composition '+tag,(1e-7*(traj['final']['Ye']-.3)-(anti-nu))/(anti+nu),2e-8)
 e0=traj['initial'];ef=traj['final'];nonph_delta=sum(ef['material'][k]['energy_density']-e0['material'][k]['energy_density']for k in['neutron','proton','electron']);ph_delta=ef['photon']['energy_density']-e0['photon']['energy_density'];ck('separate photon+material escape budget '+tag,(nonph_delta+ph_delta+Q)/Q,2e-8)
raw=dict(checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],fresh_initial=initial,endpoint=endpoint,reference_balance_results_sha=hashlib.sha256(refbytes).hexdigest(),source_change=dict(R_relative=q1['R']/q0['R']-1,Q_relative=q1['Q']/q0['Q']-1),client_calls=C.calls);(P/'endpoint_results.json').write_text(json.dumps(raw,indent=2)+'\n');print(json.dumps(dict(passed=raw['passed'],failed=raw['failed'],change=raw['source_change']),indent=2))
