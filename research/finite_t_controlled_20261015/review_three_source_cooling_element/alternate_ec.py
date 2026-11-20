"""Fresh alternate invariant-s/Dirac-trace EC at changed cooling endpoint."""
from common import *
I=load('independent_cooling_EC_phase_reference',ROOT/'review_hot_electron_capture/probe.py')
f=json.loads((P/'forward.json').read_text())['rows'][-1];b=eos(f['independent']['T'],f['independent']['Ye'])['bath'];bath=I.C.Bath(**b);checks=[];rows=[];start=time.monotonic();before=frozen()
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
for n,a,p in [(24,10,8),(32,12,10)]:
 v=I.integrate(bath,n,a,p,70.);rows.append(v);ck('changed endpoint independent proton density '+str(n),v['density']/(NB*f['independent']['Ye'])-1,3e-8)
 for k,value in zip(I.C.NAMES,v['scenarios']['empty']['net']):ck('changed endpoint trace-vs-scalar '+str(n)+' '+k,value/f['packet']['sources']['channels']['EC'][k]-1,3e-6)
 print(n,v['seconds'],flush=True)
for k,a,bv in zip(I.C.NAMES,rows[0]['scenarios']['empty']['net'],rows[1]['scenarios']['empty']['net']):ck('alternate trace refinement '+k,a/bv-1,3e-6)
ck('all frozen unchanged',int(before!=frozen()),0)
(P/'alternate_ec.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,seconds=time.monotonic()-start),indent=2)+'\n');print(len(checks),[c for c in checks if not c['passed']],time.monotonic()-start)
