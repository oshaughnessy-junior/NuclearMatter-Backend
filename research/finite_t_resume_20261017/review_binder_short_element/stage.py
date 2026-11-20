"""One prelocked actual-source Heun stage; independent energy-density variables."""
from pathlib import Path
from dataclasses import asdict
import sys,json,math,hashlib,time,signal
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;R=P.parent;B=R/'free_source_binder';A=R/'binder_short_element';PREP=R/'binder_element_review_preparation';sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'staged/src'))
from nmbackend.reference import free_matter as F,weak_reference as W,_free_matter_sources as S
signal.alarm(89);start=time.monotonic();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();N=int(sys.argv[1]);idx=int(sys.argv[2]);assert N in(2,4)and 0<=idx<2*N
assert sha(A/'MANIFEST.json')=='aa43c8eab89d3492123c022d28a79e80e1be8818558529c5ea1cb6739bacdd79'
am=json.loads((A/'MANIFEST.json').read_text());assert all(sha(A/n)==h for n,h in am.items())
bm=json.loads((B/'MANIFEST.json').read_text());assert all(sha(B/n)==h for n,h in bm.items())
assert sha(PREP/'LOCK.json')=='a647ebe709c66bd0ee92743028f7b8a9efaed4ade18977c811c75e6f1103dee3'
statepath=P/f'trajectory_{N}.json';out=P/f'stage_{N}_{idx}.json';assert not out.exists(),'no overwrite'
h=2e-4/N;n=1e-7
if idx==0:
 assert not statepath.exists();initial=F.state_density(2,n,.3,photons='absent');ctx=dict(N=N,h=h,next_stage=0,initial=asdict(initial),current_e=initial.energy_density,current_Ye=.3,acc=[0.,0.,0.,0.],stage_paths=[],corrected=[])
else:ctx=json.loads(statepath.read_text());assert ctx['next_stage']==idx
if idx%2==0:e,Y=ctx['current_e'],ctx['current_Ye']
else:
 prev=json.loads((P/ctx['stage_paths'][-1]).read_text());e=ctx['current_e']-h*prev['Q'];Y=ctx['current_Ye']+h*prev['R']/n
states={}
def residual(T):
 s=F.state_density(T,n,Y,photons='absent');states[T]=s;return s.energy_density-e
T=brentq(residual,1.9,2.1,xtol=2e-12,rtol=9e-16);s=states.get(T)or F.state_density(T,n,Y,photons='absent');root_error=s.energy_density-e
# Initial exact forward source before any dynamical inference.
source=F.source_moments(s,radiation=W.Radiation.empty(),controls=F.SourceControls.fine());Q=source.neutrino_energy_escape.value;Rnet=source.proton_number_source.value;assert Q is not None and Rnet is not None
root_gate=max(2e-18,1e-8*abs(h*Q));assert abs(root_error)<=root_gate
raw=source.channel_results;grossR=math.fsum(q.value for r in raw for q in(r.forward[0],r.backward[0]));grossQ=math.fsum(q.value for r in raw for q in(r.forward[1],r.backward[1]));nue=raw[0].net[0].value;anti=raw[1].net[0].value+raw[2].net[0].value
aff=s.proton.chemical_potential+s.electron.chemical_potential-s.neutron.chemical_potential;entropy=(-Q-aff*Rnet)/(n*T);entropy_gross=(grossQ+abs(aff)*grossR)/(n*T)
if idx==0:
 anchor=json.loads((B/'source_0.json').read_text());assert abs(Rnet/anchor['R']['value']-1)<=3e-6 and abs(Q/anchor['Q']['value']-1)<=3e-6
record=dict(N=N,index=idx,t_s=(idx//2+(idx%2))*h,role='current corrected state'if idx%2==0 else'predictor endpoint, not corrected endpoint',state=asdict(s),target_e=e,root_residual=root_error,root_gate=root_gate,source=S._payload(source),R=Rnet,Q=Q,nue=nue,anti=anti,grossR=grossR,grossQ=grossQ,entropy_specific_dot=entropy,entropy_gross=entropy_gross,source_identity=source.identity,binder_manifest_sha256=sha(B/'MANIFEST.json'),author_manifest_sha256=sha(A/'MANIFEST.json'),preparation_lock_sha256=sha(PREP/'LOCK.json'),seconds=time.monotonic()-start)
out.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');ctx['stage_paths'].append(out.name);ctx['next_stage']=idx+1
if idx%2:
 prev=json.loads((P/ctx['stage_paths'][-2]).read_text());ctx['current_e']-=h*(prev['Q']+Q)/2;ctx['current_Ye']+=h*(prev['R']+Rnet)/(2*n)
 for j,k in enumerate(['Q','nue','anti','entropy_specific_dot']):ctx['acc'][j]+=h*(prev[k]+record[k])/2
 Y=ctx['current_Ye'];e=ctx['current_e'];states={};Tend=brentq(residual,1.9,2.1,xtol=2e-12,rtol=9e-16);end=states.get(Tend)or F.state_density(Tend,n,Y,photons='absent')
 err=end.energy_density-e;assert abs(err)<=max(2e-18,1e-8*abs(h*Q));ctx['corrected'].append(dict(step=idx//2+1,t_s=(idx//2+1)*h,target_e=e,state=asdict(end),root_residual=err,acc=ctx['acc'].copy(),source_at_this_corrected_endpoint_not_claimed=True))
ctx['complete']=idx==2*N-1;statepath.write_text(json.dumps(ctx,indent=2,allow_nan=False)+'\n')
assert all(sha(A/n)==h for n,h in am.items());assert all(sha(B/n)==h for n,h in bm.items());print(N,idx,'complete-stage','T',T,'Ye',s.Ye,'seconds',time.monotonic()-start);signal.alarm(0)
