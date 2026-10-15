"""Independent raw-panel ledgers, changed-state spectra and mechanical witness."""
from pathlib import Path
import sys,json,math,time,hashlib
from dataclasses import asdict
import numpy as np
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parent.parent;HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'fd_density_feedback'))
import engine as A
start=time.monotonic();checks=[];details={}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
reports=[json.loads((ROOT/f'fd_density_feedback/results_{k}.json').read_text())for k in(2,4)]
for r in reports:
 E=L=W=entropy=0.
 for k,(row,source)in enumerate(zip(r['rows'],r['sources'])):
  req,result=source['request'],source['result'];n=row['before']['nB'];dt=row['dt']
  sums={fl:np.sum([p['values']for p in result['panels']if p['flavor']==fl],axis=0)for fl in('nue','antinue')}
  F=(sums['antinue'][0]-sums['nue'][0])/n;q=(sums['nue'][1]+sums['antinue'][1])/n
  gate(f'raw panels sourceF {r["steps"]}:{k}',F/row['F']-1,3e-14)
  gate(f'raw panels sourceq {r["steps"]}:{k}',q/row['q']-1,3e-14)
  E+=q*dt;L-=F*dt;W+=row['work_pressure'];entropy+=row['weak_entropy_integral']
  gate(f'raw energy ledger {r["steps"]}:{k}',row['after']['u']+E-W-r['initial']['u'],3e-9)
  gate(f'raw lepton ledger {r["steps"]}:{k}',row['after']['Ye']+L-r['initial']['Ye'],3e-13)
  gate(f'raw entropy ledger {r["steps"]}:{k}',row['after']['entropy']-r['initial']['entropy']-entropy,3e-9)
  gate(f'request actualstate {r["steps"]}:{k}',max(abs(req[field]-row['before'][field])for field in('T','nB','Ye')),0.)
  gate(f'request fingerprint {r["steps"]}:{k}',0. if A.digest(req)==source['fingerprint']else 1.,0.)
  expected='full fingerprint verified initial physical source'if k==0 else'fresh full physical spectral quadrature'
  gate(f'exactinitial only reuse {r["steps"]}:{k}',0. if source['origin']==expected else 1.,0.)
 gate(f'signed compression expansion {r["steps"]}',r['compression_work']+r['expansion_work']-r['mechanical_work'],2e-14)

# Independent reversible mechanical path for one actual compression substep.
row=reports[1]['rows'][1];kick=row['weak_kick'];end=row['after'];T,n,Y=kick['T'],kick['nB'],kick['Ye']
before=A.density_eos.state(T,n,Y);S=before['entropy_per_baryon'];extent=math.log(end['nB']/n);pts=[]
for x in(0.,extent/2,extent):
 nn=n*math.exp(x)
 temp=T if x==0 else brentq(lambda t:A.density_eos.state(t,nn,Y)['entropy_per_baryon']-S,T*.98,T*1.03,xtol=2e-13,rtol=2e-13)
 st=A.density_eos.state(temp,nn,Y);pts.append(dict(x=x,n=nn,T=temp,u=st['u_MeV_per_baryon'],P=st['P_MeV_fm3'],entropy=st['entropy_per_baryon']))
W=extent/6*(pts[0]['P']/pts[0]['n']+4*pts[1]['P']/pts[1]['n']+pts[2]['P']/pts[2]['n'])
gate('independent Simpson mechanicalwork',W/row['work_pressure']-1,2e-8)
gate('independent entropyroot endpoint',pts[-1]['T']/end['T']-1,3e-11)
gate('independent path work energy',(pts[-1]['u']-pts[0]['u'])/W-1,2e-8)
details['mechanical']=dict(points=pts,Simpson_work=W,author_Gauss_work=row['work_pressure'])

# New spectral energy at a genuinely changed author peak-density state.
peak=max((r['before']for r in reports[1]['rows']),key=lambda p:p['nB'])
st=A.density_eos.state(peak['T'],peak['nB'],peak['Ye']);m=st['matter'];z=m.proton.mu+m.electron.chemical_potential-m.neutron.mu
details['physical_spectral']={}
for flavor,sigma in(('nue',1.),('antinue',-1.)):
 c=A.element.physics.coefficients(m,flavor,7.3,order=48,tail=40.)
 d=A.element.physics.coefficients(m,flavor,7.3,order=64,tail=50.)
 for key in('log_j','log_chi'):gate('new peak E7.3 '+flavor+key,math.expm1(c[key]-d[key]),3e-6)
 gate('peak EOS KMS '+flavor,c['log_j']-c['log_chi']+(7.3-sigma*z)/peak['T'],3e-10)
 details['physical_spectral'][flavor]=dict(state=peak,coarse=c,fine=d)

# Reconstruct exact full state and provenance of the actual changed request.
source=reports[1]['sources'][2]
gate('new request full solved FDstate',0. if asdict(m)==source['request']['full_fd_state']else 1.,0.)
gate('physical provenance still frozen',0. if A.physical_provenance()==reports[1]['provenance']else 1.,0.)
two,four=reports
details['time_sensitivity']=dict(escaped_energy_relative=two['escaped_energy_per_baryon']/four['escaped_energy_per_baryon']-1,escaped_lepton_relative=two['escaped_electron_lepton_per_baryon']/four['escaped_electron_lepton_per_baryon']-1,temperature_absolute=two['final']['T']-four['final']['T'])
out=dict(passed=all(x['passed']for x in checks),checks=checks,details=details,runtime_seconds=time.monotonic()-start,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in[Path(__file__),ROOT/'fd_density_feedback/engine.py',ROOT/'fd_density_feedback/run_feedback.py',ROOT/'fd_density_feedback/results_2.json',ROOT/'fd_density_feedback/results_4.json']})
(HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),seconds=out['runtime_seconds'],failed=[x for x in checks if not x['passed']]),indent=2))
if not out['passed']:raise SystemExit(1)
