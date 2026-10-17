"""Assessment adapter; reuses frozen static allowed integrals without a new kernel."""
from pathlib import Path
import importlib.util,sys,hashlib,json,math,time
import numpy as np
from scipy.integrate import quad
from scipy.special import expit,gamma,zeta
from scipy.optimize import brentq
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
 s=importlib.util.spec_from_file_location(name,R/path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
S=load('allowed_assessment_existing_static','weak_rates/static_capture.py');C=load('allowed_assessment_actual_free_eos','hot_electron_capture_moments/electron_capture.py')
CHANNELS={'EC':'electron_capture','PC':'positron_capture','beta':'neutron_decay'};Q=C.MN-C.MP
PINS={f:sha(R/f)for f in ['weak_rates/static_capture.py','weak_rates/TECHNICAL_NOTE.md','transparent_capture_balance/results.json','transparent_capture_balance/balance.py','transparent_balance_grid/grid.py','transparent_balance_grid/MANIFEST.json','transparent_balance_grid/state_0.json','transparent_balance_grid/state_1.json','transparent_balance_grid/state_2.json','hot_electron_capture_moments/electron_capture.py','relativistic_capture_oracle/events.py','relativistic_free_eos/free_baryons.py','thermodynamics/lepton_reference.py']}
def verify():
 for f,h in PINS.items():
  if sha(R/f)!=h:raise RuntimeError('Frozen source/data changed: '+f)
def rates(T,mu,nn,np_,order=96,tail=70.):
 return {k:(np_ if k=='EC'else nn)*S.target_moments(T,mu,name,order=order,tail=tail,q=Q)for k,name in CHANNELS.items()}
def momentum_reference(T,mu,ch):
 # Independent p/T integration: E p dE = p² dp. Infinite capture interval.
 lower=math.sqrt(max(Q*Q-S.ME*S.ME,0))if ch=='EC'else 0.
 upper=math.sqrt(Q*Q-S.ME*S.ME)if ch=='beta'else np.inf
 def f(z,power):
  p=T*z;E=math.hypot(p,S.ME);nu=E-Q if ch=='EC'else E+Q if ch=='PC'else Q-E
  block=expit((mu-E)/T)if ch=='EC'else expit((-mu-E)/T)if ch=='PC'else expit((E-mu)/T)
  return p*p*T*nu**power*block
 return S.RATE_PREFACTOR*np.array([quad(lambda z:f(z,power),lower/T,upper/T,epsabs=1e-13,epsrel=2e-11,limit=200)[0]for power in [2,3]])
def run():
 start=time.monotonic();numerics=[];model=[];rows=[]
 def ck(n,e,t):numerics.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 def save():
  o=dict(numerical_checks=numerics,model_assessment_checks=model,rows=rows,seconds=time.monotonic()-start,pins=PINS);(P/'partial.json').write_text(json.dumps(o,indent=2)+'\n')
 verify();ck('GF unchanged',S.GF/C.E.GF-1,0);ck('Vud unchanged',S.VUD/C.E.VUD-1,0);ck('me unchanged',S.ME/C.ME-1,0)
 ck('zero-transfer GV/gA physical current',np.max(abs(C.E.vertices(np.zeros((1,4)),C.MW)[0]-(C.E.G-S.GA*(C.E.G@C.E.G5)))),2e-15)
 # Closed analytic vacuum allowed Fermi phase-space integral, no calibration.
 w=Q/S.ME;f=(math.sqrt(w*w-1)*(2*w**4-9*w*w-8)+15*w*math.acosh(w))/60
 exact=S.RATE_PREFACTOR*S.ME**5*f;vac=S.target_moments(.01,-20.,'neutron_decay',order=192,q=Q)
 ck('analytic vacuum allowed absolute width',vac[0]/exact-1,2e-12)
 independent=momentum_reference(.01,-20.,'beta');ck('vacuum energy independent momentum',vac[1]/independent[1]-1,2e-12)
 for T in [1.,3.]:
  fd=lambda k:(1-2**(-k))*gamma(k+1)*zeta(k+1,1.)
  val=S.target_moments(T,0.,'electron_capture',order=192,mass=0.,q=0.)
  for k in [0,1]:ck('massless Q0 analytic FD '+str((T,k)),val[k]/(S.RATE_PREFACTOR*T**(5+k)*fd(4+k))-1,3e-12)
 save()
 if not all(c['passed']for c in numerics):raise ArithmeticError('Forward absolute normalization gate failed before assessment')
 inputs=[('original_anchor',json.loads((R/'transparent_capture_balance/results.json').read_text())['balanced'])]
 for i in range(3):
  d=json.loads((R/f'transparent_balance_grid/state_{i}.json').read_text());assert d['status']=='AUTHOR_SELECTED_PASS';inputs.append((f'new_state_{i}',d['balanced']))
 for label,d in inputs:
  T=d.get('T',2.);nB=d.get('nB',1e-7);y=d['Ye'];b,st=C.matched(T,nB,y)
  ck(label+' fresh mue',b.mu_e-d['bath']['mu_e'],2e-12)
  ck(label+' actual neutron root',st['neutron']['density']/(nB*(1-y))-1,1e-10);ck(label+' actual proton root',st['proton']['density']/(nB*y)-1,1e-10);ck(label+' actual pair netcharge',st['electron']['net_density']/(nB*y)-1,1e-10)
  base=rates(T,b.mu_e,nB*(1-y),nB*y);fine=rates(T,b.mu_e,nB*(1-y),nB*y,192,90.)
  row=dict(label=label,T=T,nB=nB,detailed_kinetic_Ye=y,detailed_affinity=d['chemical_affinity'],actual_bath=vars(b),actual_EOS=st,channels={},source='Immutable actual physical detailed moments, arithmetic comparison; no detailed multidimensional rerun.')
  for ch in CHANNELS:
   ref=momentum_reference(T,b.mu_e,ch)*(nB*y if ch=='EC'else nB*(1-y))
   for k,name in enumerate(['number','neutrino_energy']):
    ck(label+' independent momentum '+ch+name,base[ch][k]/ref[k]-1,3e-10);ck(label+' order/tail '+ch+name,base[ch][k]/fine[ch][k]-1,3e-10)
   full=np.array([d['channels'][ch]['number'],d['channels'][ch]['neutrino_energy']]);err=base[ch]/full-1
   row['channels'][ch]=dict(allowed=base[ch].tolist(),detailed=full.tolist(),relative_error=err.tolist(),momentum_quad=ref.tolist(),refined=fine[ch].tolist())
   for k,name in enumerate(['number','energy']):model.append(dict(name=label+' 1percent '+ch+name,error=float(err[k]),gate=.01,passed=bool(abs(err[k])<=.01)))
  net=base['beta'][0]+base['PC'][0]-base['EC'][0];gross=sum(v[0]for v in base.values());row.update(allowed_source_at_detailed_root=net,allowed_dYe_dt_at_detailed_root=net/nB,allowed_net_over_gross=net/gross)
  # Fresh charge-neutral lepton/EOS roots at every approximate composition trial.
  trials=[]
  def fun(yy):
   bb,ss=C.matched(T,nB,yy);rr=rates(T,bb.mu_e,nB*(1-yy),nB*yy);v=rr['beta'][0]+rr['PC'][0]-rr['EC'][0]
   trials.append(dict(Ye=yy,bath=vars(bb),rates={k:z.tolist()for k,z in rr.items()},net=v));return v
  root=brentq(fun,.05,.9,xtol=1e-12);rb,rs=C.matched(T,nB,root);rv=rates(T,rb.mu_e,nB*(1-root),nB*root);rref=rates(T,rb.mu_e,nB*(1-root),nB*root,192,90.)
  for ch in CHANNELS:
   for k in range(2):ck(label+' approximate root refine '+str((ch,k)),rv[ch][k]/rref[ch][k]-1,3e-10)
  rn=rv['beta'][0]+rv['PC'][0]-rv['EC'][0];ck(label+' approximate actual-rate root residual',rn/sum(v[0]for v in rv.values()),1e-9)
  row.update(allowed_kinetic_Ye=root,allowed_root_affinity=rb.mu_n-rb.mu_p-rb.mu_e,delta_Ye=root-y,delta_affinity=(rb.mu_n-rb.mu_p-rb.mu_e)-d['chemical_affinity'],allowed_root_trials=trials,allowed_root_rates={k:v.tolist()for k,v in rv.items()},allowed_root_net=rn,allowed_root_escaping_energy=sum(v[1]for v in rv.values()))
  model.append(dict(name=label+' illustrative deltaYe1e-3',error=root-y,gate=.001,passed=bool(abs(root-y)<=.001)))
  rows.append(row);save()
 verify();out=dict(numerical_status='PASS'if all(c['passed']for c in numerics)else'FAIL',numerical_checks=numerics,numerical_passed=sum(c['passed']for c in numerics),numerical_failed=[c for c in numerics if not c['passed']],model_assessment_checks=model,model_assessment_failed=[c for c in model if not c['passed']],rows=rows,vacuum=dict(allowed_width_s1=exact,lifetime_s=1/exact,neutrino_energy_rate_MeV_s=vac[1]),pins=PINS,seconds=time.monotonic()-start,independent_review='PENDING',grid_source_acceptance='CONDITIONAL on different review; frozen AUTHOR_SELECTED_PASS raw used read-only')
 (P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(numerical=out['numerical_status'],passed=out['numerical_passed'],failed=out['numerical_failed'],model_failures=len(out['model_assessment_failed']),seconds=out['seconds'],roots=[dict(label=r['label'],deltaYe=r['delta_Ye'])for r in rows])))
if __name__=='__main__':run()
