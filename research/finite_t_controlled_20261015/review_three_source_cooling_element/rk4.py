"""Own conservative RK4 with independently bracketed infinite-momentum EOS."""
from common import *
if __name__=='__main__':
 forward=json.loads((P/'forward.json').read_text());assert not forward['failed'],'Forward gate before changed-state evolution'
 steps=int(sys.argv[1]);assert steps in [1,2];start=time.monotonic();checks=[];stages=[];client=C.Client();before=frozen();initial=eos(2.,.3);y=np.array([initial['u'],.3,0.,0.,0.,0.,0.,0.]);dt=.002/steps
 def ck(n,e,g):
  c=dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g));checks.append(c)
  if not c['passed']:raise ArithmeticError('Independent RK4 gate failed: '+n)
 def rhs(vector,time_s):
  if time.monotonic()-start>105:raise TimeoutError('Bounded independent RK4 no incomplete acceptance')
  state=inversion(float(vector[0]),float(vector[1]));packet=client.evaluate(state['T'],state['Ye'],action='source',profile='base');z=packet['sources'];s=packet['state'];ck('fresh independent EOS/source stage '+str(len(stages)),s['u']-vector[0],2e-9)
  rates=np.array([z['channels'][c]['number']for c in ['beta','PC','EC']]);q=np.array([z['channels'][c]['neutrino_energy']for c in ['beta','PC','EC']]);R=rates[0]+rates[1]-rates[2];Q=sum(q)
  stages.append(dict(time=time_s,vector=vector.tolist(),independent_state=state,packet=packet));return np.r_[-Q/NB,R/NB,rates,q]
 for i in range(steps):
  t=i*dt;k1=rhs(y,t);k2=rhs(y+dt*k1/2,t+dt/2);k3=rhs(y+dt*k2/2,t+dt/2);k4=rhs(y+dt*k3,t+dt);y+=dt*(k1+2*k2+2*k3+k4)/6
 final=inversion(y[0],y[1]);E=float(sum(y[5:]));anti=float(y[2]+y[3]);nu=float(y[4]);weakminus=y[2]-y[4];weakplus=-y[3];emminus=final['ne']-initial['ne']-weakminus;emplus=final['nplus']-initial['nplus']-weakplus
 energy=(final['energy_density']+E-initial['energy_density'])/E;lepton=(NB*(y[1]-.3)+nu-anti)/(anti+nu)
 ck('independent exchanged-energy ledger',energy,2e-8);ck('independent integrated Le ledger',lepton,2e-13);ck('independent EM pair-count equality',(emminus-emplus)/(anti+nu),2e-12);ck('frozen unchanged',int(before!=frozen()),0)
 out=dict(method='independent RK4 conserved(u,Ye) + integrated moments',steps=steps,duration=.002,initial=initial,final=final,vector=y.tolist(),escaped_energy=E,escaped_nue=nu,escaped_antinue=anti,pair_EM_from_electron=float(emminus),pair_EM_from_positron=float(emplus),energy_ledger_exchange=energy,Le_ledger_gross=lepton,stages=stages,checks=checks,passed=len(checks),failed=[],seconds=time.monotonic()-start)
 (P/f'rk4_{steps}.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(steps=steps,T=final['T'],Ye=final['Ye'],E=E,passed=len(checks),seconds=out['seconds']),flush=True)
