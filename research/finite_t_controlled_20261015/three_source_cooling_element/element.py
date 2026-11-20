"""Short conservative fixed-density three-CC cooling element, LTE photons/pairs."""
from pathlib import Path
import importlib.util,sys,json,time,math
P=Path(__file__).parent;NB=1e-7
sp=importlib.util.spec_from_file_location('three_source_cooling_client_function',P/'client.py');M=importlib.util.module_from_spec(sp);sys.modules[sp.name]=M;sp.loader.exec_module(M)
def invert(c,u,Y,guess):
 if not all(type(v)in(int,float)and math.isfinite(v)for v in[u,Y,guess]):raise ValueError('Finite conserved state required')
 T=guess;history=[]
 for i in range(8):
  s=c.evaluate(T,Y)['state'];err=s['u']-u;history.append(dict(T=T,u_error=err))
  if abs(err)<=2e-10:return s,history
  step=err/s['u_T'];T=T-step
  if not 1.8<=T<=2.1:raise ArithmeticError('EOS inverse left admitted temperature slice')
 raise ArithmeticError('ActualEOS energy inversion did not converge')

def trajectory(method,steps,duration=.002):
 if method not in['Euler','Heun']or type(steps)is not int or steps<1 or type(duration)not in(int,float)or not math.isfinite(duration)or not 0<duration<=.003:raise ValueError('Declared short positive trajectory only')
 c=M.Client();initial=c.evaluate(2.,.3)['state'];s=initial;dt=duration/steps;records=[];I={k:dict(number=0.,energy=0.)for k in['beta','PC','EC']};start=time.monotonic()
 def accumulate(z,w):
  for k in I:
   I[k]['number']+=w*z['channels'][k]['number'];I[k]['energy']+=w*z['channels'][k]['neutrino_energy']
 for i in range(steps):
  if time.monotonic()-start>95:raise TimeoutError('Bounded trajectory stopped; no completion claim')
  call=c.evaluate(s['T'],s['Ye'],action='source');z=call['sources'];u0=s['u'];Y0=s['Ye'];pred,pi=invert(c,u0+dt*z['u_dot'],Y0+dt*z['Ye_dot'],s['T'])
  if method=='Euler':
   end=pred;inv=pi;accumulate(z,dt);z2=None
  else:
   z2=c.evaluate(pred['T'],pred['Ye'],action='source')['sources'];end,inv=invert(c,u0+dt*(z['u_dot']+z2['u_dot'])/2,Y0+dt*(z['Ye_dot']+z2['Ye_dot'])/2,pred['T']);accumulate(z,dt/2);accumulate(z2,dt/2)
  records.append(dict(step=i,time=i*dt,state=s,source=z,predicted_state=pred,predictor_inversion=pi,second_source=z2,final_inversion=inv,final_state=end));s=end
 E=sum(v['energy']for v in I.values());anti=I['beta']['number']+I['PC']['number'];nu=I['EC']['number'];net=anti-nu
 # Actual pair population changes = weak particle sources + an equal EMpair change.
 weakminus=I['beta']['number']-I['EC']['number'];weakplus=-I['PC']['number'];emminus=s['electron_particle_density']-initial['electron_particle_density']-weakminus;emplus=s['positron_density']-initial['positron_density']-weakplus
 return dict(method=method,steps=steps,duration=duration,initial=initial,final=s,records=records,integrated=I,escaped_energy=E,escaped_nue=nu,escaped_antinue=anti,net_material_lepton=net,energy_ledger=s['energy_density']+E-initial['energy_density'],energy_ledger_relative_exchange=(s['energy_density']+E-initial['energy_density'])/E,lepton_ledger=NB*(s['Ye']-initial['Ye'])+nu-anti,pair_EMnumber_change=dict(from_electron=emminus,from_positron=emplus,equality_residual=emminus-emplus),client_calls=c.calls,seconds=time.monotonic()-start,thermostat_energy=0.,photon_energy_escape=0.,scope='shortfixednB trappedLTEphoton/pair three namedCC element only')
