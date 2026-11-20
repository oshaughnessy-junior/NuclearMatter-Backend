"""Quadrature-free Boltzmann limit of the same forbidden-side physical chain.

After exact azimuth average the fixed-transfer spin trace is quadratic in Ea.
This checks the actual Fermi integral; it does not replace it or fit a width.
"""
import json,math
from pathlib import Path
import numpy as np
from scipy.special import log_expit
import probe as P
HERE=Path(__file__).resolve().parent;V=P.A.free.V;C=P.A.free.C
r=json.loads((HERE/'results.json').read_text());checks=[];rows=[]
def array(d):return np.array(d['real'])+1j*np.array(d['imag'])
def ck(name,e,t):checks.append(dict(name=name,error=abs(float(e)),limit=t,passed=bool(abs(e)<=t)))
for row in r['rows']:
 if row['density_factor']!=.97 or row['T']not in(.1,.3):continue
 s=P.A.OrderedState(**row['state']);T=s.T;win=row['instantaneous_fields_T0_density_window'];M=s.dirac_mass;q=win['q'];w=win['w'];E0=win['Emin'];kin=row['kinematics'];actual=row['runs'][-1];L=array(actual['lepton_tensor'])
 ff=C.FormFactors().values(kin['omega']**2-q*q,939.);pieces=C.vertices(np.array([w,0.,0.,q]),939.,ff);J=[sum(pieces[k][u]for k in pieces)for u in range(4)]
 def trace_at(E):
  z=(2*w*E+w*w-q*q)/(2*q);pt2=E*E-M*M-z*z
  assert pt2> -1e-6
  pt=math.sqrt(max(0.,pt2));H=np.zeros((4,4),complex)
  for phi in 2*math.pi*np.arange(4)/4:
   a=np.array([E,pt*math.cos(phi),pt*math.sin(phi),z]);b=a+np.array([w,0.,0.,q]);aa=V.slash(a)+M*V.I;bb=V.slash(b)+M*V.I
   H+=np.array([[np.trace(bb@J[u]@aa@V.adjoint(J[v]))for v in range(4)]for u in range(4)])/4
  value=np.sum(V.METRIC[:,None]*V.METRIC[None,:]*L*H);assert abs(value.imag)<1e-10*abs(value.real)and value.real>0
  return float(value.real)
 estimates=[]
 for step in(10.,20.):
  f0,f1,f2=[trace_at(E0+i*step)for i in range(3)]
  A=(f2-2*f1+f0)/(2*step*step);B=(f1-f0)/step-A*step
  ck('fixed_transfer_trace_exact_quadratic',(trace_at(E0+3*step)-(f0+3*step*B+(3*step)**2*A))/trace_at(E0+3*step),2e-12)
  integral=f0*T+B*T*T+2*A*T**3;assert integral>0
  nua=s.total_mu_a-s.potential_a;nub=s.total_mu_b-s.potential_b;El=kin['El'];Enu=kin['Enu'];pl=math.sqrt((El-P.B.ME)*(El+P.B.ME))
  logpref=math.log(V.GF**2*V.VUD**2*pl/(32*math.pi**2*Enu*P.A.free.HBAR)/(8*math.pi*q))
  lograte=logpref+math.log(integral)+(nua-E0)/T+float(log_expit((El-row['electron']['chemical_potential'])/T));estimates.append(lograte)
 ck('Boltzmann_polynomial_step_independence',estimates[0]-estimates[1],2e-11)
 za=math.exp((nua-E0)/T);zb=math.exp((nub-E0-w)/T);lower=1/((1+za)*(1+zb));ratio=math.exp(actual['chi']['log']-estimates[-1])
 # Positive trace gives an absolute multiplicative FD-vs-Boltzmann bound.
 ck('FD_to_MB_rigorous_positive_bound',max(0.,lower-ratio,ratio-1),3e-10)
 rows.append(dict(T=T,closure=row['closure'],log_chi_FD=actual['chi']['log'],log_chi_MB_analytic=estimates[-1],FD_MB_ratio=ratio,rigorous_ratio_interval=[lower,1.],initial_neutron_activation_MeV=E0-nua,cold_window_gap_MeV=win['absorption_signed_gap'],trace_quadratic=True))
out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=sum(not c['passed']for c in checks),rows=rows,scope='Analytic dilute-tail benchmark with actual fields/chemical roots/lepton blocking/full current; not a substitute model or width fit')
P.save_json(HERE/'boltzmann_results.json',out);print(out['passed'],out['failed'])
if out['failed']:raise SystemExit(1)
