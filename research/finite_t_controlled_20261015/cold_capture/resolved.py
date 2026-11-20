"""Selected cold capture quadrature; unchanged physical free-NR response.

Two numerical coordinates: adaptive electron-Fermi y with q integration, and
cosine-first Gauss integration resolving cold kinematic electron-energy edges.
"""
from pathlib import Path
import sys,math,time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp,log_expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'urca_threshold'))
import threshold_study as model
A=model.a
from free_density_response import log_response,kinematic_lower_energy

def controls(Enu,order,tail):
 if not math.isfinite(Enu)or Enu<=0:raise ValueError('Finite Enu>0 required')
 if isinstance(order,bool)or not isinstance(order,int)or order<8:raise ValueError('Integer order>=8 required')
 if not math.isfinite(tail)or not 60<=tail<=240:raise ValueError('Finite electron upper tail60..240T required')

def represent(log):
 if not math.isfinite(log):raise ArithmeticError('Finite logarithmic result required')
 try:value=math.exp(log)
 except OverflowError:return dict(log=log,value=None,representable=False,reason='overflow')
 return dict(log=log,value=value if value>0 else None,representable=value>0,reason=None if value>0 else 'underflow')

def y_edges(state,tail):
 T=state.baryons.T;mu=state.mu_lepton;low=(A.ME-mu)/T
 return sorted({low,tail}|{x for x in [-512.,-256.,-128.,-80.,-40.,-20.,-8.,0.,8.,20.,40.,80.,128.]if low<x<tail})

def adaptive_yq(state,Enu,*,order=64,tail=100.,rtol=2e-10):
 """Former threshold independent-reference method, now dual j/chi with tails.

 Shares the frozen response; no external physical validation is implied.
 """
 controls(Enu,order,tail)
 if not math.isfinite(rtol)or not 1e-12<=rtol<=1e-6:raise ValueError('rtol1e-12..1e-6 required')
 s=state.baryons;T=s.T;mu=state.mu_lepton
 if mu<=A.ME:raise ValueError('Selected degenerate-electron domain requires mu_e>me')
 nodes,ws=leggauss(order);edges=y_edges(state,tail)
 def logs(y):
  E=mu+T*y
  if E<=A.ME:return [-math.inf,-math.inf]
  p=math.sqrt((E-A.ME)*(E+A.ME));qlo,qhi=abs(Enu-p),Enu+p
  if qhi<=qlo:raise ArithmeticError('q-coordinate interval collapsed at float64 precision')
  qs=qlo+(nodes+1)*(qhi-qlo)/2;qws=ws*(qhi-qlo)/2
  terms=[[],[]]
  for q,w in zip(qs,qws):
   c=max(-1.,min(1.,float((Enu*Enu+p*p-q*q)/(2*Enu*p))))
   common=math.log(float(w*E*q/Enu*A.angular_factor(E,c)))+math.log(A.K*T)
   terms[0].append(common+float(log_expit((mu-E)/T))+log_response(s.reverse(),float(q),E-Enu))
   terms[1].append(common+float(log_expit((E-mu)/T))+log_response(s,float(q),Enu-E))
  return [float(logsumexp(v))for v in terms]
 probes=[x for x in edges[1:-1]]+[.5*(a+b)for a,b in zip(edges[:-1],edges[1:])]
 scale=[max(logs(y)[i]for y in probes)for i in range(2)]
 out=[];errors=[]
 for k in range(2):
  val=err=0.
  for lo,hi in zip(edges[:-1],edges[1:]):
   v,e=quad(lambda y:math.exp(logs(y)[k]-scale[k]),lo,hi,epsabs=1e-13,epsrel=rtol,limit=300)
   val+=v;err+=e
  if val<=0:raise ArithmeticError('Scaled adaptive integral not represented')
  out.append(scale[k]+math.log(val));errors.append(err/val)
 return dict(method='adaptive_y_q',log_j=out[0],log_chi=out[1],j=represent(out[0]),chi=represent(out[1]),relative_quad_errors=errors,order=order,tail=tail,rtol=rtol)

def edge_energies(state,Enu,c,tail):
 s=state.baryons;T=s.T;mu=state.mu_lepton;upper=max(A.ME,mu)+tail*T
 base=[A.ME,upper]+[mu+k*T for k in [-512,-256,-128,-80,-40,-20,-8,-2,0,2,8,20,40,80,128]if A.ME<mu+k*T<upper]
 scan=sorted(set(base+list(np.linspace(A.ME,upper,100))))
 roots=[]
 def boundary(E,k):
  p=math.sqrt(max(0.,(E-A.ME)*(E+A.ME)));q=math.sqrt(Enu**2+p*p-2*Enu*p*c)
  em=kinematic_lower_energy(s,q,Enu-E)
  return em-s.kinetic_mu_a if k==0 else em+Enu-E-(s.offset_b-s.offset_a)-s.kinetic_mu_b
 for k in range(2):
  for lo,hi in zip(scan[:-1],scan[1:]):
   if boundary(lo,k)*boundary(hi,k)<0:roots.append(brentq(lambda E:boundary(E,k),lo,hi,xtol=1e-13))
 # Fermi-surface edge spacings are numerical panels, not widths in the response.
 for r in roots:
  base.extend(r+k*T for k in [-20,-8,-2,0,2,8,20]if A.ME<r+k*T<upper)
 return sorted(set(base))

def gauss_Ec(state,Enu,*,order=24,angle_order=32,tail=100.):
 controls(Enu,order,tail);controls(Enu,angle_order,tail)
 if state.mu_lepton<=A.ME:raise ValueError('Selected degenerate-electron domain requires mu_e>me')
 cs,cws=leggauss(angle_order);xs,xws=leggauss(order);s=state.baryons;T=s.T;mu=state.mu_lepton;terms=[[],[]]
 panel_counts=[]
 for c,cw in zip(cs,cws):
  edges=edge_energies(state,Enu,float(c),tail);panel_counts.append(len(edges)-1)
  for lo,hi in zip(edges[:-1],edges[1:]):
   for x,w in zip(xs,xws):
    E=lo+(x+1)*(hi-lo)/2;p=math.sqrt((E-A.ME)*(E+A.ME));q=math.sqrt(Enu**2+p*p-2*Enu*p*c)
    common=math.log(float(cw*w*(hi-lo)/2*E*p*A.angular_factor(E,c)))+math.log(A.K)
    terms[0].append(common+float(log_expit((mu-E)/T))+log_response(s.reverse(),q,E-Enu))
    terms[1].append(common+float(log_expit((E-mu)/T))+log_response(s,q,Enu-E))
 out=[float(logsumexp(v))for v in terms]
 return dict(method='Gauss_E_cosine_kinematic_edges',log_j=out[0],log_chi=out[1],j=represent(out[0]),chi=represent(out[1]),order=order,angle_order=angle_order,tail=tail,panel_count_range=[min(panel_counts),max(panel_counts)])
