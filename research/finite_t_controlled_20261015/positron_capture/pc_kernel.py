"""Named free positron capture/inverse, reciprocal massive current, npe only."""
from pathlib import Path
from dataclasses import asdict
import sys, math, time
import numpy as np
from scipy.special import log_expit,logsumexp
from numpy.polynomial.legendre import leggauss
R=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(R/'relativistic_tensor'));import tensor as K
sys.path.insert(0,str(R/'relativistic_crossed_thermal'));import kernel as crossed
V=K.V;ME=V.ME

def matched(T,nB,Ye):
 return crossed.matched_npe(T,nB,Ye)

def reciprocal_vertices(q,M,model):
 """Γ_pn(q)=Dirac-adjoint Γ_np(-q), using physical q² and bare weak M."""
 ff=model.values(float(q[0]**2-np.dot(q[1:],q[1:])),M)
 neg=K.C.vertices(-q,M,ff)
 return np.array([V.adjoint(sum(neg[key][i]for key in neg))for i in range(4)])

def differential(bath,Enu,Ep,c,*,model=K.C.FormFactors(),energy_order=16,phi_order=4,tail=50.):
 if not isinstance(bath,crossed.Bath):raise ValueError('Explicit physical-npe Bath required')
 if not all(math.isfinite(x)for x in [Enu,Ep,c])or Enu<=0 or Ep<=ME or not -1<=c<=1:raise ValueError('Positive incident energy,total positron E>me,cosine required')
 state=K.State(bath.T,crossed.MP,crossed.MN,bath.mu_p,bath.mu_n,crossed.MW)
 p=math.sqrt((Ep-ME)*(Ep+ME));omega=Enu-Ep;q=math.sqrt((Enu-p)**2+2*Enu*p*(1-c))
 q4=np.array([omega,0,0,q]);recip=reciprocal_vertices(q4,state.weak_mass,model)
 # For real spacelike form factors this equality is algebraic. Explicitly
 # check it before sharing the frozen response integration; no mass swap or
 # reused nonreciprocal current is silently assumed.
 pieces=K.C.vertices(q4,state.weak_mass,model.values(omega*omega-q*q,state.weak_mass))
 direct=np.array([sum(pieces[k][i]for k in pieces)for i in range(4)])
 err=float(np.max(abs(recip-direct)))
 if err>2e-13*max(1.,float(np.max(abs(direct)))):raise ArithmeticError('Reciprocal-current identity failed')
 r=K.response(state,omega,q,model=model,energy_order=energy_order,phi_order=phi_order,tail=tail)
 kz=(Enu*Enu+q*q-p*p)/(2*q);kx2=Enu*Enu-kz*kz
 if kx2< -1e-10*Enu*Enu:raise ArithmeticError('Lepton q-frame shell outside domain')
 kx=math.sqrt(max(0.,kx2))
 k=np.array([Enu,kx,0,kz]);l=np.array([Ep,kx,0,kz-q])
 left=V.G@(V.I-V.G5)
 # Literal anti-neutrino/positron v trace, not an electron mass substitution.
 L=np.array([[np.trace(V.slash(k)@left[u]@(V.slash(l)-ME*V.I)@V.adjoint(left[v]))for v in range(4)]for u in range(4)])
 pref=V.GF**2*V.VUD**2*p/(32*math.pi**2*Enu*K.HBAR)
 # μ_e+ = -μ_e, with TOTAL energies on both sides.
 chi=K._coefficient(r['absorption'],L,log_expit((Ep+bath.mu_e)/bath.T),pref)
 j=K._coefficient(r['emission'],L,log_expit((-bath.mu_e-Ep)/bath.T),pref)
 return dict(j=j,chi=chi,reciprocal_vertex_error=err,units='s^-1 MeV^-1 per dcos',support=r['support'])

def integrate(bath,Enu,*,lepton_order=8,angle_order=8,energy_order=16,phi_order=4,tail=50.,outer_tail=24.,budget_seconds=110.,model=K.C.FormFactors()):
 if not isinstance(bath,crossed.Bath):raise ValueError('Explicit physical-npe Bath required')
 if any(isinstance(v,bool)or not isinstance(v,int)or v<4 for v in [lepton_order,angle_order]):raise ValueError('Integer outer orders>=4 required')
 if not all(math.isfinite(x)for x in [Enu,outer_tail,budget_seconds])or Enu<=0 or not 12<=outer_tail<=60 or not 0<budget_seconds<=120:raise ValueError('Finite incident energy,outertail12..60,budget<=120 required')
 center=Enu+crossed.MP-crossed.MN
 q0=Enu+math.sqrt(max(0.,center*center-ME*ME))
 width=max(bath.T,q0*math.sqrt(bath.T/crossed.MP),q0*q0/(2*crossed.MP))
 upper=max(ME,center,-bath.mu_e)+outer_tail*bath.T+8*width
 edges=sorted({ME,upper}|{x for mid in [center,-bath.mu_e]for scale in [bath.T,width]for z in [-8,-4,-2,-1,0,1,2,4,8]if ME<(x:=mid+z*scale)<upper})
 edges=np.sqrt(np.array(edges)-ME);x,w=leggauss(lepton_order);cs,ws=leggauss(angle_order)
 logs={'j':[],'chi':[]};start=time.monotonic();calls=0;err=0
 for lo,hi in zip(edges[:-1],edges[1:]):
  for a,b in zip(x,w):
   y=lo+(a+1)*(hi-lo)/2;Ep=ME+y*y;ew=b*(hi-lo)*y
   for c,cw in zip(cs,ws):
    if time.monotonic()-start>budget_seconds:raise TimeoutError('Positron capture bounded quadrature stopped')
    d=differential(bath,Enu,float(Ep),float(c),model=model,energy_order=energy_order,phi_order=phi_order,tail=tail)
    calls+=1;err=max(err,d['reciprocal_vertex_error'])
    for key in logs:
     if d[key]['log']is not None:logs[key].append(d[key]['log']+math.log(ew*cw))
 result={key:crossed.represented(float(logsumexp(vals))if vals else None)for key,vals in logs.items()}
 muanti=bath.mu_n-bath.mu_p-bath.mu_e
 return dict(**result,bath=asdict(bath),Enu=Enu,antinu_equilibrium_mu=muanti,log_kms_expected=-(Enu-muanti)/bath.T,log_kms_actual=result['j']['log']-result['chi']['log'],controls=dict(lepton_order=lepton_order,angle_order=angle_order,energy_order=energy_order,phi_order=phi_order,tail=tail,outer_tail=outer_tail),upper_Epositron=upper,panels_Epositron=(ME+edges*edges).tolist(),max_reciprocal_vertex_error=err,calls=calls,runtime=time.monotonic()-start,units='j,chi s^-1 for ONE antinue helicity; relaxation=j+chi; DOS Eν²/(2π²hc³)',channel='n+e+ -> p+antinue and antinue+p -> n+e+ ONLY',current=asdict(model))
