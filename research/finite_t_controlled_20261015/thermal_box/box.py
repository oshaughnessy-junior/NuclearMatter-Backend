"""Closed finite spectral-bin matter-neutrino CC reference; no continuum claim."""
from pathlib import Path
import sys,math,time
import numpy as np
from scipy.special import expit,xlogy
from scipy.optimize import root
ROOT=Path(__file__).resolve().parents[1];sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'fd_fluid_element'))
import element as material

FLAVORS=('nue','antinue');SIGNS=np.array([1.,-1.])[:,None]

def grid(edges=(0.,4.,8.,14.,24.)):
 edges=np.asarray(edges,float)
 if edges.ndim!=1 or len(edges)<2 or edges[0]!=0 or not np.all(np.isfinite(edges))or np.any(np.diff(edges)<=0):raise ValueError('Finite increasing edges starting at0 required')
 weights=material.DOS*np.diff(edges**3)/3
 energies=.75*np.diff(edges**4)/np.diff(edges**3)
 return dict(edges=edges,weights=weights,energies=energies)

def eos(T,nB,Y):
 s=material.eos(T,nB,Y);m=s['matter']
 entropy=(m.neutron.entropy_density+m.proton.entropy_density+m.electron.entropy_density)/nB
 affinity=m.proton.mu+m.electron.chemical_potential-m.neutron.mu
 return dict(T=float(T),Ye=float(Y),u=float(s['u_MeV_per_baryon']),s=float(entropy),affinity=float(affinity),matter=m)

def validate_f(f,g):
 f=np.asarray(f,float)
 if f.shape!=(2,len(g['energies']))or not np.all(np.isfinite(f))or np.any((f<0)|(f>1)):raise ValueError('Radiation occupations finite0..1 with2flavors x bins required')
 return f

def invariants(T,Y,f,nB,g):
 f=validate_f(f,g);s=eos(T,nB,Y);w,E=g['weights'],g['energies']
 energy=s['u']+float(np.sum(w*E*f))/nB
 lepton=Y+float(np.sum(SIGNS*w*f))/nB
 radentropy=float(np.sum(w*(-xlogy(f,f)-xlogy(1-f,1-f))))/nB
 return dict(energy_MeV_per_baryon=energy,electron_lepton_per_baryon=lepton,entropy_per_baryon=s['s']+radentropy,matter_entropy_per_baryon=s['s'],radiation_entropy_per_baryon=radentropy)

def rates(T,Y,nB,g,*,order=48,tail=40.):
 s=eos(T,nB,Y);j=np.zeros((2,len(g['energies'])));chi=j.copy();channels=[]
 for i,flavor in enumerate(FLAVORS):
  for k,E in enumerate(g['energies']):
   c=material.physics.coefficients(s['matter'],flavor,float(E),order=order,tail=tail)
   j[i,k]=c['j_s'];chi[i,k]=c['chi_s'];channels.append(dict(flavor=flavor,E=float(E),coefficients=c))
 return dict(T=float(T),Ye=float(Y),nB=float(nB),edges=g['edges'].copy(),j=j,chi=chi,channels=channels,order=order,tail=tail)

def validate_rates(T,Y,nB,g,r):
 if r['T']!=T or r['Ye']!=Y or r['nB']!=nB or not np.array_equal(r['edges'],g['edges']):raise ValueError('Physical rates must match current matter state and grid')
 for name in ['j','chi']:
  values=np.asarray(r[name],float)
  if values.shape!=(2,len(g['energies']))or not np.all(np.isfinite(values))or np.any(values<=0):raise ValueError('Finite positive physical j/chi required on every flavor/bin')

def entropy_production(T,Y,f,nB,g,r):
 validate_rates(T,Y,nB,g,r)
 f=validate_f(f,g)
 if np.any((f<=0)|(f>=1)):raise ValueError('Interior occupations required for finite instantaneous entropy diagnostic')
 C=r['j']*(1-f)-r['chi']*f
 # Log forward/reverse ratio is evaluated directly from both positive rates.
 force=np.log(r['j'])+np.log1p(-f)-np.log(r['chi'])-np.log(f)
 return float(np.sum(g['weights']*C*force)/nB)

def step(T,Y,f,nB,g,dt,r):
 if not math.isfinite(dt)or dt<=0:raise ValueError('Finite dt>0 required')
 validate_rates(T,Y,nB,g,r)
 f=validate_f(f,g);before=eos(T,nB,Y)
 C=r['j']*(1-f)-r['chi']*f
 delta=dt*C;newf=f+delta;validate_f(newf,g)
 du=-float(np.sum(g['weights']*g['energies']*delta))/nB
 dY=-float(np.sum(SIGNS*g['weights']*delta))/nB
 newY=Y+dY
 if not 0<newY<1:raise ArithmeticError('Composition left material EOS domain')
 newT=material.recover_temperature(before['u']+du,nB,newY,T)
 return newT,newY,newf,dict(du=du,dYe=dY,dt=dt,delta_f=delta.tolist())

def equilibrium(nB,g,energy,lepton,Tguess=2.,Yguess=.3):
 def payload(T,Y):
  s=eos(T,nB,Y);f=expit((SIGNS*s['affinity']-g['energies'])/T)
  return f,invariants(T,Y,f,nB,g)
 def residual(z):
  T=math.exp(float(z[0]));Y=float(expit(z[1]));f,v=payload(T,Y)
  return [(v['energy_MeV_per_baryon']-energy)/max(abs(energy),1.),v['electron_lepton_per_baryon']-lepton]
 solution=root(residual,[math.log(Tguess),math.log(Yguess/(1-Yguess))],tol=1e-10)
 if not solution.success or max(abs(np.asarray(residual(solution.x))))>1e-9:raise ArithmeticError('Finite-grid LTE conservation solve failed: '+solution.message)
 T=math.exp(float(solution.x[0]));Y=float(expit(solution.x[1]));f,totals=payload(T,Y)
 return dict(T=T,Ye=Y,f=f,totals=totals,residual=residual(solution.x),solver_message=solution.message)
