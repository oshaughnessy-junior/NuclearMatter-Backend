"""Independent infinite-momentum scalar EOS and source-pinned client helpers."""
from pathlib import Path
import importlib.util,sys,json,math,hashlib,time
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit
P=Path(__file__).resolve().parent;ROOT=P.parent;A=ROOT/'three_source_cooling_element';NB=1e-7;HC=197.3269804;MN=939.56542052;MP=938.27208816;ME=.51099895069;sys.dont_write_bytecode=True

def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
C=load('independent_cooling_client_pure',A/'client.py')
def density(T,mu,m,anti=False):
 scale=T if m<1 else math.sqrt(2*m*T)
 def f(x):
  E=math.hypot(m,x*scale);v=expit((mu-E)/T)
  if anti:v-=expit((-mu-E)/T)
  return x*x*v
 return scale**3/(math.pi**2*HC**3)*quad(f,0,np.inf,epsabs=1e-18,epsrel=3e-12,limit=160)[0]
def eos(T,Y):
 nn=NB*(1-Y);np_=NB*Y
 mn=MN+brentq(lambda v:density(T,MN+v,MN)/nn-1,-65,0,xtol=4e-13)
 mp=MP+brentq(lambda v:density(T,MP+v,MP)/np_-1,-65,0,xtol=4e-13)
 me=brentq(lambda v:density(T,v,ME,True)/np_-1,0,3,xtol=3e-14)
 def moment(m,mu,particle=True,kinetic=False):
  scale=T if m<1 else math.sqrt(2*m*T)
  def f(x):
   E=math.hypot(m,x*scale);occupation=expit(((mu if particle else -mu)-E)/T)
   energy=(x*scale)**2/(E+m)if kinetic else E
   return x*x*occupation*energy
  return scale**3/(math.pi**2*HC**3)*quad(f,0,np.inf,epsabs=1e-17,epsrel=3e-12,limit=160)[0]
 Kn=moment(MN,mn,kinetic=True);Kp=moment(MP,mp,kinetic=True);em=moment(ME,me);ep=moment(ME,me,particle=False)
 eg=math.pi**2*T**4/(15*HC**3)
 # Baryon rest term uses independently imposed conserved particle densities;
 # pair total energy already includes both signs' rest energies.
 energy=MN*nn+MP*np_+Kn+Kp+em+ep+eg
 ne=density(T,me,ME);npair=density(T,-me,ME)
 return dict(T=T,Ye=Y,u=energy/NB,energy_density=energy,bath=dict(T=T,mu_n=mn,mu_p=mp,mu_e=me),photon_energy=eg,ne=ne,nplus=npair)
def inversion(u,Y):
 t=brentq(lambda T:eos(T,Y)['u']-u,1.8,2.1,xtol=1e-13,rtol=1e-14)
 return eos(t,Y)
def frozen():
 paths={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in A.rglob('*')if p.is_file()}
 for rel in json.loads((A/'PINS.json').read_text())['files']:paths[rel]=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
 return paths
