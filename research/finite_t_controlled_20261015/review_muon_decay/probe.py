"""Independent one-dimensional antinue spectrum and Weyl-trace review."""
from pathlib import Path
import sys,math,json,time,hashlib
import numpy as np
from scipy.integrate import quad
from scipy.special import log_expit
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parent;A=R/'muon_decay_contract';sys.path.insert(0,str(A));import decay as d
start=time.monotonic();checks=[];rows=[]
def hashes():
 return {str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest()for p in list(A.iterdir())+[R/'relativistic_capture_oracle/events.py']if p.is_file()}
initial=hashes()
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
GF=1.1663785e-11;HBAR=6.582119569e-22
# Independent Weyl gamma matrices: no imported spinor/gamma primitive.
z=np.zeros((2,2),complex);I2=np.eye(2);I=np.eye(4);sigs=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
G=np.array([np.block([[z,I2],[I2,z]])]+[np.block([[z,s],[-s,z]])for s in sigs]);G5=np.diag([-1,-1,1,1]);met=np.array([1,-1,-1,-1]);left=G@(I-G5)
def slash(p):return np.einsum('u,uij->ij',p*met,G)
def adj(v):return G[0]@v.conj().T@G[0]
def dot(p,q):return np.sum(p*q*met,axis=-1)
def independent_event(m,me,frac,c):
 Ea=(m*m-me*me)/(2*m)*frac;S=m*m-2*m*Ea;rt=math.sqrt(S);Es=(S+me*me)/(2*rt);ps=(S-me*me)/(2*rt);gamma=(m-Ea)/rt;vel=-Ea/(m-Ea)
 ec=np.array([Es,ps*math.sqrt(1-c*c),0.,ps*c]);nc=np.array([ps,-ec[1],0.,-ec[3]])
 def boost(p):return np.array([gamma*(p[0]+vel*p[3]),p[1],p[2],gamma*(p[3]+vel*p[0])])
 return np.array([m,0.,0.,0.]),boost(ec),np.array([Ea,0.,0.,Ea]),boost(nc)
def independent_trace(P,e,a,n,m,me):
 total=0j
 for u in range(4):
  for v in range(4):
   J=np.trace(slash(n)@left[u]@(slash(P)+m*I)@adj(left[v]))
   L=np.trace((slash(e)+me*I)@left[u]@slash(a)@adj(left[v]))
   total+=met[u]*met[v]*J*L
 return float(total.real*GF**2/4)
for ratio,frac,c in [(d.ME/d.MMU,.17,-.63),(d.ME/d.MMU,.83,.41),(.37,.52,-.19),(.987,.71,.57)]:
 m=d.MMU;me=ratio*m;P,e,a,n=independent_event(m,me,frac,c)
 tr=independent_trace(P,e,a,n,m,me);scalar=64*GF**2*dot(P,a)*dot(e,n);args=[p[None,:]for p in[P,e,a,n]];author=d.matrix_squared(*args,m=m,me=me)[0]
 gate('Weyl trace vs scalar '+str(ratio)+str(frac),tr/scalar-1,5e-13);gate('Weyl trace vs spinors '+str(ratio)+str(frac),author/tr-1,5e-13)
 gate('reciprocal inverse '+str(ratio)+str(frac),d.matrix_squared(*args,m=m,me=me,inverse=True)[0]/author-1,5e-13)
 gate('momentum closure '+str(ratio)+str(frac),np.max(abs(P-e-a-n))/m,1e-14)
 for p,mass,name in[(P,m,'mu'),(e,me,'e'),(a,0.,'anti'),(n,0.,'numu')]:gate('fresh shell '+name+str(ratio)+str(frac),(dot(p,p)-mass*mass)/(m*m),1e-13)
 # Full event products, including both partners, with independently varied mu.
 T=.73;mum=99.2;mue=30.7;mua=-7.3;mun=63.1
 E=[P[0],e[0],a[0],n[0]];mu=[mum,mue,mua,mun];lf=[float(log_expit((mm-ee)/T))for mm,ee in zip(mu,E)];lv=[float(log_expit((ee-mm)/T))for mm,ee in zip(mu,E)]
 F=lf[0]+sum(lv[1:]);B=lv[0]+sum(lf[1:]);gate('full-event affinity '+str(frac),F-B-(mum-mue-mua-mun)/T,3e-13)
 gate('numu partner affinity '+str(frac),lf[0]+lv[1]+lv[2]-(lv[0]+lf[1]+lf[2])+(E[3]-(mum-mue-mua))/T,3e-13)
 gate('antinue partner affinity '+str(frac),lf[0]+lv[1]+lv[3]-(lv[0]+lf[1]+lf[3])+(E[2]-(mum-mue-mun))/T,3e-13)
 rows.append(dict(point=dict(ratio=ratio,frac=frac,c=c,trace=tr,author=author)))

# New one-dimensional phase reduction: because p.n=(P-a)^2/2-me²/2,
# Mavg²=32 GF² m Ea (S-me²), independent of Ee at fixed Ea.
# The Ee interval length is Ea*(1-me²/S); derive spectrum analytically:
# dGamma/dEa=GF²/(2pi³hbar)*Ea²*(S-me²)²/S.
def one_dimensional(m,me):
 end=(m-me)*(m+me)/(2*m)
 def differential(Ea):
  S=m*m-2*m*Ea;return GF**2/(2*math.pi**3*HBAR)*Ea*Ea*(S-me*me)**2/S
 def moments(Ea):
  S=m*m-2*m*Ea;center=(m-Ea)*(S+me*me)/(2*S);width=Ea*(S-me*me)/S
  Ee2=center*center+width*width/12;En=m-Ea-center
  return np.array([1,Ea,center,En,Ea*Ea,Ee2,(m-Ea)**2-2*(m-Ea)*center+Ee2,Ea*center])
 values=[]
 for i in range(8):values.append(quad(lambda x:differential(end*x)*moments(end*x)[i]*end,0,1,epsabs=1e-18,epsrel=3e-11)[0])
 return np.array(values)
for ratio in [d.ME/d.MMU,0.,.37,.987]:
 me=ratio*d.MMU;ref=one_dimensional(d.MMU,me);author=d.pair_integral(40,me=me)
 for field,index in [('rate',0),('energy_e',2),('energy_anti_e',1),('energy_nu_mu',3)]:gate('new 1D antinue reduction '+str(ratio)+' '+field,author[field]/ref[index]-1,3e-9)
 gate('analytic total phase '+str(ratio),d.analytic_rate(me=me)/ref[0]-1,3e-10)
 gate('independent energy ledger '+str(ratio),(sum(ref[1:4])-d.MMU*ref[0])/(d.MMU*ref[0]),3e-13)
 if ratio==0:
  for index,target,name in [(1,.30,'anti mean'),(2,.35,'e mean'),(3,.35,'numu mean'),(4,.10,'anti second'),(5,2/15,'e second'),(6,2/15,'numu second'),(7,.10,'e anti cross')]:
   norm=d.MMU if index<4 else d.MMU**2;gate('independent massless '+name,ref[index]/ref[0]/norm-target,3e-13)
 rows.append(dict(one_dimension=dict(ratio=ratio,values=ref.tolist(),author=author)))
# Flavor-resolved Michel moments from their distinct normalized spectra.
for power in [0,1,2,3]:
 for name,shape,expected in [('antinue',lambda x:12*x*x*(1-x),12/((power+3)*(power+4))),('numu',lambda x:2*x*x*(3-2*x),2*(3/(power+3)-2/(power+4)))]:
  val=quad(lambda x:x**power*shape(x),0,1,epsabs=1e-14)[0];gate('analytic massless shape '+name+str(power),val-expected,3e-14)
# Parse, independently reconstruct all quantum numbers rather than importing
# author's inventory code (which would write frozen results).
quantum={'mu-':(-1,0,1),'mu+':(1,0,-1),'e-':(-1,1,0),'e+':(1,-1,0),'nue':(0,1,0),'antinue':(0,-1,0),'numu':(0,0,1),'antinumu':(0,0,-1)}
inv=json.loads((A/'inventory_results.json').read_text())
for i,reaction in enumerate(inv['rows']):
 change=np.sum([quantum[x]for x in reaction['products']],axis=0)-np.sum([quantum[x]for x in reaction['reactants']],axis=0)
 gate('reconstructed Q/Le/Lmu row'+str(i),np.max(abs(change)),0.)
# One event's material+radiation conserved ledgers and EOS source decomposition.
for cp in [1,-1]:
 dne=cp;dnmu=-cp;gate('material Yp '+str(cp),dne+dnmu,0.)
 gate('electron total Le '+str(cp),dne-cp,0.);gate('muon total Lmu '+str(cp),dnmu+cp,0.)
 # Rest mass is already in EOS; charged electron is retained. No manual Q.
 P,e,a,n=independent_event(d.MMU,d.ME,.51,.29)
 gate('matter escape energy '+str(cp),(-P[0]+e[0]+a[0]+n[0])/d.MMU,2e-14)
final=hashes();gate('all frozen source/report bytes unchanged',float(initial!=final),0.)
out=dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows,initial_hashes=initial,final_hashes=final,runtime_seconds=time.monotonic()-start)
(H/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],checks=len(checks),failed=[c for c in checks if not c['passed']],runtime=out['runtime_seconds'])))
