"""Independent explicit spinors, full unequal-mass SV polynomial and t measure."""
from pathlib import Path
import sys,math,time,json,hashlib,decimal
import numpy as np
from scipy.integrate import quad
ROOT=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'muon_vacuum'));import vacuum_cc as V
start=time.monotonic();checks=[];rows=[]
paths=['muon_vacuum/vacuum_cc.py','muon_vacuum/initial_exothermic_threshold_failure.json','muon_vacuum/initial_threshold_asymptotic_failure.json']
hashes={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error) and error<tol)))
# A separately constructed Dirac representation, used for EXPLICIT amplitudes.
pauli=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.]).astype(complex)]
zero=np.zeros((2,2),complex);one=np.eye(2);gamma=[np.block([[one,zero],[zero,-one]])]+[np.block([[zero,s],[-s,zero]]) for s in pauli]
g5=np.block([[zero,one],[one,zero]]);eta=np.array([1.,-1.,-1.,-1.]);I=np.eye(4)
basis=np.eye(2,dtype=complex)
def spinors(p,m,anti=False):
 root=math.sqrt(p[0]+m);sigma=sum(p[i+1]*pauli[i] for i in range(3))
 return [np.concatenate([sigma@x/root,root*x]) if anti else np.concatenate([root*x,sigma@x/root]) for x in basis]
def bar(u):return u.conjugate()@gamma[0]
def own_momenta(ch,E,c):
 a,b,m=ch.initial_mass,ch.final_mass,ch.lepton_mass;s=a*a+2*a*E;S=math.sqrt(s);pi=(s-a*a)/(2*S)
 El=(s+m*m-b*b)/(2*S);pf=math.sqrt(max(0.,El*El-m*m))
 return dict(k=np.array([pi,0,0,pi]),pa=np.array([(s+a*a)/(2*S),0,0,-pi]),l=np.array([El,pf*math.sqrt(1-c*c),0,pf*c]),pb=np.array([(s+b*b-m*m)/(2*S),-pf*math.sqrt(1-c*c),0,-pf*c]),s=s,pi=pi,pf=pf)
def explicit(ch,p):
 incoming=spinors(p['k'],0.,ch.antineutrino);leptons=spinors(p['l'],ch.lepton_mass,ch.antineutrino)
 targets=spinors(p['pa'],ch.initial_mass);out=spinors(p['pb'],ch.final_mass)
 by_incident=[]
 for nu in incoming:
  value=0.
  for a in targets:
   for b in out:
    H=np.array([bar(b)@g@(ch.gv*I-ch.ga*g5)@a for g in gamma])
    for l in leptons:
     L=np.array([bar(nu)@g@(I-g5)@l if ch.antineutrino else bar(l)@g@(I-g5)@nu for g in gamma])
     amp=V.GF*V.VUD/math.sqrt(2)*sum(eta*L*H)
     value+=abs(amp)**2/2  # target average ONLY
  by_incident.append(float(value))
 return float(sum(by_incident)),by_incident
# Full unequal-mass Strumia/Vissani Eq6, f2=g2=0. Source normalizes its reduced
# M² differently: physicalM²=8 GF² Vud² reducedM². Delta in source follows
# neutron-proton ordering; for the neutrino crossed channel ma/mb interchange.
def reduced(ch,s,t):
 a,b,m=ch.initial_mass,ch.final_mass,ch.lepton_mass;M=(a+b)/2
 Delta=(b-a) if ch.antineutrino else (a-b);v,g=ch.gv,ch.ga
 A=(t-m*m)/4*(v*v*(4*M*M+t+m*m)+g*g*(-4*M*M+t+m*m))
 A-=Delta*Delta/4*(v*v*(4*M*M+t-m*m)+g*g*(4*M*M-t+m*m))
 A+=2*m*m*M*Delta*v*g
 B=-t*v*g;C=(v*v+g*g)/4;u=a*a+b*b+m*m-s-t;D=s-u
 return A+(-1 if ch.antineutrino else 1)*D*B+D*D*C

def sigma_t(ch,E):
 p=own_momenta(ch,E,0.);s=p['s'];k=p['pi'];El=p['l'][0];pf=p['pf'];lo=ch.lepton_mass**2-2*k*El-2*k*pf;hi=ch.lepton_mass**2-2*k*El+2*k*pf
 result=quad(lambda t:reduced(ch,s,t),lo,hi,epsabs=1e-5,epsrel=1e-12)[0]
 return V.GF**2*V.VUD**2/(2*math.pi*(s-ch.initial_mass**2)**2)*result*V.HBARC**2*1e-26
for anti in (False,True):
 for m,label in [(V.ME,'electron'),(V.MMU,'muon')]:
  ch=V.Channel(V.MP if anti else V.MN,V.MN if anti else V.MP,m,anti)
  with decimal.localcontext() as ctx:
   ctx.prec=60;D=decimal.Decimal;a,b,l=map(lambda v:D(str(v)),(ch.initial_mass,ch.final_mass,m));threshold=max(D(0),((b+l)**2-a*a)/(2*a))
  gate(f'decimal exact threshold {anti}{label}',abs(ch.threshold-float(threshold)),5e-13)
  if ch.threshold:
   gate(f'below threshold support {anti}{label}',V.total(ch,ch.threshold-1e-6),1e-100)
   gate(f'positive nextafter threshold {anti}{label}',float(V.total(ch,float(np.nextafter(ch.threshold,math.inf)))<=0),.1)
  for E in ((17.3,77.7) if m==V.ME else (137.2,247.3,511.1)):
   for c in (-.83,.17,.76):
    p=own_momenta(ch,E,c);point,helicities=explicit(ch,p);trace=V.trace_amplitude(ch,p)
    t=(p['k'][0]-p['l'][0])**2-np.dot(p['k'][1:]-p['l'][1:],p['k'][1:]-p['l'][1:])
    polynomial=8*V.GF**2*V.VUD**2*reduced(ch,p['s'],float(t))
    gate(f'explicitspinor versus trace {anti}{label}E{E}c{c}',abs(point/trace-1),3e-12)
    gate(f'full unequalmass sourcepolynomial {anti}{label}E{E}c{c}',abs(polynomial/trace-1),3e-12)
    gate(f'one inactive chiral incident state {anti}{label}E{E}c{c}',min(helicities)/max(helicities),1e-28)
    # Flux+phase: Eq3 dt measure transformed with dt/dcos=2pi pf.
    source_diff=V.GF**2*V.VUD**2/(2*math.pi*(p['s']-ch.initial_mass**2)**2)*reduced(ch,p['s'],float(t))*2*p['pi']*p['pf']*V.HBARC**2*1e-26
    gate(f'independent dt CM differential {anti}{label}E{E}c{c}',abs(source_diff/V.differential(ch,E,c)-1),3e-12)
    rows.append(dict(anti=anti,lepton=label,E=E,c=c,explicit_amp2=point,incoming_chiral_contributions=helicities,source_polynomial_amp2=polynomial))
   gate(f'adaptive invariant-t total {anti}{label}E{E}',abs(sigma_t(ch,E)/V.total(ch,E,order=12)-1),3e-12)
# New common-heavyM masses and couplings: tests the 1/2 target average and
# absence of incoming1/2 against the independently imposed static limit.
heavy=[]
for anti in (False,True):
 for m,E in [(V.ME,31.7),(V.MMU,183.4)]:
  errors=[]
  for M in (2e6,2e7):
   ch=V.Channel(M,M,m,anti,gv=.9,ga=1.1);static=V.GF**2*V.VUD**2/math.pi*(ch.gv**2+3*ch.ga**2)*E*math.sqrt(E*E-m*m)*V.HBARC**2*1e-26
   value=V.total(ch,E,order=12);error=abs(value/static-1);errors.append(error)
   gate(f'independent static normalization anti{anti}m{m}M{M}',error,8e-4 if M==2e6 else 8e-5)
   heavy.append(dict(anti=anti,m=m,E=E,M=M,error=error))
  gate(f'heavy recoil convergence anti{anti}m{m}',float(errors[1]>=errors[0]/8),.1)
# Independent explicit-spinor Lorentz invariance from CM to target-rest lab.
for anti in (False,True):
 ch=V.Channel(V.MP if anti else V.MN,V.MN if anti else V.MP,V.MMU,anti)
 p=own_momenta(ch,191.3,.42);beta=p['pi']/p['pa'][0];gamma_boost=1/math.sqrt(1-beta*beta)
 def boost(v):return np.array([gamma_boost*(v[0]+beta*v[3]),v[1],v[2],gamma_boost*(v[3]+beta*v[0])])
 lab={k:boost(p[k]) for k in ('k','pa','l','pb')}
 gate(f'independent targetrest boost anti{anti}',abs(lab['pa'][3])/ch.initial_mass,2e-15)
 gate(f'independent incoming labenergy anti{anti}',abs(lab['k'][0]/191.3-1),2e-15)
 gate(f'explicitspinor Lorentz invariance anti{anti}',abs(explicit(ch,lab)[0]/explicit(ch,p)[0]-1),3e-12)
 # Electron/massless limit is parameterized per channel, without global edits.
 zero=V.Channel(ch.initial_mass,ch.final_mass,0.,anti);tiny=V.Channel(ch.initial_mass,ch.final_mass,1e-3,anti)
 gate(f'new masslesslimit anti{anti}',abs(V.total(tiny,217.4)/V.total(zero,217.4)-1),1e-9)
for v,g,label in [(1.,0.,'vector'),(0.,1.,'axial')]:
 nu=V.Channel(939.,939.,V.MMU,False,v,g);anti=V.Channel(939.,939.,V.MMU,True,v,g)
 gate(f'noVAinterference pure{label} crossing',abs(V.total(nu,217.4)/V.total(anti,217.4)-1),3e-12)
final={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths};gate('source and preserved failures unchanged',float(hashes!=final),.1)
result=dict(status='PASS' if all(x['passed'] for x in checks) else 'FAIL',checks=checks,rows=rows,heavy=heavy,source_sha256=hashes,end_source_sha256=final,runtime_seconds=time.monotonic()-start,scope='Vacuum pointV-A currents only, noWM/PS/formfactor/radiative terms or warmkernel; algebra agreement is not physicaluncertainty')
(P/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'],sum(x['passed'] for x in checks),'/',len(checks),result['runtime_seconds'])
for x in checks:
 if not x['passed']:print('FAIL',x)
