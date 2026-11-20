"""Explicit external-spin sums, independent of author's trace contraction."""
from pathlib import Path
import sys,json,math,hashlib,time,atexit,itertools
import numpy as np
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(ROOT/'muon_full_current'))
import current as C
start=time.monotonic();checks=[];rows=[];phase='start';paths=[ROOT/'muon_full_current'/x for x in ['current.py','source_polynomial.py','results.json']]+[ROOT/'muon_vacuum/vacuum_cc.py'];hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
def save():
 end={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
 (HERE/'results.json').write_text(json.dumps(dict(passed=phase=='complete'and all(c['passed']for c in checks)and hashes==end,phase=phase,checks=checks,rows=rows,runtime_seconds=time.monotonic()-start,source_sha256=hashes,end_source_sha256=end),indent=2)+'\n')
atexit.register(save)
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
# Own Pauli/gamma and normalized u,v construction; no V.G/slash/adjoint use.
pauli=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.array([[1,0],[0,-1]],complex)];zero=np.zeros((2,2),complex);G=[np.diag([1,1,-1,-1]).astype(complex)]+[np.block([[zero,s],[-s,zero]])for s in pauli];G5=1j*G[0]@G[1]@G[2]@G[3];metric=[1,-1,-1,-1];I=np.eye(4)
def bar(v):return v.conj()@G[0]
def adj(M):return G[0]@M.conj().T@G[0]
def spinors(p,m,antiparticle=False):
 E=p[0];root=math.sqrt(E+m);sigma=sum(p[k+1]*pauli[k]for k in range(3));out=[]
 for chi in np.eye(2):
  big=root*chi;small=sigma@chi/root
  out.append(np.r_[small,big]if antiparticle else np.r_[big,small])
 return out

def vertex(q,M,ff):
 v,a,f,g=ff;out={k:[]for k in ['VV','AA','FF','PP']}
 for u in range(4):
  comm=sum((G[u]@G[w]-G[w]@G[u])*metric[w]*q[w]for w in range(4))
  out['VV'].append(v*G[u]);out['AA'].append(-a*G[u]@G5);out['FF'].append(-f*comm/(4*M));out['PP'].append(-g*G5*q[u]/M)
 return out

def spin_sum(ch,p,ff):
 M=(ch.initial_mass+ch.final_mass)/2;q=p['pb']-p['pa']
 if ch.antineutrino:
  # Physical conjugate transition constructed as barGamma(-q), not guessed signs.
  currents={name:[adj(x)for x in val]for name,val in vertex(-q,M,ff).items()}
 else:currents=vertex(q,M,ff)
 ua=spinors(p['pa'],ch.initial_mass);ub=spinors(p['pb'],ch.final_mass);uk=spinors(p['k'],0.,ch.antineutrino);ul=spinors(p['l'],ch.lepton_mass,ch.antineutrino)
 amps=[]
 for a,b,k,l in itertools.product(ua,ub,uk,ul):
  L=np.array([bar(k)@G[u]@(I-G5)@l if ch.antineutrino else bar(l)@G[u]@(I-G5)@k for u in range(4)])
  amps.append([sum(metric[u]*L[u]*(bar(b)@mat[u]@a)for u in range(4))for mat in currents.values()])
 amps=np.array(amps);pref=.5*C.V.GF**2*C.V.VUD**2/2;parts={};names=['V','A','F','P']
 for i in range(4):
  for j in range(i,4):parts[names[i]+names[j]]=float(pref*np.real(np.sum(amps[:,i]*amps[:,j].conj()))*(1 if i==j else 2))
 total=float(pref*np.sum(abs(np.sum(amps,axis=1))**2));return total,parts

for anti in [False,True]:
 for ml,label in [(C.V.ME,'electron'),(C.V.MMU,'muon')]:
  for couplings in [(1.,1.2723,3.706),(0.82,1.09,4.2)]:
   model=C.FormFactors(gv=couplings[0],ga=couplings[1],xi=couplings[2],vector_mass=823.,axial_mass=1071.,pion_mass=138.7)
   ch=C.V.Channel(C.V.MP if anti else C.V.MN,C.V.MN if anti else C.V.MP,ml,anti,model.gv,model.ga);M=(ch.initial_mass+ch.final_mass)/2
   for E,cosine in ([(17.3,-.44),(63.7,.51)]if label=='electron'else[(ch.threshold+.37,-.31),(187.4,.63)]):
    phase=f'{anti,label,E,cosine,couplings}';p=C.V.momenta(ch,E,cosine);q=p['pb']-p['pa'];t=q[0]**2-float(q[1:]@q[1:]);ff=model.values(t,M)
    # First forward point gate verifies independent spinor normalization.
    point,_=spin_sum(ch,p,(model.gv,model.ga,0.,0.));gate('point spinor vs frozen invariant '+phase,point/C.V.invariant_amplitude(ch,p)-1,3e-12)
    total,parts=spin_sum(ch,p,ff);trace=C.trace(ch,p,ff);reference=C.trace(ch,p,ff,decompose=True)
    gate('full explicit spinors vs trace '+phase,total/trace-1,3e-12)
    for key,value in parts.items():
     scale=abs(reference[key])if key not in ['VP','FP']else trace
     gate('component '+key+' '+phase,(value-reference[key])/scale,3e-7 if key in ['AP','PP']and label=='electron'else 3e-11)
    den=(1-t/(4*M*M))*(1-t/823.**2)**2
    independent=[model.gv*(1-(1+model.xi)*t/(4*M*M))/den,model.ga/(1-t/1071.**2)**2,model.xi/den,2*M*M*(model.ga/(1-t/1071.**2)**2)/(138.7**2-t)]
    gate('independent pole conventions '+phase,np.max(abs(np.array(ff)/independent-1)),3e-14)
    rows.append(dict(anti=anti,lepton=label,E=E,cosine=cosine,t=t,model=model.__dict__,ff=ff,spinor_amplitude=total,trace_amplitude=trace,spinor_parts=parts,trace_parts=reference));save()
    if not all(c['passed']for c in checks):raise ArithmeticError('Independent spinor comparison failed; preserved before more cases')
phase='complete';save();print(len(checks),all(c['passed']for c in checks),time.monotonic()-start)
