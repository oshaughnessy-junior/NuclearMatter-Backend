"""Fresh Weyl-basis literal traces vs independent frozen scalar/Dirac paths."""
from pathlib import Path
import sys,importlib.util,json,math,datetime
import numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True
(P/'POINT_LOCK.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_imported=False,EC=dict(Ee=4.3,cosCM=-.47,phiCM=.31),PC=dict(Ee=2.71,cosCM=.29,phiCM=-.42),NEUTRON_DECAY=dict(Ee=.947,cos_e_nu=-.37),relative_gate=3e-10,scope='Physicalon-shellpointtraceonly; independentWeylrepresentation; notnewtotalrates/externalvalidation'),indent=2)+'\n')
s=importlib.util.spec_from_file_location('weak_point_unique_oracles',P/'oracles.py');O=importlib.util.module_from_spec(s);sys.modules[s.name]=O;s.loader.exec_module(O)
I2=np.eye(2);Z=np.zeros((2,2));paulis=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),np.diag([1.,-1.])];g=np.array([np.block([[Z,I2],[I2,Z]])]+[np.block([[Z,x],[-x,Z]])for x in paulis]);g5=np.diag([-1.,-1.,1.,1.]);I=np.eye(4);eta=np.array([1.,-1.,-1.,-1.]);L=g@(I-g5)
adj=lambda a:g[0]@a.conj().T@g[0];slash=lambda p:sum(eta[j]*p[j]*g[j]for j in range(4));mdot=lambda p:p[0]*p[0]-p[1:]@p[1:]
def vertex(q):
 t=mdot(q);MW=O.MW;den=(1-t/(4*MW*MW))*(1-t/840**2)**2;v=(1-4.706*t/(4*MW*MW))/den;a=1.2723/(1-t/1000**2)**2;f=3.706/den;gp=2*MW*MW*a/(139.57**2-t)
 return np.array([v*g[u]-a*g[u]@g5+1j*f/(2*MW)*sum(.5j*(g[u]@g[j]-g[j]@g[u])*eta[j]*q[j]for j in range(4))-gp/O.MW*q[u]*g5 for u in range(4)])
def literal(ch,k,n,l,p):
 V=vertex(p-n);sn=slash(n)+O.MN*I;sp=slash(p)+O.MP*I;sk=slash(k);sl=slash(l)+(-1 if ch=='PC'else 1)*O.ME*I;val=0j
 for u in range(4):
  for v in range(4):
   if ch=='EC':H=np.trace(sn@adj(V[u])@sp@V[v]);LT=np.trace(sk@L[u]@sl@adj(L[v]))
   else:H=np.trace(sp@V[u]@sn@adj(V[v]));LT=np.trace(sl@L[u]@sk@adj(L[v]))
   val+=eta[u]*eta[v]*H*LT
 return (1.1663785e-11**2*.974**2/(4 if ch=='NEUTRON_DECAY'else 8))*val
checks=[];details=[]
def ck(name,e,g=3e-10):checks.append(dict(name=name,error=float(e),gate=g,passed=bool(abs(e)<=g)))
for ch,Ee,c,ph in[('EC',4.3,-.47,.31),('PC',2.71,.29,-.42)]:
 mi,mf=(O.MP,O.MN)if ch=='EC'else(O.MN,O.MP);pe=math.sqrt(Ee*Ee-O.ME*O.ME);pa=np.array([mi,0,0,0.]);l=np.array([Ee,0,0,pe]);total=pa+l;s=mdot(total);rt=math.sqrt(s);pf=(s-mf*mf)/(2*rt);v=total[1:]/total[0];kc=np.array([[pf,pf*math.sqrt(1-c*c)*math.cos(ph),pf*math.sqrt(1-c*c)*math.sin(ph),pf*c]]);k=O.E.boost(kc,v[None,:])[0];pb=total-k;n,p=(pb,pa)if ch=='EC'else(pa,pb)
 z=literal(ch,k,n,l,p);ref=O.EC.trace_amp(k[None,:],n[None,:],l[None,:],p[None,:])[0]if ch=='EC'else O.PC.amplitude(k[None,:],p[None,:],l[None,:],n[None,:])[0]
 ck(ch+' Weylrelative',z.real/ref-1);ck(ch+' imaginary',z.imag/max(abs(z.real),1e-300));ck(ch+' target/finalshell',(mdot(pb)-mf*mf)/(mf*mf),2e-10);details.append(dict(channel=ch,weyl=float(z.real),reference=float(ref),k=k.tolist(),n=n.tolist(),l=l.tolist(),p=p.tolist()))
Ee=.947;c=-.37;pe=math.sqrt(Ee*Ee-O.ME*O.ME);Emax=((O.MN-O.MP)*(O.MN+O.MP)+O.ME*O.ME)/(2*O.MN);en=O.MN*(Emax-Ee)/(O.MN-Ee+pe*c);n=np.array([O.MN,0,0,0]);l=np.array([Ee,0,0,pe]);k=np.array([en,en*math.sqrt(1-c*c),0,en*c]);p=n-l-k;z=literal('NEUTRON_DECAY',k,n,l,p);q=p-n;ff=O.C.FormFactors().values(mdot(q),O.MW);ref=sum(O.C.guo_invariants(O.V.Channel(O.MN,O.MP,O.ME),dict(k=k,pa=n,l=l,pb=p),ff).values());ck('beta Weylrelative',z.real/ref-1);ck('beta imaginary',z.imag/max(abs(z.real),1e-300));ck('beta protononshell',(mdot(p)-O.MP*O.MP)/(O.MP*O.MP),2e-10);details.append(dict(channel='NEUTRON_DECAY',weyl=float(z.real),reference=float(ref),qGamma=q.tolist(),Qleptons=(l+k).tolist()))
# Explicitreciprocalcurrentrelation, noidentificationofoppositearguments.
for row in details[:2]:
 q=np.array(row['p'])-np.array(row['n']);a=vertex(q);b=vertex(-q);ck('reciprocal '+row['channel'],np.linalg.norm(a-np.array([adj(x)for x in b]))/np.linalg.norm(a))
out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,points=details,candidate_imported=False);(P/'point_results.json').write_text(json.dumps(out,indent=2)+'\n');print('POINTPASS',out['passed'],'FAIL',len(out['failed']))
