"""Fresh boosted CM events against independent published-source invariants."""
from pathlib import Path
import sys,math,json,numpy as np
from scipy.special import log_expit
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'relativistic_capture_oracle'));import events as e
sys.path.insert(0,str(ROOT/'muon_full_current'));import current as F
V=F.V;checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and error<tol)))
for anti,Enu,p,ci,co,phi in [(False,157.,73.,-.42,.27,.91),(True,181.,91.,.17,-.38,1.13)]:
 ma,mb=(V.MP,V.MN)if anti else(V.MN,V.MP);b=e.Bath(2.7,ma,mb,933.,930.,-98. if anti else 98.,Enu,V.MMU,anti);Ea=math.hypot(ma,p);k=np.array([[Enu,0.,0.,Enu]]);pa=np.array([[Ea,p*math.sqrt(1-ci*ci),0.,p*ci]]);Ptot=k+pa;s=Ptot[0,0]**2-Ptot[0,1:]@Ptot[0,1:];rt=math.sqrt(s);beta=Ptot[:,1:]/Ptot[:,0,None];kc=e.boost(k,-beta);axis=kc[0,1:]/np.linalg.norm(kc[0,1:]);e2=np.array([0.,1.,0.]);e1=np.cross(e2,axis);direction=co*axis+math.sqrt(1-co*co)*(math.cos(phi)*e1+math.sin(phi)*e2);pf=math.sqrt((s-(mb+V.MMU)**2)*(s-(mb-V.MMU)**2))/(2*rt);El=(s+V.MMU**2-mb*mb)/(2*rt)
 l=e.boost(np.r_[El,pf*direction][None,:],beta);pb=e.boost(np.r_[rt-El,-pf*direction][None,:],beta);amp=e.amplitude(b,k,pa,l,pb)[0];mom=dict(k=k[0],pa=pa[0],l=l[0],pb=pb[0]);ch=V.Channel(ma,mb,V.MMU,anti);q=k[0]-l[0];ff=F.FormFactors().values(V.dot(q,q),b.Mweak);published=sum(F.guo_invariants(ch,mom,ff).values());trace=F.trace(ch,mom,ff)
 gate('new CM sourcepolynomial amplitude '+str(anti),abs(amp/published-1),3e-11);gate('new CM independenttrace amplitude '+str(anti),abs(amp/trace-1),3e-11)
 gate('new CM boostedenergy conservation '+str(anti),np.max(abs(k+pa-l-pb))/max(Enu,ma),3e-14)
 for label,v,mass in [('k',k[0],0.),('pa',pa[0],ma),('l',l[0],V.MMU),('pb',pb[0],mb)]:gate('new CM massshell '+str(anti)+label,abs(V.dot(v,v)-mass*mass)/max(v[0]**2,1.),3e-14)
 Erest=V.dot(k[0],pa[0])/ma;pi=(s-ma*ma)/(2*rt);ds_event=amp/(64*math.pi**2*s)*pf/pi*2*math.pi*V.HBARC**2*1e-26;ds_published=F.differential(ch,Erest,co,method='source');gate('new CM absolute crosssection '+str(anti),abs(ds_event/ds_published-1),3e-11)
 flux=1-p*ci/Ea;gate('new CM invariantMollerflux '+str(anti),abs(flux/(V.dot(k[0],pa[0])/(Enu*Ea))-1),3e-14)
 logs_abs=float(log_expit((b.mua-Ea)/b.T)+log_expit((pb[0,0]-b.mub)/b.T)+log_expit((l[0,0]-b.mul)/b.T));logs_inv=float(log_expit((b.mub-pb[0,0])/b.T)+log_expit((b.mul-l[0,0])/b.T)+log_expit((Ea-b.mua)/b.T));expected=-(Enu-(b.mul+b.mub-b.mua))/b.T;gate('new original boostedFD blocking KMS '+str(anti),abs(logs_inv-logs_abs-expected),3e-11)
 rows.append(dict(anti=anti,amp_events=amp,amp_published=published,amp_trace=trace,s=s,Erest=Erest,bathframe_lepton_energy=float(l[0,0]),CM_lepton_energy=El,bathframe_finalbaryon_energy=float(pb[0,0]),flux=flux,crosssection_dcos_cm2=ds_event,log_abs_occupation=logs_abs,log_inv_occupation=logs_inv))
(P/'cm_results.json').write_text(json.dumps(dict(passed=all(c['passed']for c in checks),checks=checks,rows=rows),indent=2)+'\n');print(sum(c['passed']for c in checks),'/',len(checks))
for c in checks:
 if not c['passed']:print('FAIL',c)
