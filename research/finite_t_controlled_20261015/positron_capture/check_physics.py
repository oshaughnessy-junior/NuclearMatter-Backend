"""Selected independent shell/spinor, exact vacuum threshold and moving tail."""
from pathlib import Path
import sys,json,math
import numpy as np
import pc_kernel as pc
sys.path.insert(0,str(pc.R/'relativistic_delta_oracle'));import oracle
sys.path.insert(0,str(pc.R/'relativistic_capture_oracle'));import events as ev
H=Path(__file__).parent;checks=[];rows=[]
def gate(name,error,tol):checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(math.isfinite(error)and abs(error)<=tol)))
b,metadata=pc.matched(2.,1e-7,.3)
# Independently reduced original delta with explicit external v spinors.
for Enu,El,c in [(5.,3.7,.3),(8.,6.5,-.4)]:
 p=oracle.Point(b.T,pc.crossed.MP,pc.crossed.MN,b.mu_p,b.mu_n,Enu,El,c,pc.ME,-b.mu_e,True,pc.crossed.MW)
 z=oracle.integrate(p,order=24,nphi=8,tail=60.)
 a=pc.differential(b,Enu,El,c,energy_order=24,phi_order=8,tail=60.)
 for key in ['j','chi']:gate('original delta absolute '+str(Enu)+' '+key,a[key]['value']/z[key]-1,3e-9)
 gate('original delta residual '+str(Enu),z['max_shell_residual'],3e-12)
 rows.append(dict(inputs=vars(p),oracle={k:v for k,v in z.items()if k not in ['W','lepton_tensor','contractions']},response=a))
# Exact free proton rest threshold is invariant, not a warm-bath cut.
ch=pc.V.Channel(pc.crossed.MP,pc.crossed.MN,pc.ME,True)
gate('physical rest threshold',ch.threshold/1.8060662080187608-1,3e-11)
gate('vacuum below threshold',float(pc.V.momenta(ch,1.7,0.)is not None),0)
gate('vacuum above threshold',float(pc.V.momenta(ch,2.,0.)is None),0)
# Moving initial proton: original CM construction at Eanti1.7 < rest threshold.
Enu=1.7;pz=-100.;Ea=math.hypot(pc.crossed.MP,pz)
k=np.array([[Enu,0,0,Enu]]);pa=np.array([[Ea,0,0,pz]]);P=k+pa
s=P[0,0]**2-np.sum(P[0,1:]**2);rt=math.sqrt(s)
gate('moving invariant support',float(s<=(pc.crossed.MN+pc.ME)**2),0)
beta=P[:,1:]/P[:,0,None];pf=math.sqrt((s-(pc.crossed.MN+pc.ME)**2)*(s-(pc.crossed.MN-pc.ME)**2))/(2*rt)
El=(s+pc.ME**2-pc.crossed.MN**2)/(2*rt)
lcm=np.array([[El,pf,0,0]]);pbcm=np.array([[rt-El,-pf,0,0]])
l=ev.boost(lcm,beta);pb=ev.boost(pbcm,beta)
c=float(l[0,3]/np.linalg.norm(l[0,1:]));El=float(l[0,0])
d=pc.differential(b,Enu,El,c,energy_order=24,phi_order=8,tail=60.)
eb=ev.Bath(b.T,pc.crossed.MP,pc.crossed.MN,b.mu_p,b.mu_n,-b.mu_e,Enu=Enu,ml=pc.ME,anti=True,Mweak=pc.crossed.MW)
amp=float(ev.amplitude(eb,k,pa,l,pb)[0])
gate('sub-rest moving full spinor amplitude positive',float(amp<=0),0)
gate('sub-rest moving response absorption positive',float(d['chi']['value']is None or d['chi']['value']<=0),0)
gate('sub-rest moving response emission positive',float(d['j']['value']is None or d['j']['value']<=0),0)
rows.append(dict(moving_witness=dict(Enu=Enu,proton_momentum_z=pz,proton_energy=Ea,s=s,positron_energy=El,cosine=c,amplitude_squared=amp,response=d)))
# Physical sign and units: KMS applies to chi, not chi+j.
muanti=b.mu_n-b.mu_p-b.mu_e
gate('moving direct KMS',d['j']['log']-d['chi']['log']+(Enu-muanti)/b.T,3e-11)
for name,fn in [('nonpositive Enu',lambda:pc.differential(b,0,2.,0)),('below positron rest energy',lambda:pc.differential(b,5,.1,0)),('boolean outer order',lambda:pc.integrate(b,5,angle_order=True)),('invalid muons-absent charge',lambda:pc.matched(2,1e-7,1.1))]:
 try:fn();fail=True
 except ValueError:fail=False
 gate('reject '+name,int(fail),0)
(H/'physics_checks.json').write_text(json.dumps(dict(checks=checks,rows=rows,all_pass=all(c['passed']for c in checks)),indent=2)+'\n')
print(json.dumps(dict(checks=len(checks),failed=[c for c in checks if not c['passed']])))
