"""Physical anchors and crossed3body spectrum refinement beforemomentreconstruction."""
from pathlib import Path
import importlib.util,sys,json,math,time
import numpy as np
P=Path(__file__).resolve().parent;sys.dont_write_bytecode=True

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
S=load('beta_new_pilot_spectrum',P/'spectrum.py');TR=load('beta_new_pilot_tensor',P/'tensor_reference.py')
def run():
 start=time.monotonic();rows=[];checks=[]
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 for E in [.2,.42,.5,.8]:
  vals=[S.integrate(E,parent_order=p,cone_order=c,electron_order=e,phi_order=f,tail=t)for p,c,e,f,t in[(8,10,10,6,50.),(12,16,16,10,70.)]]
  for k in ['j','chi']:gate('cone+parent refinement '+str((E,k)),vals[0][k]['value']/vals[1][k]['value']-1,3e-6)
  r=vals[-1];gate('directFD KMS '+str(E),r['log_kms_actual']-r['log_kms_expected'],3e-12);gate('positiveaboveendpoint '+str(E),0 if r['j']['value']>0 and r['chi']['value']>0 else 1,0)
  if E==.42:
   gate('oldaccepted .42j',r['j']['value']/.07083091288795236-1,3e-6);gate('oldaccepted .42chi',r['chi']['value']/.02144761484832465-1,3e-6)
  rows.append(dict(E=E,rules=vals));(P/'pilot_partial.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2)+'\n')
 # Fresh literaldecay/inversespin sum reciprocity before KMS-based checks.
 for enu in [.23,.63]:
  ss=S.MN**2-2*S.MN*enu;rt=math.sqrt(ss);Es=(ss+S.ME**2-S.MP**2)/(2*rt);ps=math.sqrt((Es-S.ME)*(Es+S.ME));z=np.array([-.61,.27]);g=(S.MN-enu)/rt;v=enu/(S.MN-enu);El=g*(Es-v*ps*z);px=ps*np.sqrt(1-z*z);pz=g*(ps*z-v*Es);n=np.tile([S.MN,0,0,0],(2,1));nu=np.tile([enu,0,0,enu],(2,1));e=np.column_stack([El,px,np.zeros(2),pz]);p=n-e-nu;f=S.B.amplitude(n,p,e,nu);b=S.B.amplitude(n,p,e,nu,inverse=True)
  for i in range(2):gate('actualreciprocalfullcurrent '+str((enu,i)),b[i]/f[i]-1,3e-12)
 # Onlynew .8wholelepton/tensorreference initially, beforebroadenergycollection.
 refs=[TR.reference(S,.8,angle=a,lepton=l,energy=e,phi=f,widths=wi,tail=t)for a,l,e,f,wi,t in[(6,8,12,4,14.,60.),(10,12,20,6,20.,80.)]]
 r=rows[-1]['rules'][-1]
 for k in ['j','chi']:gate('newtensor aboveendpointrefine '+k,refs[0][k]['value']/refs[1][k]['value']-1,3e-6);gate('absolutecrossedcone/tensor aboveendpoint '+k,r[k]['value']/refs[1][k]['value']-1,3e-6)
 out=dict(rows=rows,checks=checks,tensor_aboveendpoint=refs,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],seconds=time.monotonic()-start);(P/'pilot_results.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k]for k in ['passed','failed','seconds']})
if __name__=='__main__':run()
