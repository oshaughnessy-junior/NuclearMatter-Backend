"""Independent local Jacobian to the original PR55 delta-shell coordinates."""
from pathlib import Path
import math,json
import numpy as np
from conditional_pc import minimum,BATH
from pr55_reference import _weak_current as C

def event(E,z,u,co,phi):
 T=BATH[0];D,emin,pmin=minimum(E);ea=emin+T*z;pa=math.sqrt((ea-C.MP)*(ea+C.MP));R=T*z*(1+(ea+emin)/(pa+pmin));width=R/pa
 ca=-1+width*u;sint=math.sqrt(width*u*(2-width*u))
 a=np.array([[ea,0.,0.,pa]]);k=np.array([[E,E*sint,0.,E*ca]])
 excess=2*E*R*(1-u);s=(C.MN+C.ME)**2+excess;root=math.sqrt(s);pf=math.sqrt(excess*(excess+4*C.MN*C.ME))/(2*root)
 total=a+k;beta=total[:,1:]/total[:,0,None];kc=C.boost(k,-beta);axis=kc[:,1:]/np.linalg.norm(kc[:,1:],axis=1)[:,None];ey=np.array([[0.,1.,0.]]);ex=np.cross(ey,axis)
 direction=co*axis+math.sqrt(1-co*co)*(math.cos(phi)*ex+math.sin(phi)*ey)
 el=(s+C.ME*C.ME-C.MN*C.MN)/(2*root);l=C.boost(np.column_stack([[el],pf*direction]),beta)
 n=total-l
 qv=k[0,1:]-l[0,1:];q=np.linalg.norm(qv);qh=qv/q;kp=k[0,1:]-np.dot(k[0,1:],qh)*qh;xx=kp/np.linalg.norm(kp);yy=np.cross(qh,xx)
 phia=math.atan2(np.dot(a[0,1:],yy),np.dot(a[0,1:],xx));pe=np.linalg.norm(l[0,1:]);cos=float(np.dot(k[0,1:],l[0,1:])/(E*pe))
 return np.array([l[0,0],cos,phia]),width*pa*pf*q/(pe*root),{'proton':a[0].tolist(),'nu':k[0].tolist(),'lepton':l[0].tolist(),'neutron':n[0].tolist(),'q':float(q),'omega':float(E-l[0,0])}

rows=[]
for E in (.1,.15):
 z=2.3;point=np.array([.37,.21,1.1]);out,pred,evt=event(E,z,*point)
 for h in (.01,.003,.001,.0003):
  jac=np.empty((3,3))
  for j in range(3):
   d=np.zeros(3);d[j]=h
   yp=event(E,z,*(point+d))[0];ym=event(E,z,*(point-d))[0];diff=yp-ym;diff[2]=(diff[2]+math.pi)%(2*math.pi)-math.pi;jac[:,j]=diff/(2*h)
  numerical=abs(float(np.linalg.det(jac)));rows.append({'E':E,'z':z,'physical_u_co_phi':point.tolist(),'h':h,'analytic_measure_jacobian':pred,'numerical_jacobian':numerical,'relative_error':numerical/pred-1,'old_coordinates_El_cosphi_phi':out.tolist(),'event':evt})
  print(E,h,numerical/pred-1)
Path(__file__).with_name('PHASE_JACOBIAN_RAW.json').write_text(json.dumps(rows,indent=2)+'\n')
