"""Selected algebra tests; fresh scalar-index construction, not a physical rate oracle."""
from __future__ import annotations
import json,time,hashlib
from pathlib import Path
import numpy as np
import external_legs as w
HERE=Path(__file__).resolve().parent
checks=[]
def gate(name,error,tol=2e-12):
    checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<=tol)))
def compare(name,a,b,scale=None,tol=2e-12):
    scale=max(np.max(abs(a)),np.max(abs(b)),1e-30) if scale is None else scale
    gate(name,np.max(abs(a-b))/scale,tol)

def scalar_v(q,model):
    # Explicit spin/isospin indices; no imported strong operator or Kronecker.
    x,y,z=q;s=np.array([[z,x-1j*y],[x+1j*y,-z]])
    v=np.zeros((16,16),complex)
    for f in range(16):
      sf,tf=divmod(f,4);a,b=divmod(sf,2);c,d=divmod(tf,2)
      for i in range(16):
        si,ti=divmod(i,4);e,g=divmod(si,2);h,j=divmod(ti,2)
        iso=2*(c==j)*(d==h)-(c==h)*(d==j)
        v[f,i]=-model.coefficient*s[a,e]*s[b,g]*iso/(np.dot(q,q)+model.m_pi**2)
    return v

def scalar_j(line,component,p,Q,mass,outgoing,raising):
    v=np.zeros((16,16),complex)
    ps=[np.array([[0,1],[1,0]]),np.array([[0,-1j],[1j,0]]),np.diag([1,-1])]
    for f in range(16):
      sf,tf=divmod(f,4);fs=list(divmod(sf,2));ft=list(divmod(tf,2))
      for i in range(16):
        si,ti=divmod(i,4);ins=list(divmod(si,2));it=list(divmod(ti,2))
        if ft[1-line]!=it[1-line] or fs[1-line]!=ins[1-line]:continue
        if (it[line],ft[line])!=((1,0) if raising else (0,1)):continue
        if component[0]=='A':val=-1.26*ps['xyz'.index(component[1])][fs[line],ins[line]]
        else:
            val=float(fs[line]==ins[line])
            if component!='V0':
                ax='xyz'.index(component[1]);val*=(2*p[line+2 if outgoing else line,ax]+(1 if outgoing else -1)*Q[ax])/(2*mass)
        v[f,i]=val
    return v

def scalar_direct(p,omega,Q,component,model,raising):
    E=lambda x:np.dot(x,x)/(2*model.common_mass)
    result=np.zeros((16,16),complex)
    for line in (0,1):
        spectator=1-line
        v=scalar_v(p[spectator+2]-p[spectator],model)
        di=E(p[line])-omega-E(p[line]-Q)
        do=E(p[line+2])+omega-E(p[line+2]+Q)
        result+=v@scalar_j(line,component,p,Q,model.common_mass,False,raising)/di
        result+=scalar_j(line,component,p,Q,model.common_mass,True,raising)@v/do
    return result

def run():
    started=time.perf_counter();model=w.ope.Model();rows=[]
    points=[(100.,5.,[2.,-1.,3.],[.31,.72,-.41]),(135.,8.,[-3.,4.,2.],[.61,-.24,.45])]
    for j,(pmag,omega,Q,u) in enumerate(points):
      Q=np.array(Q);p=w.emission_point(pmag,omega,Q,u);_,_,ds=w.validate(p,omega,Q,model,.05)
      row=dict(p=p.tolist(),omega=omega,Q=Q.tolist(),denominators=ds.tolist(),channels={})
      for raising in (False,True):
       for comp in ('V0','Vx','Vy','Vz','Ax','Ay','Az'):
        a=w.amplitude(p,omega,Q,comp,raising=raising)
        b=scalar_direct(p,omega,Q,comp,model,raising)
        bs=scalar_direct(p[[0,1,3,2]],omega,Q,comp,model,raising)
        # Explicit permutation of output spin and iso labels instead of imported P.
        outswap=[]
        for f in range(16):
          s,t=divmod(f,4);s1,s2=divmod(s,2);t1,t2=divmod(t,2)
          outswap.append((2*s2+s1)*4+2*t2+t1)
        compare(f'point{j}_{raising}_{comp}_scalar_reference',a,b-bs[outswap])
        compare(f'point{j}_{raising}_{comp}_out_exchange',w.ope.EXCHANGE@w.amplitude(p[[0,1,3,2]],omega,Q,comp,raising=raising),-a)
        compare(f'point{j}_{raising}_{comp}_in_exchange',w.amplitude(p[[1,0,2,3]],omega,Q,comp,raising=raising)@w.ope.EXCHANGE,-a)
        # Charge selection: only one nucleon flips, forbidden total charge blocks zero.
        for ini,fin in [('nn','pp'),('np','np'),('pp','nn')]:
          gate(f'point{j}_{raising}_{comp}_forbidden_{ini}_{fin}',np.max(abs(w.channel(a,ini,fin)))/max(np.max(abs(a)),1e-30))
      vec=[w.amplitude(p,omega,Q,c) for c in ('V0','Vx','Vy','Vz')]
      defect=w.vector_defect_direct(p,omega,Q)
      defect-=w.ope.EXCHANGE@w.vector_defect_direct(p[[0,1,3,2]],omega,Q)
      compare(f'point{j}_exact_vector_divergence_defect',omega*vec[0]-sum(Q[k]*vec[k+1] for k in range(3)),defect)
      gate(f'point{j}_finiteQ_defect_NOTzero',float(np.max(abs(defect)))<=0,0.)
      row['vector_divergence_max_MeV^-2']=float(np.max(abs(defect)))
      for comp in ('V0','Ax','Ay','Az'):
        a=w.channel(w.amplitude(p,omega,Q,comp),'nn','np')
        row['channels'][comp]={'spin_sum_abs2_MeV^-6':float(np.sum(abs(a)**2)),'matrix_real':a.real.tolist(),'matrix_imag':a.imag.tolist()}
      lam=1.7;scaled=w.ope.Model(model.f_piNN,model.m_pi*lam,model.common_mass*lam)
      for comp in ('V0','Vx','Ax','Az'):
        compare(f'point{j}_{comp}_dimensional_scale',w.amplitude(p*lam,omega*lam,Q*lam,comp,model=scaled,pole_gap=.05*lam)*lam**3,w.amplitude(p,omega,Q,comp))
      rows.append(row)
    omega=5.;p=w.emission_point(100.,omega,[0,0,0],[.41,.53,.71]);q=p[2]-p[0];k=p[3]-p[0]
    astrong=w.ope.antisymmetrized(q,k)
    jtot=w.rho(0)+w.rho(1)
    compare('strong_isospin_scalar_total_charge_commutator',jtot@astrong,astrong@jtot)
    gate('soft_vector_zero',np.max(abs(w.amplitude(p,omega,[0,0,0],'V0')))/model.coefficient)
    for axis in 'xyz':
      jj=-1.26*sum(w.sigma(l,'xyz'.index(axis))@w.rho(l) for l in (0,1))
      expected=(jj@astrong-astrong@jj)/omega
      compare('soft_axial_'+axis,w.amplitude(p,omega,[0,0,0],'A'+axis),expected)
      gate('soft_axial_'+axis+'_nonzero',float(np.max(abs(expected)))<=0,0.)
    poleomega=(100*10-100/2)/model.common_mass
    polep=w.emission_point(100,poleomega,[0,0,10],[.2,.3,.4])
    bad=[('pole',lambda:w.amplitude(polep,poleomega,[0,0,10],'Ax')),
         ('gap_zero',lambda:w.amplitude(p,omega,[0,0,0],'Ax',pole_gap=0)),
         ('gap_nan',lambda:w.amplitude(p,omega,[0,0,0],'Ax',pole_gap=float('nan'))),
         ('omega_nan',lambda:w.amplitude(p,float('nan'),[0,0,0],'Ax')),
         ('bad_energy',lambda:w.amplitude(p,omega+.01,[0,0,0],'Ax')),
         ('bad_momentum',lambda:w.amplitude(p,omega,[0,0,.01],'Ax')),
         ('bad_component',lambda:w.amplitude(p,omega,[0,0,0],'A0')),
         ('coupling_inf',lambda:w.amplitude(p,omega,[0,0,0],'Ax',ga=float('inf')))]
    bad.extend([('raising_nan',lambda:w.amplitude(p,omega,[0,0,0],'Ax',raising=float('nan'))),
                ('complex_momentum',lambda:w.amplitude(p.astype(complex)+1j,omega,[0,0,0],'Ax')),
                ('overflow_invariant',lambda:w.amplitude(np.array([[0,0,1e308],[0,0,-1e308],[0,0,1e308],[0,0,-1e308]]),0.,[0,0,0],'Ax'))])
    rejection={}
    for name,fn in bad:
      try:fn();msg='NOT REJECTED'
      except ValueError as exc:msg=str(exc)
      gate('reject_'+name,0 if msg!='NOT REJECTED' else 1,0);rejection[name]=msg
    result=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=sum(not x['passed'] for x in checks),seconds=time.perf_counter()-started,points=rows,controls=rejection,
                scope='Algebra-only off-pole common-mass Born external legs; no completed MU current/rate',
                source_sha256=hashlib.sha256((HERE/'external_legs.py').read_bytes()).hexdigest())
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','failed','seconds','source_sha256')}))
    return result['failed']
if __name__=='__main__':raise SystemExit(run())
