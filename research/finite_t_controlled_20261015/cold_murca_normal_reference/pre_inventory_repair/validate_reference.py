"""Bounded source-normalization/angular/support controls; no frozen drivers."""
from pathlib import Path
from dataclasses import replace
import json,math,itertools,time,hashlib
from scipy.optimize import brentq
from normal_reference import State,evaluate,angular_proton,angular_neutron,HC,ME,MM,MN,MP,KB
from angular_oracle import angular
HERE=Path(__file__).resolve().parent

def neutral(nb,yp,T=6.37e8):
    kp=(3*math.pi**2*nb*yp)**(1/3);kn=(3*math.pi**2*nb*(1-yp))**(1/3)
    def muon(ke):return math.sqrt(max(ke*ke+(ME*ME-MM*MM)/HC**2,0.))
    ke=brentq(lambda ke:ke**3+muon(ke)**3-kp**3,0,kp,xtol=1e-14)
    return State(T,kn,kp,ke,muon(ke),.731,.863,math.hypot(HC*ke,ME))

def main():
    start=time.monotonic();checks=[];angle_rows=[];rows=[];invalid=[]
    def gate(name,err,tol=3e-12):checks.append(dict(name=name,error=float(err),tolerance=tol,passed=bool(math.isfinite(err) and abs(err)<=tol)))
    for kn,kp,kl in [(2.4,1,.3),(2.69,1,.3),(2.7,1,.3),(2.71,1,.3),(2.9,1,.3),(3.29,1,.3),(3.3,1,.3),(3.31,1,.3),(1.8,.8,.6),(2.01,.5,.5)]:
        analytic,region=angular_proton(kn,kp,kl)
        for order in [(kn,kp,kp,kp,kl),(kp,kl,kn,kp,kp),(kp,kp,kl,kp,kn)]:
            oracle=angular(order);gate('absolute proton angular '+str(order),(oracle['angular']-analytic)/max(abs(analytic),1.))
        an=angular_neutron(kn,kp,kl);on=angular([kn,kn,kn,kp,kl])
        gate('absolute neutron angular '+str((kn,kp,kl)),on['angular']/an-1)
        G=(analytic/an)*(kp/kn)**2
        wanted=0. if kn>=3*kp+kl else ((kl+3*kp-kn)**2/(8*kl*kp) if kn>=3*kp-kl else (3-kn/kp)/2)
        gate('phase-space remapping '+str((kn,kp,kl)),G-wanted)
        angle_rows.append(dict(kn=kn,kp=kp,kl=kl,analytic=analytic,region=region,independent=oracle,geometry_ratio=G))
    # The 2^3 in Eq15 is verified by absolute phase-space quadrature, not OCR.
    kp=1.;kl=.3;bound=3*kp-kl
    for h in [1e-3,1e-4,1e-5]:
        fn=lambda x:angular_proton(x,kp,kl)[0]/angular_neutron(x,kp,kl)*(kp/x)**2
        left=(fn(bound)-fn(bound-h))/h;right=(fn(bound+h)-fn(bound))/h
        gate('C1 internal region boundary '+str(h),(left-right)/(.5/kp),h/(2*kl*kp)+1e-10)
    for nb,yp in [(.1937,.0673),(.413,.0519)]:
        s=neutral(nb,yp);v=evaluate(s);rows.append(dict(state=vars(s),result=v))
        gate('charge '+str(nb),(s.kfe**3+s.kfmu**3)/s.kfp**3-1)
        gate('equal mu '+str(nb),math.hypot(HC*s.kfe,ME)/math.hypot(HC*s.kfmu,MM)-1)
        for fac in [.37,1.7]:
            w=evaluate(replace(s,T_K=s.T_K*fac));gate('T8 '+str((nb,fac)),w['total_Q']/v['total_Q']/fac**8-1)
        for species,field in [('n','mn_landau_ratio'),('p','mp_landau_ratio')]:
            fac=1.11;w=evaluate(replace(s,**{field:getattr(s,field)*fac}))
            for l in ['electron','muon']:
                for branch,power in [('neutron_Q',3 if species=='n' else 1),('proton_Q',1 if species=='n' else 3)]:gate('mass polynomial '+str((nb,species,l,branch)),w['channels'][l][branch]/v['channels'][l][branch]/fac**power-1)
        vf=s.kfmu/s.kfe
        gate('equalmu neutron flavor ratio '+str(nb),v['channels']['muon']['neutron_Q']/v['channels']['electron']['neutron_Q']/vf-1)
        # Source-source audit only; no fit and no forced agreement.
        alpha_page=1.76-.63*(1.68/s.kfn)**2
        page_neutron=8.55e21*s.mn_landau_ratio**3*s.mp_landau_ratio*(s.kfe+s.kfmu)/1.68*alpha_page*.68*(s.T_K/1e9)**8
        ratio=(8.1/8.55)*(1.68/(3*math.pi**2*.16)**(1/3))*(s.kfp/s.kfe)*(s.kfe*HC/s.delta_mu_np_MeV)*(1.13/alpha_page)
        gate('independent primary-source normalization remapping '+str(nb),(v['channels']['electron']['neutron_Q']+v['channels']['muon']['neutron_Q'])/page_neutron/ratio-1)
        rows[-1]['source_diagnostic']=dict(Page2004_neutron_Q=page_neutron,YKGH_to_Page_ratio=ratio,Page_alpha=alpha_page,interpretation='two distinct source approximations, not uncertainty bound or calibrated equality')
    s=State(1e9,2.01,.5,.5,0.,.78,.69,math.hypot(.5*HC,ME));v=evaluate(s)
    gate('physical forbidden proton exactlyzero',v['channels']['electron']['proton_Q'],0)
    gate('absent muon bothbranches exactlyzero',v['channels']['muon']['neutron_Q']+v['channels']['muon']['proton_Q'],0)
    gate('Tzero energy exactlyzero',evaluate(replace(s,T_K=0))['total_Q'],0)
    # Controls reject rather than floor/correct physics outside admitted slice.
    bad=[('legacy negativealpha extrapolation',replace(s,kfn=.8,kfp=.4,kfe=.4,delta_mu_np_MeV=math.hypot(.4*HC,ME))),('nonneutral',replace(s,kfe=.49)),('nonbeta',replace(s,delta_mu_np_MeV=s.delta_mu_np_MeV+1)),('warm',replace(s,T_K=1e11)),('bool',replace(s,T_K=True)),('NaN',replace(s,mn_landau_ratio=float('nan'))),('complex',replace(s,T_K=1+1j)),('missing coldmuon',replace(s,kfp=.8,kfe=.8,delta_mu_np_MeV=math.hypot(.8*HC,ME))),('negative T',replace(s,T_K=-1)),('endpoint proton',replace(s,kfp=0)),('DUopen',replace(s,kfn=.9+1.68,kfp=1.3,kfe=1.3,delta_mu_np_MeV=math.hypot(1.3*HC,ME)))]
    # DU-open control needs physical lepton closure; reconstruct at chosen kFp.
    x=neutral(.3,.2);bad[-1]=('DUopen',replace(x,kfn=1.68))
    for name,ss in bad:
        try:evaluate(ss); rejected=False;reason='accepted'
        except ValueError as e:rejected=True;reason=str(e)
        gate('reject '+name,0 if rejected else 1,0);invalid.append(dict(name=name,rejected=rejected,reason=reason))
    out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],angular=angle_rows,states=rows,forbidden_reference=dict(state=vars(s),result=v),rejections=invalid,seconds=time.monotonic()-start,source_sha256={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in [HERE/'normal_reference.py',HERE/'angular_oracle.py']})
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed','seconds']}))
    return len(out['failed'])

if __name__=='__main__':raise SystemExit(main())
