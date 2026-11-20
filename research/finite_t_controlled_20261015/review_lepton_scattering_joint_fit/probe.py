"""Independent frozen v2 audit and fresh close-zero physics; no author drivers."""
from pathlib import Path
import importlib.util,sys,json,hashlib,math,time
import numpy as np
from scipy.optimize import brentq
from numpy.polynomial.chebyshev import chebval,chebfit
from energy_shell import integrate
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;P=ROOT/'lepton_scattering_joint_fit';sys.dont_write_bytecode=True

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m

def main():
    start=time.monotonic();dirs=['lepton_scattering_joint_fit','lepton_scattering_fits','review_lepton_scattering_fits','neutrino_lepton_scattering','leptonic_pair_annihilation','relativistic_capture_oracle']
    before={str(p.relative_to(ROOT)):sha(p) for d in dirs for p in (ROOT/d).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    assert sha(P/'MANIFEST.json')=='3d914218212a3e12ef62266f28b0a48c325d284379bb24e334b1de2ec544c01d'
    assert sha(P/'api.py')=='edf38db39a3776965c97e36592fc0a95ab4e95110f35002e2bc45f2794edc658'
    checks=[]
    def gate(name,err,tol):checks.append(dict(name=name,error=float(err),tolerance=tol,passed=bool(math.isfinite(err) and abs(err)<=tol)))
    m=json.loads((P/'model.json').read_text());reg=json.loads((P/'PREREGISTRATION.json').read_text());lock=json.loads((P/'FITTING_LOCK.json').read_text());tr=json.loads((P/'training.json').read_text())
    for key,file in [('model_sha','model.json'),('training_sha','training.json'),('preregistration_sha','PREREGISTRATION.json')]:gate('locked '+file,0 if sha(P/file)==lock[key] else 1,0)
    E=np.array([r['E'] for r in tr['rows']]);x=2*np.log(E)/math.log(30)-1
    gate('actual training matches prereg',0 if list(E)==reg['training_MeV'] else 1,0)
    gate('fresh eight distinct from training',0 if len(reg['fresh_heldout_MeV'])==8 and set(E).isdisjoint(reg['fresh_heldout_MeV']) else 1,0)
    for k in ['rate','transport_rate','g']:
        y=np.array([r['fine']['total']['neutrino_energy_change']/(2*r['fine']['total']['rate']) if k=='g' else math.log(r['fine']['total'][k]) for r in tr['rows']]);passing=[]
        for deg in reg['candidate_degrees']:
            c=chebfit(x,y,deg);a=chebval(x,c)-y;loo=[]
            for i in range(len(E)):
                use=np.arange(len(E))!=i;loo.append(chebval(x[i],chebfit(x[use],y[use],deg))-y[i])
            if k!='g':a=np.expm1(a);loo=np.expm1(loo)
            good=max(abs(a))<=(.0005 if k=='g' else .002) and max(abs(np.array(loo)))<=(.002 if k=='g' else .005)
            if good:passing.append(deg)
        selected=m['models'][k]['selected'];gate('smallest passing degree '+k,selected['degree']-min(passing),0)
        gate('training-only coefficients '+k,np.max(abs(chebfit(x,y,selected['degree'])-selected['coefficients'])),2e-13)
    gate('18 coefficients',sum(len(z['selected']['coefficients']) for z in m['models'].values())-18,0)
    api=load('review_joint_fit_frozen_api',P/'api.py');cg=m['models']['g']['selected']['coefficients']
    g=lambda e:float(chebval(2*math.log(e)/math.log(30)-1,cg))
    boundaries={str(target):brentq(lambda e:g(e)-target,7.,9.) for target in [.08,0.,-.08]}
    # These new points were selected after coefficient freeze, before physics.
    held=[8.1137,8.1293]
    known=set(reg['training_MeV']+reg['fresh_heldout_MeV'])
    gate('new energies not author train/heldouts',0 if set(held).isdisjoint(known) else 1,0)
    for e in [1.,30.,7.2,9.4]:
        a=api.evaluate(e,method='diagnostic_fit');gate('joint identity '+str(e),a['outgoing_energy_moment']-a['rate']*(e+2*a['g']),1e-12)
    rejects=[]
    bad=[('boolE',True,{}),('numpyE',np.float64(8.),{}),('below',.999999,{}),('above',30.000001,{}),('nan',float('nan'),{}),('boolT',8.,{'T':True}),('changedT',8.,{'T':2.0000001}),('changedmu',8.,{'mu_e':api.MU+1e-10}),('targetlist',8.,{'targets':['e-','e+']}),('wrongflavor',8.,{'flavor':'anti_nu_e'}),('blocked',8.,{'outgoing':'FD'}),('method',8.,{'method':'cached'})]
    for name,e,kw in bad:
        try:api.evaluate(e,**kw);ok=False;why='accepted'
        except (ValueError,TypeError):ok=True;why='rejected'
        gate('reject '+name,0 if ok else 1,0);rejects.append(dict(name=name,rejected=ok))
    # Actual auto-fallback at a fresh close-zero point, not a reference cache.
    physics=[]
    for e in held:
        actual=api.evaluate(e);fit=api.evaluate(e,method='diagnostic_fit')
        gate('actual fallback invoked '+str(e),0 if actual['fallback'] and actual['method']=='actual detailed CM quadrature' and actual['controls']==[32,40,40,48,80.] else 1,0)
        a=integrate(e,outer=14,angle=24,upper=150.,tail=70.);b=integrate(e,outer=20,angle=34,upper=180.,tail=90.)
        for key in ['rate','transport_rate','outgoing_energy_moment']:
            gate('independent shell refinement '+str((e,key)),a['total'][key]/b['total'][key]-1,1e-6)
            gate('independent shell vs actual fallback '+str((e,key)),b['total'][key]/actual[key]-1,1e-6)
            gate('raw fit positive 1percent '+str((e,key)),fit[key]/actual[key]-1,.01)
        scale=2*b['total']['rate'];dq=b['total']['neutrino_energy_change'];da=actual['neutrino_energy_change']
        gate('shell signed refinement RT '+str(e),(a['total']['neutrino_energy_change']-dq)/scale,1e-6)
        gate('actual signed vs shell RT '+str(e),(da-dq)/scale,1e-6)
        gate('raw fit signed unchanged gate '+str(e),(fit['neutrino_energy_change']-dq)/scale,.005)
        gate('actual fallback resolves fresh sign '+str(e),0 if da*dq>0 and abs(dq)>10*abs(da-dq) else 1,0)
        physics.append(dict(E=e,auto=actual,diagnostic_fit=fit,independent_coarse=a,independent_fine=b))
        (HERE/'partial.json').write_text(json.dumps(dict(checks=checks,physics=physics,boundaries=boundaries),indent=2)+'\n')
    # Band edge: diagnostic flag versus auto routing; use one physical inside
    # call, and nearby outside call which must remain on the fit path.
    edge=boundaries['0.08'];inside=edge+1e-7;outside=edge-1e-7
    inn=api.evaluate(inside);out=api.evaluate(outside);fi=api.evaluate(inside,method='diagnostic_fit')
    gate('inside positive-g boundary actual',0 if inn['fallback'] and abs(fi['g'])<=.08 else 1,0)
    gate('outside positive-g boundary fit',0 if not out['fallback'] and abs(out['g'])>.08 else 1,0)
    for key in ['rate','transport_rate','outgoing_energy_moment']:gate('fresh boundary physical fit residual '+key,fi[key]/inn[key]-1,.01)
    gate('fresh boundary signed physical residual',(fi['neutrino_energy_change']-inn['neutrino_energy_change'])/(2*inn['rate']),.005)
    after={p:sha(ROOT/p) for p in before};gate('all frozen bytes unchanged',0 if before==after else 1,0)
    result=dict(checks=checks,passed=sum(z['passed'] for z in checks),failed=[z for z in checks if not z['passed']],physics=physics,boundaries=boundaries,band_boundary=dict(inside=inside,outside=outside,auto_inside=inn,auto_outside=out,raw_inside=fi),rejections=rejects,frozen_before=before,frozen_after=after,seconds=time.monotonic()-start,qualification='selected independent new signs/near-boundary, no uniform sign/error certificate; fresh-process physics invocation')
    (HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','failed','seconds']}))

if __name__=='__main__':main()
