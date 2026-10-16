"""Forward photon+pairs EOS, blackbody and derivative gates before sources."""
from pathlib import Path
import sys,importlib.util,json,math
from scipy.integrate import quad
P=Path(__file__).parent;s=importlib.util.spec_from_file_location('three_source_preflight_client',P/'client.py');M=importlib.util.module_from_spec(s);sys.modules[s.name]=M;s.loader.exec_module(M);C=M.Client();checks=[]
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
on=C.evaluate(2.,.3)['state'];off=C.evaluate(2.,.3,photons=False)['state'];HC=197.3269804
I=quad(lambda y:(-math.log(y))**3/(1-y)if y<1 else 0,0,1,epsabs=1e-11,epsrel=1e-12)[0];eg=2.**4/(math.pi**2*HC**3)*I
ck('two-photon-polarization Planck normalization',on['photon']['energy_density']/eg-1,3e-12);ck('photon pressure',on['photon']['pressure']/(eg/3)-1,3e-12);ck('photon entropy',on['photon']['entropy_density']/(4*eg/6)-1,3e-12);ck('photon Cv',on['photon']['cv']/(4*eg/2)-1,3e-12)
h=1e-4;plus=C.evaluate(2+h,.3)['state'];minus=C.evaluate(2-h,.3)['state'];yp=C.evaluate(2.,.3+h)['state'];ym=C.evaluate(2.,.3-h)['state'];ck('fullEOS uT forward finite difference',(plus['u']-minus['u'])/(2*h)/on['u_T']-1,3e-7);ck('fullEOS uY forward finite difference',(yp['u']-ym['u'])/(2*h)-on['u_Y'],3e-7);ck('photon Cv finite difference',(plus['photon']['energy_density']-minus['photon']['energy_density'])/(2*h)/on['photon']['cv']-1,3e-7)
ck('photons add energy once',(on['energy_density']-off['energy_density'])/eg-1,3e-12);ck('photons add Cv once',(on['cv']-off['cv'])/on['photon']['cv']-1,3e-12);ck('photons independent composition derivative',on['u_Y']-off['u_Y'],0)
euler=on['energy_density']+on['pressure']-2*on['entropy_density']-sum(on['material'][k]['chemical_potential']*on['material'][k]['density']for k in['neutron','proton'])-on['material']['electron']['chemical_potential']*1e-7*.3;ck('restinclusive fullEuler identity',euler/on['energy_density'],3e-12)
out=dict(checks=checks,passed=sum(z['passed']for z in checks),failed=[z for z in checks if not z['passed']],initial=on,photon_off_initial=off,finite_difference=dict(Tplus=plus,Tminus=minus,Yplus=yp,Yminus=ym),Planck_integral=I,client_calls=C.calls);(P/'preflight.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed'],u=on['u'],uT=on['u_T'],uY=on['u_Y']),indent=2))
if out['failed']:raise SystemExit(1)
