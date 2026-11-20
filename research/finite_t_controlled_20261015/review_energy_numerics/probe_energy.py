"""Bounded independent log-energy/angular integration of charged moments."""
from pathlib import Path
import sys,math,json,hashlib,time
import numpy as np
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'quantum_energy_moments'))
import integrated
b=integrated.slab
START=time.monotonic()
matter=b.matter_state(2.,7e-5,3e-5)
DOS=1/(2*math.pi**2*197.3269804**3)
LIGHT=2.99792458e10
MEV_ERG=1.602176634e-6
FIELDS=['emission_number','emission_energy','incoming_number','incoming_energy','outgoing_number','outgoing_energy','radiation_number_gain','radiation_energy_gain']
mus,mws=leggauss(80);mus=.5*(mus+1);mws=.5*mws
ENERGIES=[0.,.5,.78233336,1.29333236,2.,4.,8.,16.,32.,64.,128.,240.,320.]
checks=[];runs=[];failures=[]
paths=[Path(integrated.__file__),Path(b.__file__),ROOT/'quantum_slab/guarded_capture.py',ROOT/'nr_baryon_eos/free_nucleons.py']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}

def run(flavor,order):
    xs,ws=leggauss(order);total=np.zeros(8);panels=[]
    for Elo,Ehi in zip(ENERGIES[:-1],ENERGIES[1:]):
        ans=np.zeros(8);zlo,zhi=math.log1p(Elo),math.log1p(Ehi)
        for x,w in zip(xs,ws):
            if time.monotonic()-START>80:raise TimeoutError('Bounded independent experiment reached80seconds')
            z=.5*(zlo+zhi)+.5*(zhi-zlo)*x
            E=math.expm1(z);weight=.5*(zhi-zlo)*w*math.exp(z)
            c=b.coefficients(matter,flavor,float(E),order=48,tail=40.)
            left,right=b.boundary_occupations(E)
            tau=c['rate_s']*1e5/LIGHT
            att=np.exp(-tau/mus);escape=-np.expm1(-tau/mus);eq=c['j_s']/c['rate_s']
            outgoingL=eq*escape+right*att;outgoingR=eq*escape+left*att
            angular=float(np.dot(mws,mus*(outgoingL+outgoingR)))
            delta=float(np.dot(mws,mus*((eq-left)*escape+(eq-right)*escape)))
            A=LIGHT*DOS*E*E*1e39
            emit=DOS*E*E*1e39*c['j_s'];inc=A*.25*(left+right);out=A*.5*angular;net=A*.5*delta
            value=np.array([emit,E*emit,inc,E*inc,out,E*out,net,E*net])
            ans+=weight*value
        total+=ans;panels.append(dict(lo=Elo,hi=Ehi,values=ans.tolist()))
    result=dict(flavor=flavor,order=order,values=dict(zip(FIELDS,total.tolist())),panels=panels)
    runs.append(result)
    return total

summary=json.loads((ROOT/'quantum_energy_moments/summary.json').read_text())
def check(name,a,expected,tol=3e-5,scale=None):
    error=abs(a-expected)/max(abs(expected),scale or 1e-300)
    checks.append(dict(name=name,actual=float(a),expected=float(expected),error=float(error),passed=bool(error<tol)))

try:
    for flavor in ['nue','antinue']:
        coarse=run(flavor,8);fine=run(flavor,12)
        v=next(x['values']for x in summary['species']if x['state']=='dilute'and x['flavor']==flavor)
        reference=[v['emission_number_cm3_s'],v['emission_energy_MeV_cm3_s'],v['incoming_left_number_cm2_s']+v['incoming_right_number_cm2_s'],v['incoming_left_energy_MeV_cm2_s']+v['incoming_right_energy_MeV_cm2_s'],v['outgoing_left_number_cm2_s']+v['outgoing_right_number_cm2_s'],v['outgoing_left_energy_MeV_cm2_s']+v['outgoing_right_energy_MeV_cm2_s'],v['radiation_number_gain_cm2_s'],v['radiation_energy_gain_MeV_cm2_s']]
        for i,name in enumerate(FIELDS):
            scale=max(abs(reference[i]),1e-300)
            check(flavor+' independent logE/angle '+name,fine[i],reference[i],3e-5)
            check(flavor+' logE refinement '+name,coarse[i],fine[i],3e-5)
        check(flavor+' direct boundary/volume number',fine[4]-fine[2],fine[6],1e-10,abs(fine[4])+abs(fine[2]))
        check(flavor+' direct boundary/volume energy',fine[5]-fine[3],fine[7],1e-10,abs(fine[5])+abs(fine[3]))
        check(flavor+' material total-energy sign+erg',-fine[7]*MEV_ERG,v['bath_energy_source_erg_cm2_s'],3e-5)
        lepton=-(1 if flavor=='nue'else-1)*fine[6]
        check(flavor+' material electron-lepton sign',lepton,v['bath_electron_lepton_source_cm2_s'],3e-5)
except Exception as e:
    failures.append(dict(kind=type(e).__name__,message=str(e)))

final={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
out=dict(passed=not failures and all(x['passed']for x in checks) and hashes==final,checks=checks,failures=failures,runs=runs,source_sha256=hashes,end_source_sha256=final,runtime_seconds=time.monotonic()-START,
         scope='Independent selected dilute full-inventory moment and fixed-bath slab integration; no moderate/heavy inventory promotion')
Path(__file__).with_name('energy_review.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(passed=out['passed'],checks=len(checks),failures=failures,failed=[x for x in checks if not x['passed']],runtime=out['runtime_seconds']),indent=2))
