"""Independent Dirac/Pauli trace and optical reciprocity probes."""
from pathlib import Path
import sys,json,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'allowed_opacity'))
import allowed_capture as a

checks=[]
def add(name,error,tol):
    checks.append(dict(name=name,error=float(error),tolerance=tol,passed=bool(error<tol)))
I=np.eye(2,dtype=complex); Z=np.zeros((2,2),dtype=complex)
sig=[np.array([[0,1],[1,0]],complex),np.array([[0,-1j],[1j,0]],complex),
     np.array([[1,0],[0,-1]],complex)]
gam=[np.block([[I,Z],[Z,-I]])]+[np.block([[Z,s],[-s,Z]]) for s in sig]
g5=1j*gam[0]@gam[1]@gam[2]@gam[3]
PL2=np.eye(4)-g5
def slash(p):return p[0]*gam[0]-sum(p[i+1]*gam[i+1] for i in range(3))

worst=0.
for Ee in [a.ME*(1+1e-8),1.,10.]:
    p=math.sqrt(Ee*Ee-a.ME*a.ME)
    for c in [-.9,0.,.8]:
        k=np.array([7.,0.,0.,7.]); ell=np.array([Ee,p*math.sqrt(1-c*c),0.,p*c])
        L=np.array([np.trace((slash(ell)+a.ME*np.eye(4))@gam[i]@PL2@slash(k)@gam[i]@PL2)
                    for i in range(4)])
        # Independent spin-averaged static hadron traces.
        H=np.array([np.trace((a.GV*I)@(a.GV*I))/2]+
                   [np.trace((a.GA*s)@(a.GA*s))/2 for s in sig])
        W=(L@H).real/(8*k[0]*Ee)
        worst=max(worst,abs(W/a.angular_factor(Ee,c)-1))
add('9 explicit gamma-matrix and Pauli-spin angular contractions',worst,1e-13)

s=a.PairState(2.,939.,100.,104.,107.,106.)
opt=a.OpticalState(s,3.)
base=a.log_coefficients(opt,8.,energy_order=16,angle_order=16)
shift=a.PairState(s.T,s.mass,s.offset_a+111,s.offset_b+111,s.mu_a+111,s.mu_b+111)
shifted=a.log_coefficients(a.OpticalState(shift,3.),8.,energy_order=16,angle_order=16)
add('absolute optical common-energy-zero invariance',max(abs(np.array(base)-shifted)),1e-11)
add('arbitrary chemical-imbalance integrated KMS',abs(base[0]-base[1]+(8.-opt.mu_neutrino)/s.T),1e-11)
rev=a.OpticalState(s.reverse(),-3.)
reverse=a.log_coefficients(rev,5.,energy_order=16,angle_order=16)
add('reversed-species negative-lepton-mu integrated KMS',abs(reverse[0]-reverse[1]+(5.-rev.mu_neutrino)/s.T),1e-11)

out=dict(scope='Independent allowed-current targeted checks; no full suite replay',checks=checks,
         all_passed=all(c['passed'] for c in checks))
Path(__file__).with_name('allowed_probe_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
