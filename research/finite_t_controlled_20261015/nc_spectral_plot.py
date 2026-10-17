"""Scientific plot of free neutron NC energy kernels; no evolution claim."""
from pathlib import Path
import os
import sys
import json
import hashlib
import math
import numpy as np
from scipy.integrate import quad
os.environ.setdefault('MPLCONFIGDIR','/tmp/nmb-finite-t-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'neutral_current'))
import free_nc as nc
OUT=ROOT/'papers'
OUT.mkdir(exist_ok=True)


def angular_q(pair,Ein,Eout,rtol):
    def integrand(q):
        u=(Ein*Ein+Eout*Eout-q*q)/(2*Ein*Eout)
        u=min(1.,max(-1.,u))  # endpoint roundoff only
        return q/(Ein*Eout)*nc.kernel(pair,Ein,Eout,u)
    return quad(integrand,abs(Ein-Eout),Ein+Eout,epsabs=1e-14,epsrel=rtol,limit=250)[0]


def main():
    Ein=10.
    energy=np.unique(np.r_[np.linspace(.2,8,50),np.linspace(8,12,161),np.linspace(12.1,20,50)])
    records=[];checks=[]
    fig,ax=plt.subplots(figsize=(7.3,4.5),layout='constrained')
    for T,n,color in ((2.,7e-5,'#18789b'),(2.,.002,'#bb4a30'),(.5,.005,'#5b4e9b')):
        sp=nc.state_density(T,n)
        pair=nc.pair_from_species(sp)
        vals=np.array([angular_q(pair,Ein,float(E),2e-8) for E in energy])
        for E in (7.3,9.5,10.,10.5,13.):
            a=angular_q(pair,Ein,E,2e-8);b=angular_q(pair,Ein,E,4e-11)
            checks.append(dict(T=T,n=n,Eout=E,relative_change=abs(a/b-1),passed=abs(a/b-1)<2e-7))
        records.append(dict(T_MeV=T,n_fm3=n,Ein_MeV=Ein,Eout_MeV=energy.tolist(),dGamma_dEout_s_inv_MeV_inv=vals.tolist()))
        ax.plot(energy,vals/n,color=color,label=f'T={T:g} MeV, n={n:g} fm⁻³',lw=1.7)
    ax.axvline(Ein,color='.6',ls='--',lw=.8)
    ax.set(xlabel='Outgoing neutrino energy [MeV]',ylabel=r'$(1/n)\,d\Gamma/dE_{out}$ [fm³ s⁻¹ MeV⁻¹]',yscale='log',ylim=(1e2,3e8),xlim=(.2,20),title='Free neutron scattering: incident energy 10 MeV')
    ax.grid(alpha=.15)
    ax.legend(frameon=False,fontsize=9)
    fig.text(.5,-.025,'Bare incident-event kernels; no final neutrino blocking or nuclear correlations',ha='center',fontsize=9)
    for ext in ('png','pdf'):fig.savefig(OUT/f'nc_energy_kernels.{ext}',dpi=180,bbox_inches='tight')
    report=dict(model='Free common-mass NR neutron FD response; leading NC current; no final neutrino blocking',records=records,checks=checks,source_sha256=hashlib.sha256(Path(nc.__file__).read_bytes()).hexdigest())
    (OUT/'nc_energy_kernels.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=all(c['passed'] for c in checks),checks=len(checks),max_error=max(c['relative_change'] for c in checks))))
    if not all(c['passed'] for c in checks):raise SystemExit(1)

if __name__=='__main__':main()
