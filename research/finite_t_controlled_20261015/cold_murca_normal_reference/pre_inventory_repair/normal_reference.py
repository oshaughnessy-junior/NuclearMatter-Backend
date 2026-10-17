"""Optional normal degenerate MU recipe: YKGH2001 normalization/Gusakov2002 geometry.

Research model, not a production correction or complete MU calculation.
All kF are fm^-1, T kelvin, Landau masses/bare masses dimensionless; Q cgs.
"""
from __future__ import annotations
from dataclasses import dataclass
import math,numbers
HC=197.3269804; KB=8.617333262145e-11
MN=939.56542052; MP=938.27208816; ME=.51099895069; MM=105.6583755
N0=.16; COEFFICIENT=8.1e21; ALPHA=1.13; BETA=.68

def real(x,name,*,positive=False):
    if isinstance(x,bool) or not isinstance(x,numbers.Real) or not math.isfinite(x) or (positive and x<=0):
        raise ValueError(name+' must be finite real'+(' positive' if positive else ''))
    return float(x)

def angular_proton(kn,kp,kl):
    """Exact angular A^p, neutrino solid angle included, fm^3.

    Complete regions on neutron-rich domain 0<kl<=kp<=kn; no matrix element.
    """
    kn=real(kn,'kn',positive=True);kp=real(kp,'kp',positive=True);kl=real(kl,'kl',positive=True)
    if not kl<=kp<=kn: raise ValueError('requires 0<kl<=kp<=kn')
    if kn>=3*kp+kl:return 0.,'forbidden or exact polygon edge'
    if kn>=3*kp-kl:
        return 2*(2*math.pi)**5*(kl+3*kp-kn)**2/(kn*kp**3*kl),'Gusakov2002 Eq14'
    return 8*(2*math.pi)**5/kp**2*(3/kn-1/kp),'Gusakov2002 Eq15'

def angular_neutron(kn,kp,kl):
    kn=real(kn,'kn',positive=True);kp=real(kp,'kp',positive=True);kl=real(kl,'kl',positive=True)
    if not 0<kl<=kp<=kn or not kn>kp+kl:raise ValueError('Eq13 restricted to neutron-rich direct-Urca-closed region')
    return 2*math.pi*(4*math.pi)**4/kn**3

@dataclass(frozen=True)
class State:
    T_K:float
    kfn:float
    kfp:float
    kfe:float
    kfmu:float
    mn_landau_ratio:float
    mp_landau_ratio:float
    delta_mu_np_MeV:float

def evaluate(s:State):
    if not isinstance(s,State):raise ValueError('State required')
    values={k:real(v,k) for k,v in vars(s).items()}
    T=values['T_K'];kn=s.kfn;kp=s.kfp
    if T<0 or not 0<kp<=kn or s.kfe<=0 or s.kfmu<0 or s.mn_landau_ratio<=0 or s.mp_landau_ratio<=0:raise ValueError('invalid sign/composition')
    # Deliberately narrow research admission, NOT an accuracy-certified fit band.
    if kn<1.68:raise ValueError('outside admitted core-density recipe slice: kFn>=1.68; no low-density alpha clipping')
    if abs((s.kfe**3+s.kfmu**3)/kp**3-1)>1e-10:raise ValueError('actual charge neutrality required')
    mun=math.hypot(s.kfe*HC,ME)
    if s.kfmu==0 and mun>MM+1e-8:raise ValueError('cold muon-equilibrated closure requires muons above onset')
    if abs(s.delta_mu_np_MeV-mun)>1e-8:raise ValueError('beta affinity must vanish: supplied mu_n-mu_p != mu_e')
    if s.kfmu>0 and abs(math.hypot(s.kfmu*HC,MM)-mun)>1e-8:raise ValueError('lepton chemical potentials must match')
    mn=s.mn_landau_ratio*MN;mp=s.mp_landau_ratio*MP
    if max(kn*HC/mn,kp*HC/mp)>.8:raise ValueError('outside admitted NR Fermi-velocity slice vF/c<=.8')
    out={};scales=[(kn*HC)**2/(2*mn),(kp*HC)**2/(2*mp),mun-ME]
    if s.kfmu>0:scales.append(mun-MM)
    if T*KB>.03*min(scales):raise ValueError('outside admitted strongly degenerate slice kBT<=.03 minimum kinetic scale')
    np_=kp**3/(3*math.pi**2)
    base=COEFFICIENT*s.mn_landau_ratio**3*s.mp_landau_ratio*(np_/N0)**(1/3)*ALPHA*BETA*(T/1e9)**8
    for lepton,kl,ml in [('electron',s.kfe,ME),('muon',s.kfmu,MM)]:
        if kl==0:
            out[lepton]=dict(neutron_Q=0.,proton_Q=0.,status='absent species; no artificial density floor')
            continue
        An=angular_neutron(kn,kp,kl);Ap,region=angular_proton(kn,kp,kl)
        # YKGH2001 Sec3.4d: vFl/c factor, including electron when not exactly c.
        vf=kl*HC/math.hypot(kl*HC,ml)
        qn=base*vf
        # From Eq10 phase-space products; common spin-summed matrix approximation.
        ratio_geom=(Ap/An)*(kp/kn)**2
        qp=qn*(mp/mn)**2*ratio_geom
        out[lepton]=dict(neutron_Q=qn,proton_Q=qp,neutron_angular=An,proton_angular=Ap,proton_region=region,geometry_ratio=ratio_geom,vF_over_c=vf,status='normal unpaired degenerate beta-equilibrium pair-emission energy')
    return dict(channels=out,total_Q=sum(x['neutron_Q']+x['proton_Q'] for x in out.values()),units='erg cm^-3 s^-1; both beta-decay antinu and capture nu energy combined',normalization='YKGH2001 Eq140:8.1e21,(np/.16)^(1/3),alpha1.13,beta.68; Sec3.4d vFl/c; equal branch matrix assumption Eq142',angular_source='Gusakov2002 Eqs13-15, neutron-rich kl<=kp<=kn; neutron branch only DU-closed',net_material_Le=0.,net_material_Lmu=0.,net_material_Yp=0.,net_radiation_Le=0.,net_radiation_Lmu=0.,ledger_status='zero number/lepton drift is an EQUILIBRIUM ASSUMPTION; no number moments calculated',minimum_kinetic_scale_MeV=min(scales),degeneracy_ratio=T*KB/min(scales),bare_masses_MeV=dict(n=MN,p=MP),missing='full strong/current matrix and uncertainty, warm or nonequilibrium kernel, pairing, width/DU matching, finite-Eν polygon smoothing')
