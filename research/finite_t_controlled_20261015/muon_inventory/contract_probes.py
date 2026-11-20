"""Muon contract checks, not a muon weak-rate implementation. No new prefactor."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'thermodynamics'))
import lepton_reference as L

# Charges: baryon number, electric charge, electron family, muon family.
charges = {'n':(1,0,0,0),'p':(1,1,0,0),'e-':(0,-1,1,0),
           'e+':(0,1,-1,0),'mu-':(0,-1,0,1),'mu+':(0,1,0,-1),
           'nue':(0,0,1,0),'anue':(0,0,-1,0),
           'numu':(0,0,0,1),'anumu':(0,0,0,-1)}
reactions = []
for l, nu, anu in [('e','nue','anue'),('mu','numu','anumu')]:
    reactions.extend([([l+'-','p'],['n',nu]),([l+'+','n'],['p',anu]),
                      (['n'],['p',l+'-',anu]),(['p'],['n',l+'+',nu])])
reactions += [(['mu-'],['e-','anue','numu']),(['mu+'],['e+','nue','anumu']),
              (['nue','mu-'],['numu','e-']),(['anue','mu+'],['anumu','e+']),
              (['anumu','mu-'],['anue','e-']),(['numu','mu+'],['nue','e+']),
              (['e-','mu+'],['nue','anumu']),(['e+','mu-'],['anue','numu'])]

def main():
    out = {'scope':'EOS/stoichiometry/vacuum support only; no weak rates', 'checks':[]}
    def gate(name, residual, tolerance):
        out['checks'].append(dict(name=name,residual=float(residual),tolerance=tolerance,
                                  passed=bool(abs(residual)<=tolerance)))
    for a,b in reactions:
        residual = [sum(charges[x][k] for x in b)-sum(charges[x][k] for x in a) for k in range(4)]
        gate('conserved:'+ '+'.join(a)+'<->'+'+'.join(b),max(map(abs,residual)),0)
    out['states']=[]
    for T in (5.,30.):
        nb,Ye,Ymu=.16,.06,.02
        e,m=L.state_density(T,nb*Ye,L.ELECTRON_MASS),L.state_density(T,nb*Ymu,L.MUON_MASS)
        pair=L.state_mu(T,0.,L.MUON_MASS)
        cp=L.state_mu(T,-m.chemical_potential,L.MUON_MASS)
        gate(f'charge:T{T}',(e.net_density+m.net_density)/(nb*(Ye+Ymu))-1,2e-10)
        gate(f'muonCP:T{T}',cp.net_density/m.net_density+1,2e-12)
        gate(f'GibbsMuon:T{T}',(m.energy_density+m.pressure-T*m.entropy_density-m.chemical_potential*m.net_density)/m.energy_density,2e-11)
        gate(f'pairSymmetry:T{T}',pair.particle_density/pair.antiparticle_density-1,2e-12)
        out['states'].append(dict(T=T,nB=nb,Ye=Ye,Ymu=Ymu,Yp=Ye+Ymu,
                                   electron=e.asdict(),muon=m.asdict(),zero_net_muon_pairs=pair.asdict(),
                                   prescribed_composition_not_weak_equilibrium=True))
    mn,mp=939.56542052,938.27208816
    out['vacuum_threshold_MeV']={
        'nu_mu_on_rest_neutron':((mp+L.MUON_MASS)**2-mn**2)/(2*mn),
        'antinu_mu_on_rest_proton':((mn+L.MUON_MASS)**2-mp**2)/(2*mp),
        'static_nu_mu_threshold':L.MUON_MASS-(mn-mp),
        'vacuum_neutron_muon_decay_energy_excess':mn-mp-L.MUON_MASS,
        'inverse_muon_decay_nu_mu_on_rest_electron':(L.MUON_MASS**2-L.ELECTRON_MASS**2)/(2*L.ELECTRON_MASS)}
    gate('vacuumNeutronMuonDecayForbidden',max(0.,out['vacuum_threshold_MeV']['vacuum_neutron_muon_decay_energy_excess']),0)
    out['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'thermodynamics/lepton_reference.py']}
    out['passed']=all(x['passed'] for x in out['checks'])
    (Path(__file__).parent/'contract_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'passed':out['passed'],'checks':len(out['checks']),'thresholds':out['vacuum_threshold_MeV']}))
    if not out['passed']:raise SystemExit(1)

if __name__=='__main__':main()
