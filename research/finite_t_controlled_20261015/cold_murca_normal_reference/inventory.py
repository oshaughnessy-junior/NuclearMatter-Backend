"""Reaction particle-count stoichiometry; supplies NO physical number rates."""
from pathlib import Path
import json

def inventory():
    events=[]
    for lepton in ['electron','muon']:
        for spectator in ['neutron','proton']:
            for direction in ['beta_decay','capture']:
                d=1 if direction=='beta_decay' else -1
                events.append(dict(id=spectator+'_'+lepton+'_'+direction,spectator=spectator,lepton=lepton,direction=direction,emitted='anti_nu_'+lepton if d==1 else 'nu_'+lepton,delta_Ne_per_event=d if lepton=='electron' else 0,delta_Nmu_per_event=d if lepton=='muon' else 0,delta_Np_per_event=d,delta_Nn_per_event=-d,delta_radiation_Le_per_event=-d if lepton=='electron' else 0,delta_radiation_Lmu_per_event=-d if lepton=='muon' else 0,baryon_count_delta=0,electric_charge_delta=0))
    return dict(units='Every delta_N*_per_event is an integer particle COUNT change per one directed reaction event; radiation Le/Lmu are signed lepton-number counts per event.',composition='Y_i=N_i/N_B=n_i/nB. For one event in volume V, delta Y_i=nu_i/(nB*V), NOT nu_i.',source_relation='If R_a is a separately calculated directed event-number rate [fm^-3 s^-1], dY_i/dt=sum_a nu_i,a R_a/nB [s^-1]. Radiation flavor-number source is sum_a nu_rad,a R_a [fm^-3 s^-1].',scope='This cold MU reference calculates energy emissivities Q only. It does NOT compute R_a, and Q cannot be substituted for R_a. Inventory is stoichiometry and a future integration convention.',events=events)

if __name__=='__main__':
    (Path(__file__).resolve().parent/'reaction_inventory.json').write_text(json.dumps(inventory(),indent=2)+'\n')
