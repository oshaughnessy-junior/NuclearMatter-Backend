"""Independent electron/muon ideal pair states, with explicit MeV/fm units."""
import json

from nmbackend.constants import M_E, M_MU
from nmbackend.reference.finite_t_leptons import state_density


def main():
    temperature_mev = 20.0
    # Independent net NUMBER inputs: neither weak equilibrium nor neutrality
    # is imposed, and mu_e need not equal mu_mu. Both flavors share local T.
    electron = state_density(temperature_mev, 1e-6, mass_mev=M_E)
    muon = state_density(temperature_mev, .005, mass_mev=M_MU)
    print(json.dumps({
        "scope": "CPU ideal lepton pair components; no matter EOS or JAX AD",
        "units": {"temperature_chemical_potential_mass": "MeV",
                  "number_entropy_cv": "fm^-3 (k_B=1)",
                  "energy_pressure_free_energy": "MeV fm^-3",
                  "number_responses": "fm^-3 MeV^-1",
                  "dmu_dtemperature": "dimensionless"},
        "net_number_sign": "negative lepton minus positive antiparticle; charge=-e*n_net",
        "electron": electron.asdict(), "muon": muon.asdict(),
    }, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
