"""Explicit chemical free bath: detailed total channel moments, empty radiation."""
from nmbackend.reference import weak_reference as w


def main():
    # Chemical potentials from one independently solved neutral free-npe bath;
    # this API does not solve neutrality or beta equilibrium and imports no EOS.
    bath = w.FreeElectronBath(2., 919.8287578100662, 916.8449303084382, .17446333200826397)
    for channel in ('EC', 'PC', 'NEUTRON_DECAY'):
        result = w.channel_moments(channel, bath)
        print(channel, 'R [fm^-3s^-1]=', result.net[0].value,
              'Q [MeV fm^-3s^-1]=', result.net[1].value,
              'gross-normalized energy ledger=', result.energy_ledger_scaled)


if __name__ == '__main__':
    main()
