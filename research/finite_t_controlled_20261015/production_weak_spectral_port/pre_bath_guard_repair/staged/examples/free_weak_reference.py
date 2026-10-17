"""Explicit chemical free bath: detailed total channel moments, empty radiation."""
from nmbackend.reference import weak_reference as w


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectra",action="store_true",help="selected fixed-bath actual j/true chi")
    parser.add_argument("--finite-range",action="store_true",help="bounded2-node finite-range pilot, whole moments OPEN")
    args=parser.parse_args()
    # Chemical potentials from one independently solved neutral free-npe bath;
    # this API does not solve neutrality or beta equilibrium and imports no EOS.
    bath = w.FreeElectronBath(2., 919.8287578100662, 916.8449303084382, .17446333200826397)
    if args.spectra:
        for channel, energy in [('EC',5.),('PC',5.),('NEUTRON_DECAY',.8)]:
            row=w.spectral_coefficients(channel,bath,energy,controls=w.SpectralControls())
            print(channel,'E [MeV]=',energy,'j [s^-1]=',row.j.value,
                  'TRUE chi [s^-1]=',row.true_chi.value,
                  'relaxation [s^-1]=',row.relaxation.value,
                  'true absorption [cm^-1]=',row.absorption_cm_inverse.value)
        return
    if args.finite_range:
        row=w.reconstruct_spectral_moments('EC',bath,w.Radiation.empty(),
                w.EnergyRule.gauss(2.,3.,2),w.SpectralControls())
        print('finite2–3MeV PILOT R/Q=',[v.value for v in row.forward],
              'whole moments qualified=',row.whole_moment_qualified)
        return
    for channel in ('EC', 'PC', 'NEUTRON_DECAY'):
        result = w.channel_moments(channel, bath)
        print(channel, 'R [fm^-3s^-1]=', result.net[0].value,
              'Q [MeV fm^-3s^-1]=', result.net[1].value,
              'gross-normalized energy ledger=', result.energy_ledger_scaled)


if __name__ == '__main__':
    main()
