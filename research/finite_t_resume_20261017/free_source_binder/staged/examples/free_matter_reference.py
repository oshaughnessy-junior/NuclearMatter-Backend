"""Actual package free-matter instantaneous example; no evolution driver."""
import argparse
import json
from nmbackend.reference import free_matter as F
from nmbackend.reference import weak_reference as W

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--sources',action='store_true',help='Run actual fine total EC/PC/beta moments once')
parser.add_argument('--photons',choices=['absent','trapped_LTE'],default='absent')
args=parser.parse_args()
state=F.state_density(2.,1e-7,.3,photons=args.photons)
out=dict(model=state.model,state_identity=state.identity,T_MeV=state.temperature_mev,
         energy_MeV_fm3=state.energy_density,Cv_fm3=state.cv_fixed_nB_Ye,
         electron_net_fm3=state.electron.net_density,electron_fm3=state.electron.particle_density,
         positron_fm3=state.electron.antiparticle_density,muons_absent=state.muons_absent,
         photon_closure=state.photons,anti_log_fraction_bounds=[b.log_fraction_upper_bound for b in state.anti_bounds])
if args.sources:
    source=F.source_moments(state,radiation=W.Radiation.empty(),controls=F.SourceControls.fine())
    rhs=F.instantaneous_rhs(state,source,baryon_density_dot_fm3_s=2.5e-8)
    out.update(R_fm3_s=source.proton_number_source.value,Qescape_MeV_fm3_s=source.neutrino_energy_escape.value,
               Tdot_MeV_s=rhs.temperature_dot_mev_s,Yedot_s=rhs.Ye_dot_s,
               channel_inventory=source.complete_channels,source_identity=source.identity)
print(json.dumps(out,indent=2,allow_nan=False))
