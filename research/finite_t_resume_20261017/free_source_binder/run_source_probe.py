"""One bounded actual changing-state source experiment per process invocation."""
from pathlib import Path
from dataclasses import asdict
import json,sys,time,signal
from nmbackend.reference import free_matter as F
from nmbackend.reference import weak_reference as W
P=Path(__file__).resolve().parent
cases=[(2.,1e-7,.3,'absent'),(2.03,1.05e-7,.34,'trapped_LTE'),(1.7,2e-7,.21,'absent')]

def main(index):
    signal.alarm(88);start=time.monotonic();T,n,y,photons=cases[index]
    s=F.state_density(T,n,y,photons=photons)
    radiation=W.Radiation.empty() if index<2 else W.Radiation.fd(T,-1.3,2.1)
    src=F.source_moments(s,radiation=radiation,controls=F.SourceControls.fine())
    rhs=F.instantaneous_rhs(s,src,baryon_density_dot_fm3_s=2.5e-8)
    out=dict(state=asdict(s),state_identity=s.identity,source_identity=src.identity,channels=[asdict(r) for r in src.channel_results],R=asdict(src.proton_number_source),Q=asdict(src.neutrino_energy_escape),ledgers=asdict(src.ledgers),rhs=asdict(rhs),seconds=time.monotonic()-start,source_complete=src.complete_channels)
    (P/('source_'+str(index)+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(index=index,R=src.proton_number_source.value,Q=src.neutrino_energy_escape.value,Tdot=rhs.temperature_dot_mev_s,Yedot=rhs.Ye_dot_s,seconds=out['seconds'])))
    signal.alarm(0)

if __name__=='__main__':main(int(sys.argv[1]))
