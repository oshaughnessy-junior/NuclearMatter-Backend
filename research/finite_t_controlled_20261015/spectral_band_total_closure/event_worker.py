from pathlib import Path
import sys, json, time, dataclasses
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
channel, scenario = sys.argv[1], int(sys.argv[2])
if channel not in C.LOCK['panels_MeV'] or not 0 <= scenario < 3: raise ValueError('Only locked actual event totals')
spec = C.LOCK['radiation'][scenario]
rad = C.radiation(spec)
controls = C.W.Controls.fine(channel)
path = C.ROOT/'events'/f'{channel}_{spec["name"]}.json'
if path.exists(): raise RuntimeError('Refuse overwriting actual event total')
out = dict(**C.metadata(), channel=channel, radiation=spec, actual_controls=dataclasses.asdict(controls), complete=False, kind='FRESH real-package StageA event-phase total; distinct phase from spectra')
path.write_text(json.dumps(out, indent=2)+'\n')
start=time.monotonic()
try:
    C.preflight()
    result=C.W.channel_moments(channel,C.BATH,rad,controls)
    C.verify()
    if any(v.log is None for v in result.forward[:2]): raise ArithmeticError('Positive event emission unresolved')
    if rad.mode=='FD' and any(v.log is None for v in result.backward[:2]): raise ArithmeticError('Positive FD event inverse unresolved')
    out.update(result=dataclasses.asdict(result), names=result.names, units=result.units, complete=True)
except Exception as error:
    out['error']=dict(type=type(error).__name__, message=str(error))
out['seconds']=time.monotonic()-start
path.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
print(channel, spec['name'], 'COMPLETE' if out['complete'] else out['error'], out['seconds'], flush=True)
