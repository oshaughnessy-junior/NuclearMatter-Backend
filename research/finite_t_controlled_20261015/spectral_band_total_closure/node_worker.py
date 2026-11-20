from pathlib import Path
import sys, json, time, dataclasses
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
channel, order, panel, first = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
if channel not in C.LOCK['panels_MeV'] or order not in (12,20) or not 0 <= panel < 5 or first % 4 or not 0 <= first < order:
    raise ValueError('Only prelocked band jobs')
edges = C.LOCK['panels_MeV'][channel]
rule = C.W.EnergyRule.gauss(edges[panel], edges[panel+1], order)
stop = min(order, first+4)
path = C.ROOT/'raw'/f'{channel}_{order}_p{panel}_{first:02d}.json'
if path.exists(): raise RuntimeError('Refuse overwriting actual spectral job')
out = dict(**C.metadata(), channel=channel, order=order, panel_index=panel, all_band_edges_MeV=edges, panel_rule=dataclasses.asdict(rule), first_index=first, last_index_exclusive=stop, controls=dataclasses.asdict(C.FINE), worker_sha256=C.sha(Path(__file__)), common_harness_sha256=C.sha(C.ROOT/'common.py'), rows=[], complete=False, whole_moment_qualified=False)
start=time.monotonic()
def save():
    out['seconds']=time.monotonic()-start
    path.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
save()
try:
    for index in range(first,stop):
        C.preflight(); begin=time.monotonic()
        row=C.W.spectral_coefficients(channel,C.BATH,rule.nodes[index],C.FINE)
        row.__post_init__(); C.verify()
        if row.j.log is None or row.true_chi.log is None or not row.complete: raise ArithmeticError('Positive integrated physical branch unresolved')
        if row.channel!=channel or row.bath!=C.BATH or row.controls!=C.FINE or row.energy!=rule.nodes[index] or row.differential_coordinates is not None:
            raise ArithmeticError('Changed actual coefficient identity')
        out['rows'].append(dict(index=index,energy_MeV=rule.nodes[index],weight_MeV=rule.weights[index],result=dataclasses.asdict(row),KMS_log_residual=row.kms_log_residual,seconds=time.monotonic()-begin));save()
    out['complete']=True;save()
except Exception as error:
    out['error']=dict(type=type(error).__name__,message=str(error));save()
print(channel,order,panel,first,'COMPLETE' if out['complete'] else out['error'],out['seconds'],flush=True)
