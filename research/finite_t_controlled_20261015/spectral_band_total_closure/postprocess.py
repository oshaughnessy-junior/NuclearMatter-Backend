"""Own log-safe finite-band source arithmetic; no physical coefficient calls."""
from pathlib import Path
import json, math, hashlib
import numpy as np
from numpy.polynomial.legendre import leggauss
R = Path(__file__).resolve().parent
CAMPAIGN = R.parent
LOCK = json.loads((R/'LOCK.json').read_text())
PINS = json.loads((R/'SOURCE_PIN.json').read_text())
HC = 197.3269804
CURRENT = 'Guo2020-v2-full-free-VA-F2-GP-MV840-MA1000-mpi139.57'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def verify():
    for name, h in PINS.items():
        if sha(CAMPAIGN/name) != h: raise RuntimeError('Frozen source changed: '+name)
def lse(values):
    m = max(values)
    return m+math.log(math.fsum(math.exp(v-m) for v in values))
def positive(log):
    if not math.isfinite(log): raise ValueError('Finite positive log required')
    try: value = math.exp(log)
    except OverflowError: value = math.inf
    reason = 'underflow' if value == 0 else 'overflow' if not math.isfinite(value) else 'represented'
    return dict(value=value if reason == 'represented' else None, log=log, reason=reason)
def logfd(x):
    return -math.log1p(math.exp(-x)) if x >= 0 else x-math.log1p(math.exp(x))
def collect_panel(channel, order, panel):
    verify()
    edges = LOCK['panels_MeV'][channel]
    lo, hi = edges[panel:panel+2]
    x,w = leggauss(order)
    energies = [float(v) for v in lo+(x+1)*(hi-lo)/2]
    weights = [float(v) for v in w*(hi-lo)/2]
    nodes, pins, missing = [], [], []
    for first in range(0,order,4):
        path = R/'raw'/f'{channel}_{order}_p{panel}_{first:02d}.json'
        if not path.is_file():
            missing.extend(range(first,min(order,first+4)));continue
        raw = path.read_bytes(); job = json.loads(raw)
        if job['lock_sha256'] != sha(R/'LOCK.json') or job['source_pin_sha256'] != sha(R/'SOURCE_PIN.json') or job['candidate_manifest_sha256'] != LOCK['candidate_manifest_sha256']:
            raise ValueError('Wrong source/state protocol')
        if job['worker_sha256'] != sha(R/'node_worker.py') or job['common_harness_sha256'] != sha(R/'common.py'):
            raise ValueError('Wrong actual execution harness')
        if job['channel'] != channel or job['order'] != order or job['panel_index'] != panel or job['all_band_edges_MeV'] != edges or job['controls'] != LOCK['controls_fine'] or job['bath'] != LOCK['bath']:
            raise ValueError('Changed channel/rule/controls/bath')
        if job['panel_rule'] != dict(nodes=energies, weights=weights, lower=lo, upper=hi):
            raise ValueError('Changed declared nodes/coverage')
        if job['first_index'] != first or job['last_index_exclusive'] != min(order,first+4):
            raise ValueError('Duplicate/overlapping job')
        for row in job['rows']:
            i = row['index']; a = row['result']
            if not first <= i < min(order,first+4) or row['energy_MeV'] != energies[i] or row['weight_MeV'] != weights[i]: raise ValueError('Wrong node identity')
            if a['channel'] != channel or a['bath'] != LOCK['bath'] or a['controls'] != LOCK['controls_fine'] or a['energy'] != energies[i] or a['current_source'] != CURRENT or a['units'] != 's^-1 per occupation' or a['differential_coordinates'] is not None or a['complete'] is not True:
                raise ValueError('Wrong integrated physical coefficient identity')
            for key in ('j','true_chi'):
                value = a[key]
                if value['support'] != 'positive_support' or isinstance(value['log'],bool) or not isinstance(value['log'],(int,float)) or not math.isfinite(value['log']) or isinstance(value['value'],bool): raise ValueError('Falsezero/unresolved branch')
                if positive(value['log']) != dict(value=value['value'],log=value['log'],reason=value['reason']): raise ValueError('Wrong coefficient representation')
            nodes.append(row)
        existing = {row['index'] for row in job['rows']}
        missing.extend(i for i in range(first,min(order,first+4)) if i not in existing)
        if job['complete'] is not True and not missing: raise ValueError('Incomplete flag cannot be promoted')
        pins.append(dict(path=str(path.relative_to(R)),sha256=hashlib.sha256(raw).hexdigest(),complete=job['complete']))
    indices = [row['index'] for row in nodes]
    if len(set(indices)) != len(indices): raise ValueError('Duplicate actual node')
    complete = not missing and len(nodes) == order
    return dict(channel=channel,order=order,panel=panel,range_MeV=[lo,hi],nodes=sorted(nodes,key=lambda a:a['index']),actual_job_pins=pins,missing_indices=sorted(set(missing)),complete=complete)

def assemble(channel, nodes, radiation):
    fs,bs = [[],[]],[[],[]]
    ns = [[],[]]; entropy = []
    for row in nodes:
        energy,weight = row['energy_MeV'],row['weight_MeV']
        a = row['result']
        base = math.log(weight)+2*math.log(energy)-math.log(2*math.pi**2*HC**3)
        if radiation['mode'] == 'empty':
            lf,lb = a['j']['log'],None
        else:
            mu = radiation['mu_nue'] if channel == 'EC' else radiation['mu_antinue']
            lf = a['j']['log']+logfd((energy-mu)/radiation['T'])
            lb = a['true_chi']['log']+logfd((mu-energy)/radiation['T'])
        delta = None if lb is None else lf-lb
        sign = 1 if lb is None else 1 if delta > 0 else -1 if delta < 0 else 0
        netlog = lf if lb is None else max(lf,lb)+math.log(-math.expm1(-abs(delta))) if sign else None
        if lb is not None and sign: entropy.append(base+netlog+math.log(abs(delta)))
        for i in (0,1):
            le = i*math.log(energy)
            fs[i].append(base+lf+le)
            if lb is not None:bs[i].append(base+lb+le)
            if sign:ns[i].append((base+netlog+le,sign))
    forward = [positive(lse(v)) for v in fs]
    backward = [positive(lse(v)) if v else dict(value=0.,log=None,reason='absent_initial_neutrino_occupation') for v in bs]
    net = []
    for values in ns:
        if not values: net.append(dict(value=0.,log_abs=None,sign=0));continue
        m = max(v[0] for v in values)
        v = math.fsum(s*math.exp(z-m) for z,s in values)
        if not v:net.append(dict(value=0.,log_abs=None,sign=0));continue
        q = positive(m+math.log(abs(v)))
        net.append(dict(value=None if q['value'] is None else math.copysign(q['value'],v),log_abs=q['log'],sign=1 if v > 0 else -1))
    ent = None if radiation['mode'] == 'empty' else positive(lse(entropy)) if entropy else dict(value=0.,log=None,reason='exact_numerical_cancellation')
    return dict(forward=forward,backward=backward,net=net,entropy=ent,actual_node_count=len(nodes),units=['fm^-3 s^-1','MeV fm^-3 s^-1'],whole_moment_qualified=False)

def process(channel):
    output = []
    for order in LOCK['incident_orders']:
        panels = [collect_panel(channel,order,i) for i in range(5)]
        complete = all(p['complete'] for p in panels)
        for radiation in LOCK['radiation']:
            panelrows = []
            for p in panels:
                row = {key:value for key,value in p.items() if key != 'nodes'}
                if p['complete']: row['moments'] = assemble(channel,p['nodes'],radiation)
                else: row['moments'] = None
                panelrows.append(row)
            band = assemble(channel,[v for p in panels for v in p['nodes']],radiation) if complete else None
            output.append(dict(kind='OWN detailed finite-band source postprocessing; never an official complete-source API record',channel=channel,order=order,radiation=radiation,bath=LOCK['bath'],controls=LOCK['controls_fine'],band_edges_MeV=LOCK['panels_MeV'][channel],panels=panelrows,complete_declared_band=complete,band_moments=band,source_pin_sha256=sha(R/'SOURCE_PIN.json'),lock_sha256=sha(R/'LOCK.json'),whole_moment_qualified=False,omitted_energy_ranges='nonzero low/high ranges remainOPEN; no rigorous tail bound'))
    verify()
    (R/f'{channel}_BAND_RESULTS.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    return output

if __name__ == '__main__':
    import sys
    rows=process(sys.argv[1]);print(sys.argv[1], 'completebands',sum(v['complete_declared_band'] for v in rows),'of',len(rows))
