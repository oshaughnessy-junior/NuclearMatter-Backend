"""Locked diagnostic gates. Failures/missing nodes never become zero or PASS."""
from pathlib import Path
import json,math
R=Path(__file__).resolve().parent
LOCK=json.loads((R/'LOCK.json').read_text())
checks=[];blocked=[];comparisons=[]
def check(name,error,gate,category):
    row=dict(name=name,error=float(error),gate=gate,category=category,passed=bool(math.isfinite(error) and abs(error)<=gate))
    checks.append(row);return row

def compare_moments(a,b,label,category):
    for branch in ('forward','backward'):
        for i in (0,1):
            x,y=a[branch][i],b[branch][i]
            if x['log'] is None or y['log'] is None:
                if not (branch=='backward' and x['log'] is None and y['log'] is None and x['value']==y['value']==0.):
                    raise ValueError('Falsezero/missing positive moment')
                continue
            check(f'{label} {branch}{i}',math.expm1(x['log']-y['log']),3e-6,category)

controls=json.loads((R/'CONTROL_ASSESSMENT.json').read_text())
inner_status={channel:True for channel in LOCK['panels_MeV']}
for row in controls:
    channel=row['channel']
    if not row['complete']:
        inner_status[channel]=False
        blocked.append(dict(category='inner_control_unresolved',channel=channel,energy=row['E'],variation=row['variation'],error=row['error'],path=row['path']))
    for branch,error in row.get('relative_changes',{}).items():
        c=check(f'{channel} E{row["E"]} {row["variation"]["name"]} {branch}',error,3e-6,'inner_control')
        if not c['passed']:inner_status[channel]=False

channel_status={};event_summary=[]
for channel in LOCK['panels_MeV']:
    path=R/f'{channel}_BAND_RESULTS.json'
    if not path.is_file():
        blocked.append(dict(category='missing_channel_result',channel=channel));continue
    rows=json.loads(path.read_text())
    for radiation in LOCK['radiation']:
        selected=[r for r in rows if r['radiation']==radiation]
        assert len(selected)==2
        selected.sort(key=lambda r:r['order']);a,b=selected
        assert [a['order'],b['order']]==[12,20]
        if a['complete_declared_band'] and b['complete_declared_band']:
            compare_moments(a['band_moments'],b['band_moments'],f'{channel} {radiation["name"]} BAND12to20','incident_band')
        else:
            blocked.append(dict(category='incomplete_band',channel=channel,radiation=radiation,orders=[r['order'] for r in selected if not r['complete_declared_band']],missing_nodes={str(r['order']):[dict(panel=p['panel'],indices=p['missing_indices']) for p in r['panels'] if not p['complete']] for r in selected}))
        for panel in range(5):
            x,y=a['panels'][panel],b['panels'][panel]
            if x['complete'] and y['complete']:
                compare_moments(x['moments'],y['moments'],f'{channel} {radiation["name"]} panel{panel}12to20','incident_panel')
        event_path=R/'events'/f'{channel}_{radiation["name"]}.json'
        if not event_path.is_file():
            blocked.append(dict(category='missing_fresh_event',channel=channel,radiation=radiation));continue
        event=json.loads(event_path.read_text())
        if not event['complete']:
            blocked.append(dict(category='fresh_event_incomplete',channel=channel,radiation=radiation,error=event.get('error')));continue
        if event['radiation']!=radiation or event['bath']!=LOCK['bath']:
            raise ValueError('Fresh event/spectral state or radiation mismatch')
        er=event['result']; event_summary.append(dict(channel=channel,radiation=radiation,forward=er['forward'][:2],backward=er['backward'][:2],net=er['net'][:2],actual_controls=event['actual_controls'],seconds=event['seconds']))
        if b['complete_declared_band']:
            compare_moments(b['band_moments'],er,f'{channel} {radiation["name"]} BAND20vsFRESH_EVENT','band_event')
            for i in (0,1):
                sb=b['band_moments'];gross=er['forward'][i]['value']+er['backward'][i]['value']
                check(f'{channel} {radiation["name"]} signednet{i} band-event gross',(sb['net'][i]['value']-er['net'][i]['value'])/gross,3e-6,'band_event_signed')
                if radiation['name']=='LTE':check(f'{channel} LTE BAND20 net{i}/gross',sb['net'][i]['value']/(sb['forward'][i]['value']+sb['backward'][i]['value']),2e-9,'LTE_gross')
            if radiation['mode']=='FD':
                z=LOCK['bath']['mu_p']+LOCK['bath']['mu_e']-LOCK['bath']['mu_n'];mu_eq=z if channel=='EC' else -z
                mu=radiation['mu_nue'] if channel=='EC' else radiation['mu_antinue']
                affinity=(mu_eq-mu)/radiation['T'];sb=b['band_moments'];gross=sb['forward'][0]['value']+sb['backward'][0]['value']
                check(f'{channel} {radiation["name"]} entropy-affinity',(sb['entropy']['value']-sb['net'][0]['value']*affinity)/gross,3e-9,'source_algebra')
        F=er['forward'];B=er['backward'];N=er['net']
        loss=N[4]['value']-N[3]['value']-N[2]['value'] if channel=='NEUTRON_DECAY' else N[2]['value']+N[3]['value']-N[4]['value'] if channel=='EC' else N[2]['value']+N[4]['value']-N[3]['value']
        grossQ=F[1]['value']+B[1]['value']
        check(f'{channel} {radiation["name"]} event species energy',(loss-N[1]['value'])/grossQ,3e-9,'event_ledger')
        check(f'{channel} {radiation["name"]} event directloss',(N[5]['value']-N[1]['value'])/grossQ,3e-9,'event_ledger')
    relevant=[c for c in checks if c['name'].startswith(channel+' ')]
    incomplete=any(v.get('channel')==channel and v['category'] in ('incomplete_band','missing_fresh_event','fresh_event_incomplete') for v in blocked)
    channel_status[channel]=dict(inner_prechecks_pass=inner_status[channel],has_incomplete_spectral_or_event_data=incomplete,all_available_numeric_gates_pass=all(c['passed'] for c in relevant),whole_moment_qualified=False,status='UNQUALIFIED diagnostic' if incomplete or not inner_status[channel] or any(not c['passed'] for c in relevant) else 'selected finite-band numerical agreement only; missing tails OPEN')

out=dict(kind='AUTHOR finite-band/event-total diagnostic; not infinite-energy qualification',checks=checks,passed=sum(c['passed'] for c in checks),failed=[c for c in checks if not c['passed']],blocked=blocked,channel_status=channel_status,fresh_event_totals=event_summary,whole_moment_qualified=False,missing_tails='captureE<.05/>60,betaE<.001/>1.4 nonzero/unbounded here',source_pins='SOURCE_PIN.json',protocol='LOCK.json',independent_review='PENDING')
(R/'ANALYSIS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print('PASS',out['passed'],'FAIL',len(out['failed']),'blocked',len(blocked));print(out['channel_status']);print([(c['name'],c['error']) for c in out['failed']])
