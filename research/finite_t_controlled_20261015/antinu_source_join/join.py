"""Join frozen named-channel evidence only at identical matter/radiation states."""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import math

BASE=Path(__file__).resolve().parent.parent
REQUIRED=frozenset(['hot_beta_moments/results.json','hot_beta_moments/hot_moments.py','positron_capture_moments/refinement.json','positron_capture_moments/moments.py','positron_capture_moments/MATCHED_STATE.json','review_hot_beta/REVIEW.md','review_hot_beta/CHECKPOINT.json','review_pc_moments/REVIEW.md','review_pc_moments/MANIFEST.json','relativistic_crossed_contract/beta.py','muon_full_current/current.py','relativistic_capture_oracle/events.py','relativistic_free_eos/free_baryons.py','thermodynamics/lepton_reference.py'])
SCOPE='Two named free electron-antineutrino charged-current channels only; no pair processes, neutrino-sector balance, transparent equilibrium or evolution.'

def radiation_key(s):
    if not isinstance(s,dict):raise ValueError('Explicit radiation scenario required')
    expected={'name','mode'} if s.get('mode')=='empty' else {'name','mode','mu_nu'}
    if set(s)!=expected:raise ValueError('Unsupported radiation fields; common material temperature only')
    if s.get('mode')=='empty':return ('empty',)
    if s.get('mode')=='FD' and not isinstance(s['mu_nu'],bool) and math.isfinite(s['mu_nu']):return ('common_material_T_FD',s['mu_nu'])
    raise ValueError('Only explicit empty or common-T FD radiation supported')

def compatible(a,b,sa,sb):
    if a!=b:raise ValueError('Material states/species/roots differ')
    if radiation_key(sa)!=radiation_key(sb):raise ValueError('Radiation occupations differ')
    if a.get('muons')!='absent':raise ValueError('Named npe-only species ansatz required')
    np_=a['proton']['density'];net=a['electron']['net_density']
    if not math.isclose(net,np_,rel_tol=1e-10):raise ValueError('Net pair charge is not neutral')

def value(s,d,k):
    r=s[d][k]
    if not r['representable'] or r['value'] is None or not math.isfinite(r['value']) or r['value']<0:
        raise ValueError('Positive gross moments must be represented, not floored')
    if r['value']==0:
        if r.get('log') is not None or r.get('reason') not in {'empty_support','absent_initial_antinu_occupation'}:raise ValueError('Finite positive log cannot be a zero floor')
    elif r.get('log') is None or not math.isfinite(r['log']):raise ValueError('Positive moment requires a finite log')
    return r['value']

def combine(a,b,state_a,state_b):
    compatible(state_a,state_b,a['scenario'],b['scenario'])
    out={'radiation':radiation_key(a['scenario']),'channels':{},'scope':SCOPE}
    for name,s in [('neutron_beta',a),('positron_capture',b)]:
        out['channels'][name]={d:{k:value(s,d,k) for k in ['number','neutrino_energy','material_energy_loss']} for d in ['forward','backward']}
        out['channels'][name]['net']={k:float(s['net'][k]) for k in ['number','neutrino_energy','material_energy_loss']}
        if any(not math.isfinite(v) for v in out['channels'][name]['net'].values()):raise ValueError('Nonfinite net moment')
    for direction in ['forward','backward','net']:
        out[direction]={k:math.fsum(c[direction][k] for c in out['channels'].values()) for k in ['number','neutrino_energy','material_energy_loss']}
    r=out['net']['number'];q=out['net']['neutrino_energy'];qm=out['net']['material_energy_loss']
    out['sources']={'dn_n_dt':-r,'dn_p_dt':r,'dn_e_net_dt':r,'dn_antinu_dt':r,'matter_Le_dt':r,'radiation_Le_dt':-r,'matter_energy_dt':-qm,'radiation_energy_dt':q}
    out['direct_energy_residual']=q-qm
    out['gross_normalized_net']=r/(out['forward']['number']+out['backward']['number'])
    if radiation_key(a['scenario'])[0]=='common_material_T_FD':
        mu_eq=state_a['neutron']['chemical_potential']-state_a['proton']['chemical_potential']-state_a['electron']['chemical_potential']
        affinity=(mu_eq-a['scenario']['mu_nu'])/state_a['T']
        out['affinity_over_T']=affinity
        out['reservoir_entropy_from_source']=affinity*r
    return out

def verify_manifest(manifest):
    if manifest.get('status')!='accepted_selected_sources':raise ValueError('Independent source reviews required')
    hashes=manifest.get('sha256')
    if not isinstance(hashes,dict) or not REQUIRED.issubset(hashes):raise ValueError('Required author/review/upstream hash coverage missing')
    for rel,digest in hashes.items():
        target=(BASE/rel).resolve()
        if not target.is_relative_to(BASE.resolve()) or not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid source path or SHA256')
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('Changed source/evidence '+rel)

def build():
    manifest=json.loads((Path(__file__).parent/'ACCEPTED_MANIFEST.json').read_text())
    verify_manifest(manifest)
    h=json.loads((BASE/'hot_beta_moments/results.json').read_text())
    p=json.loads((BASE/'positron_capture_moments/refinement.json').read_text())
    if h['all_pass'] is not True or p['passed'] is not True:raise ValueError('Qualified report must pass; coarse failed report is not a replacement')
    state_h=h['rows'][0]['matched_state'];state_p=json.loads((BASE/'positron_capture_moments/MATCHED_STATE.json').read_text())
    hr=[x['result'] for x in h['rows'] if x.get('result',{}).get('controls',{}).get('rest_order')==32]
    pr=[x['result'] for x in p['rows'] if x.get('kind')=='total' and x['result']['controls']['initial_order']==12]
    if len(hr)!=1 or len(pr)!=1:raise ValueError('Missing or ambiguous qualified rule')
    if hr[0]['units']!='number moments fm^-3 s^-1; energy moments MeV fm^-3 s^-1; entropy kB fm^-3 s^-1' or pr[0]['units']!='number fm^-3 s^-1; energy MeV fm^-3 s^-1; entropy kB fm^-3 s^-1':raise ValueError('Moment unit conventions differ')
    hs=hr[0]['scenarios'];ps=pr[0]['scenarios']
    results={n:combine(hs[n],ps[n],state_h,state_p) for n in ['empty','LTE']}
    regressions=[]
    for label,ah,bh,sa,sb in [('different_affinity',hs['emission_dominated']['scenario'],ps['noneq']['scenario'],state_h,state_p),('different_matter',hs['empty']['scenario'],ps['empty']['scenario'],dict(state_h,T=2.1),state_p)]:
        try:compatible(sa,sb,ah,bh)
        except ValueError as e:regressions.append({'name':label,'rejected':True,'reason':str(e)})
        else:raise AssertionError('Known incompatible inputs accepted')
    return {'status':'complete','scope':SCOPE,'matched_state':state_h,'qualified_controls':{'beta':hr[0]['controls'],'positron_capture':pr[0]['controls']},'results':results,'regressions':regressions,'source_manifest':manifest}

if __name__=='__main__':
    output=build();(Path(__file__).parent/'results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'status':output['status'],'empty_net':output['results']['empty']['net'],'LTE_roundoff':output['results']['LTE']['gross_normalized_net'],'rejected_incompatible':len(output['regressions'])}))
