"""Independent malformed-request, acceptance and returned-packet adversaries."""
from common import *
import copy,io
checks=[];before=frozen();client=C.Client();forward=json.loads((P/'forward.json').read_text());packet=forward['rows'][0]['packet'];q=packet['request'];W=C.W

def reject(name,f):
 try:f();ok=False;message='accepted'
 except (ValueError,RuntimeError)as e:ok=True;message=str(e)
 except Exception as e:ok=False;message=type(e).__name__+':'+str(e)
 checks.append(dict(name=name,passed=ok,result=message))
for label,update in [('T nan',{'T':float('nan')}),('T bool',{'T':True}),('Ye bool',{'Ye':False}),('T low',{'T':1.79}),('Ye high',{'Ye':.401}),('nB changed',{'nB':1.1e-7}),('photon integer',{'photons':1}),('nonliteral empty action',{'action':'opacity'}),('unknown profile',{'profile':'fast'}),('wrong identity',{'identity':'0'*64})]:
 altered=dict(q);altered.update(update);reject(label,lambda altered=altered:W.request(altered,client.pin))
for label,mutation in [('accepted integer',lambda p:p.update(accepted_balance=1)),('accepted false',lambda p:p.update(accepted_balance=False)),('missing review',lambda p:p['files'].pop('review_transparent_capture_balance/REVIEW.md')),('missing loaded path',lambda p:p['expected_loaded_python'].pop()),('extra loaded path',lambda p:p['expected_loaded_python'].append('unexpected.py'))]:
 p=copy.deepcopy(client.pin);mutation(p);reject('pin record '+label,lambda p=p:W.validate_pin_record(p))
for label,mutation in [('photon missing energy',lambda p:p['state']['photon'].update(energy_density=0.)),('source energy derivative sign',lambda p:p['sources'].update(u_dot=-p['sources']['u_dot'])),('source Ye derivative sign',lambda p:p['sources'].update(Ye_dot=-p['sources']['Ye_dot'])),('temperature derivative wrong',lambda p:p['sources'].update(T_dot=p['sources']['T_dot']+.1)),('nu/antinu number wrong',lambda p:p['sources']['number_sources'].update(nue=0.)),('pair particle source wrong',lambda p:p['sources']['number_sources'].update(positron=0.)),('state photon Cv double count',lambda p:p['state'].update(cv=p['state']['cv']+p['state']['photon']['cv'])),('loaded source absent',lambda p:p['loaded_sources'].pop(next(iter(p['loaded_sources'])))),('source controls altered',lambda p:p['sources']['controls']['EC'].__setitem__(0,999)),('separate moment zero',lambda p:p['sources']['channels']['EC'].update(number=0.))]:
 p=copy.deepcopy(packet);mutation(p);reject('returned packet '+label,lambda p=p:client.validate(p,q))
# Real worker entry point rejects bad request before physical imports or calculations.
oldin=sys.stdin;oldphysical=W.physical;calls=[]
def spy():calls.append('physical');raise AssertionError('physical executed before admission')
W.physical=spy
try:
 altered=dict(q);altered['photons']=1;sys.stdin=io.StringIO(json.dumps(altered));reject('worker bad request before physical()',W.main)
finally:sys.stdin=oldin;W.physical=oldphysical
checks.append(dict(name='no physical preparation ran in entry-point adversary',passed=not calls,result=calls))
# Valid frozen packet and exact required dependency list are positive controls.
client.validate(packet,q);checks.append(dict(name='fresh genuine source packet admits',passed=True,result='exact request/control/loaded sets, source signs, photon energy and counts'))
checks.append(dict(name='all frozen files unchanged',passed=before==frozen(),result='byte comparison'))
(P/'admission.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']]),indent=2)+'\n');print(len(checks),[c for c in checks if not c['passed']])
