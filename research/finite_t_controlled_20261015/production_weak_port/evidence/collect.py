"""Bounded owned evidence driver; never shipped/imported by the candidate."""
from pathlib import Path
import sys,json,math,time,dataclasses,importlib.util,hashlib
P=Path(__file__).resolve().parents[1];R=P.parent
sys.dont_write_bytecode=True;sys.path.insert(0,str(P/'test_mirror/src'))
from nmbackend.reference import weak_reference as W
LOCK=json.loads((P/'LOCK.json').read_text())
case=int(sys.argv[1]);channel=sys.argv[2];mode=sys.argv[3] if len(sys.argv)>3 else'candidate'
bathdata=LOCK['saved_bath']if case==0 else LOCK['heldouts'][case-1]
b=W.FreeElectronBath(**bathdata);mu=b.equilibrium_mu_nue;anti=-mu
scenarios=[('empty',W.Radiation.empty()),('LTE',W.Radiation.fd(b.T,mu,anti)),('emission',W.Radiation.fd(b.T,mu-.6*b.T,anti-.8*b.T)),('inverse',W.Radiation.fd(b.T,mu+.6*b.T,anti+.8*b.T))]
if case==0:
 # Actual saved fixture channel-specific common-T nonequilibrium radiation.
 scenarios=[('empty',W.Radiation.empty()),('LTE',W.Radiation.fd(b.T,mu,anti))]
 scenarios += [('noneq',W.Radiation.fd(b.T,mu-.6*b.T,anti-.8))]if channel!='NEUTRON_DECAY'else [('emission_dominated',W.Radiation.fd(b.T,mu,anti-1.6)),('inverse_dominated',W.Radiation.fd(b.T,mu,anti+1.6))]
def oldmodules():
 path=R/'transparent_capture_balance/balance.py';s=importlib.util.spec_from_file_location('weak_port_owned_evidence_old_pure_balance',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);m.verify();return m
start=time.monotonic();out={'case':case,'channel':channel,'mode':mode,'bath':bathdata,'profiles':{},'complete':False,'candidate_sources':{q.name:hashlib.sha256(q.read_bytes()).hexdigest()for q in(P/'test_mirror/src/nmbackend/reference').glob('*weak*.py')}};old=None
if mode=='reference':old=oldmodules()
for profile in ['base','fine']:
 controls=W.Controls.base(channel)if profile=='base'else W.Controls.fine(channel)
 rows={}
 for name,rad in scenarios:
  if mode=='reference' and channel=='PC' and name not in ('empty','LTE'):continue
  if time.monotonic()-start>110:raise TimeoutError('Owned evidence process bounded at110s; completed rows saved')
  if mode=='candidate':rows[name]=dataclasses.asdict(W.channel_moments(channel,b,rad,controls))
  else:
   s=[dict(name=name,mode=rad.mode)]
   if rad.mode=='FD':s[0].update(mu_nu=rad.mu_nue if channel=='EC'else rad.mu_antinue,temperature=b.T)
   if channel=='EC':
    data=old.EC.integrate(old.EC.Bath(**bathdata),proton_order=controls.radial_order,electron_order=controls.lepton_order,initial_order=controls.polar_order,outgoing_order=controls.polar_order,phi_order=controls.azimuth_order,tail=controls.tail,scenarios=s)
    rows[name]=data
   elif channel=='PC':
    # Independent accepted momentum-coordinate/CM-axis polynomial oracle;
    # evidence only, never a package runtime dependency. Its actual integrated
    # radiation scenarios cover empty and LTE for this cross-method check.
    data=old.PC.integral(bathdata,order_n=controls.radial_order,order_e=controls.lepton_order,order_angle=controls.polar_order,tail=controls.tail)
    rows[name]=dict(raw=data,selected_scenario_index=0 if name=='empty'else 1)
   else:rows[name]=old.BETA.integrate(old.BETA.K.Bath(**bathdata),rest_order=controls.rest_order,parent_order=controls.parent_order,orient_order=controls.orient_order,phi_order=controls.decay_azimuth_order,coordinate=controls.decay_coordinate,parent_coordinate=controls.parent_coordinate,tail=controls.tail,scenarios=s)
  out['profiles'].setdefault(profile,{})[name]=rows[name];out['seconds']=time.monotonic()-start
  target=P/'evidence'/f'{mode}_{case}_{channel}.json';target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
out['complete']=True;target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(case=case,channel=channel,mode=mode,seconds=time.monotonic()-start,complete=True)))
