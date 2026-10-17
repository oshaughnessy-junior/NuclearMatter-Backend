"""Declared time convergence, exact discrete ledgers, changing physical rates."""
from pathlib import Path
import importlib.util,sys,json,math
P=Path(__file__).parent;sp=importlib.util.spec_from_file_location('three_source_element_owned_functions',P/'element.py');E=importlib.util.module_from_spec(sp);sys.modules[sp.name]=E;sp.loader.exec_module(E)
def main():
 if(P/'results.json').exists():raise RuntimeError('Frozen trajectory evidence exists')
 pre=json.loads((P/'preflight.json').read_text());plan=json.loads((P/'EXPERIMENT_PLAN.json').read_text())
 if pre['failed']:raise RuntimeError('Forward thermo gate not passed')
 rows=[];checks=[]
 def ck(n,er,g):checks.append(dict(name=n,error=float(er),gate=g,passed=bool(math.isfinite(er)and abs(er)<=g)))
 for method,steps in plan['methods']:
  r=E.trajectory(method,steps,plan['duration_s']);rows.append(r);sc=r['initial'];f=r['final'];gross=r['escaped_nue']+r['escaped_antinue'];ck('energy ledger '+str((method,steps)),r['energy_ledger_relative_exchange'],2e-8);ck('Le ledger '+str((method,steps)),r['lepton_ledger']/gross,2e-8);ck('EMpair particle accounting '+str((method,steps)),r['pair_EMnumber_change']['equality_residual']/gross,2e-8);ck('positive escapingQ '+str((method,steps)),0 if r['escaped_energy']>0 and f['T']<sc['T']and f['Ye']>sc['Ye']else 1,0)
  for rec in r['records']:
   z=rec['source'];q=z['number_sources'];ck('event baryon ledger '+str((method,steps,rec['step'])),(q['n']+q['p'])/max(abs(z['R']),1e-30),2e-12);ck('event Le ledger '+str((method,steps,rec['step'])),(q['electron_net']+q['nue']-q['antinue'])/(q['nue']+q['antinue']),2e-12)
  (P/'partial_trajectories.json').write_text(json.dumps(dict(rows=rows,checks=checks),indent=2));print(method,steps,f['T'],f['Ye'],r['escaped_energy'],r['seconds'],flush=True)
 eu=[r for r in rows if r['method']=='Euler'];he=[r for r in rows if r['method']=='Heun']
 for key in['escaped_energy','Ye']:
  vals=[r[key]if key=='escaped_energy'else r['final'][key]for r in eu];ratio=(vals[0]-vals[1])/(vals[1]-vals[2]);ck('Euler difference ratio '+key,0 if 1.7<=ratio<=2.3 else 1,0);print('Euler ratio',key,ratio,flush=True)
 for key in['escaped_energy','Ye']:
  a=he[0][key]if key=='escaped_energy'else he[0]['final'][key]-.3;b=he[1][key]if key=='escaped_energy'else he[1]['final'][key]-.3;ck('Heun2to4 time error '+key,a/b-1,1e-3)
 z=rows[0]['records'][0]['source'];on=pre['initial'];off=pre['photon_off_initial'];diagnostic=dict(source=z,photon_ON=dict(cv=on['cv'],T_dot=(-z['Q']/E.NB-on['u_Y']*z['R']/E.NB)/on['u_T']),photon_OFF=dict(cv=off['cv'],T_dot=(-z['Q']/E.NB-off['u_Y']*z['R']/E.NB)/off['u_T']),same_source_for_both=True,thermostat=0.)
 out=dict(rows=rows,checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],initial_photon_diagnostic=diagnostic,source_pin_identity=E.M.Client().identity);(P/'results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(passed=out['passed'],failed=out['failed']),indent=2))
if __name__=='__main__':main()
