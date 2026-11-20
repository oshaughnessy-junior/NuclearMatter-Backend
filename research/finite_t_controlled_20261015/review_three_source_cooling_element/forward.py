from common import *
checks=[];rows=[];start=time.monotonic()
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
before=frozen();(P/'before.json').write_text(json.dumps(before,indent=2)+'\n')
for f,h in {'MANIFEST.json':'9c6abbbe5734c1905748809e7d681b75499c8db81afb5c301798ab1ff0dc0c77','element.py':'a2711240850f639d0259e9eb2e74b3931d4a4f3ec1726c7bb4934fdcf402a27b','worker.py':'0fe9f30a9dcf985c92a58f9223cae74f0174816aa592ad57828d1dc1db1c240b','client.py':'15c93ab17f10a895324fac06210847cf9972c929ba2b7de34c52d68d158f07e7'}.items():ck('commissioned SHA '+f,int(hashlib.sha256((A/f).read_bytes()).hexdigest()!=h),0)
for rel,h in json.loads((A/'MANIFEST.json').read_text())['files'].items():ck('manifest '+rel,int(hashlib.sha256((A/rel).read_bytes()).hexdigest()!=h),0)
coefficient=quad(lambda x:x**3*math.exp(-x)/(-math.expm1(-x)),0,100,epsabs=1e-13,epsrel=2e-13)[0]/math.pi**2
ck('independent Planck two-polarization prefactor',coefficient/(math.pi**2/15)-1,3e-13)
client=C.Client();author=json.loads((A/'final_replay.json').read_text())['trajectory'];final=author['final']
for T,Y in [(2.,.3),(final['T'],final['Ye'])]:
 packet=client.evaluate(T,Y,action='source',profile='fine');s=packet['state'];mine=eos(T,Y);z=packet['sources'];rows.append(dict(packet=packet,independent=mine))
 ck('independent EOS u '+str(T),mine['u']-s['u'],2e-9)
 for k in ['mu_n','mu_p','mu_e']:ck('independent EOS '+k+' '+str(T),mine['bath'][k]-s['bath'][k],3e-10)
 ck('independent actual pair charge '+str(T),(mine['ne']-mine['nplus'])/(NB*Y)-1,3e-10)
 for key,expected in [('electron_particle_density',mine['ne']),('positron_density',mine['nplus'])]:ck('independent pair population '+key+' '+str(T),s[key]/expected-1,3e-10)
 dT=1e-4;dy=1e-5;uT=(eos(T+dT,Y)['u']-eos(T-dT,Y)['u'])/(2*dT);uY=(eos(T,Y+dy)['u']-eos(T,Y-dy)['u'])/(2*dy)
 ck('FD independent photon+pair u_T '+str(T),uT/s['u_T']-1,3e-7)
 ck('FD independent restinclusive u_Y '+str(T),uY-s['u_Y'],3e-7)
 ck('conserved chain temperature derivative '+str(T),(z['u_dot']-uY*z['Ye_dot'])/uT/z['T_dot']-1,3e-7)
 n,p,e=[s['material'][k]for k in ['neutron','proton','electron']];eg=coefficient*T**4/HC**3;chemical=n['chemical_potential']*NB*(1-Y)+p['chemical_potential']*NB*Y+e['chemical_potential']*NB*Y
 ck('restinclusive Euler thermodynamics '+str(T),(s['energy_density']+s['pressure']-T*s['entropy_density']-chemical)/s['energy_density'],3e-11)
 ck('photon energy '+str(T),eg/s['photon']['energy_density']-1,3e-13)
 ck('photon cv '+str(T),(4*coefficient*T**3/HC**3)/s['photon']['cv']-1,3e-13)
 rates=z['rates'];qs=z['energy'];gross=sum(rates.values());R=rates['beta']+rates['PC']-rates['EC'];Q=sum(qs.values())
 ck('source restinclusive total no extra Q '+str(T),(z['material_energy_from_species']+Q)/Q,3e-10)
 ck('actual-source net Le '+str(T),(z['number_sources']['electron_net']+z['number_sources']['nue']-z['number_sources']['antinue'])/gross,1e-14)
 ck('escaping source positive '+str(T),int(Q<=0),0)
 # Forward values must agree before any subsequent time-integration claim.
 if any(not c['passed']for c in checks):raise ArithmeticError('Independent forward gate failed; stop')
# New endpoint is tested with base vs fine; retained original tail evidence not relabelled.
base=client.evaluate(final['T'],final['Ye'],action='source',profile='base');fine=rows[-1]['packet']
for c in fine['sources']['channels']:
 for k,v in fine['sources']['channels'][c].items():ck('changed endpoint actual phase refinement '+c+' '+k,base['sources']['channels'][c][k]/v-1,3e-6)
ck('frozen unchanged',int(before!=frozen()),0)
(P/'forward.json').write_text(json.dumps(dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],rows=rows,base_endpoint=base,seconds=time.monotonic()-start),indent=2)+'\n');print(len(checks),sum(not c['passed']for c in checks),time.monotonic()-start)
