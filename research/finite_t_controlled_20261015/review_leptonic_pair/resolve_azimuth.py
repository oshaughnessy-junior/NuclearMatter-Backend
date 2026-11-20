from pathlib import Path
import sys,json,math
P=Path(__file__).resolve().parent;R=P.parent;sys.dont_write_bytecode=True;sys.path.insert(0,str(R/'leptonic_pair_annihilation'));import pair as A
r=A.thermal(2,.17446333200826397,A.Couplings('e'),radial=20,incident=10,outgoing=10,azimuth=24,tail=60.,radiation=A.Radiation(2,.5,1.1));old=next(x['result']for x in json.loads((P/'fd_results.json').read_text())['rows']if x['kind']=='author'and x['name']=='negative');ind=json.loads((P/'angle_refinement.json').read_text())['result']['scenarios']['negative'];checks=[];names=['number','nu_energy','antinu_energy','electron_energy','positron_energy']
def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for branch,ib in [('forward','F'),('backward','B')]:
 for i,key in enumerate(names):gate('isolated authorphi12to24 '+branch+key,old[branch][key]['value']/r[branch][key]['value']-1,2e-6);gate('refined phi independent absolute '+branch+key,r[branch][key]['value']/ind[ib][i]-1,2e-6)
(P/'azimuth_results.json').write_text(json.dumps(dict(checks=checks,result=r,passed=all(c['passed']for c in checks),diagnosis='Authorcontrol ladderskeptphi12fixed; newisolatedphi24 distinguishes that quadrature error from different CM-axis implementation.'),indent=2)+'\n');print(len(checks),all(c['passed']for c in checks),r['runtime']);print('isolated',max(abs(c['error'])for c in checks if'isolated'in c['name']),'absolute',max(abs(c['error'])for c in checks if'absolute'in c['name']))
