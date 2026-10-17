from pathlib import Path
import importlib.util,sys,json,math,hashlib
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parent;sys.dont_write_bytecode=True

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
C=load('heavy_point_frozen_Dirac_control',ROOT/'muon_full_current/current.py');K=load('heavy_point_owned_scalar_control',P/'point_current.py');lock=json.loads((P/'LOCK.json').read_text());checks=[];rows=[];Q0=C.V.MN-C.V.MP;M0=(C.V.MN+C.V.MP)/2

def ck(n,e,t=3e-10):checks.append(dict(name=n,error=float(e),gate=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
for scale in lock['scales']:
 M=scale*M0;mn=M+Q0/2;mp=M-Q0/2;ck('physicalQfixed '+str(scale),mn-mp-Q0,2e-12)
 for anti in[False,True]:
  ch=C.V.Channel(mp if anti else mn,mn if anti else mp,.51099895069,antineutrino=anti)
  for en,cos in[(4.7,-.61),(17.3,.22)]:
   p=C.V.momenta(ch,en,cos);t=C.V.dot(p['pb']-p['pa'],p['pb']-p['pa']);ff=C.FormFactors(weak_magnetism=False,pseudoscalar=False,q2_dependence=False).values(t,M);ck('literalpointFF '+str((scale,anti,en)),max(abs(a-b)for a,b in zip(ff,[1.,1.2723,0.,0.])),0);amp=float(K.amplitude(*[p[k][None,:]for k in['k','pa','l','pb']],ch.initial_mass,ch.final_mass,anti)[0]);trace=C.trace(ch,p,ff)/2;ck('scaledliteralDiractrace '+str((scale,anti,en)),amp/trace-1);ck('positiveon-shellamp '+str((scale,anti,en)),0 if amp>0 else 1,0)
   residual=p['k']+p['pa']-p['l']-p['pb'];ck('exactkinematicfourmomentum '+str((scale,anti,en)),max(abs(residual))/M)
   for name,mass in [('pa',ch.initial_mass),('pb',ch.final_mass)]:ck('deformedonshellmass '+str((scale,anti,en,name)),C.V.dot(p[name],p[name])/(mass*mass)-1)
   rows.append(dict(scale=scale,anti=anti,Enu=en,cos=cos,ma=ch.initial_mass,mb=ch.final_mass,scalar=amp,trace=trace))
out=dict(checks=checks,passed=sum(x['passed']for x in checks),failed=[x for x in checks if not x['passed']],rows=rows);(P/'current_controls.json').write_text(json.dumps(out,indent=2)+'\n');print(out['passed'],out['failed'])
