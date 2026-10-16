"""Independent reduced-energy-excess PBF integrals and literal source parity."""
from pathlib import Path
import importlib.util,sys,math,json,hashlib,time,subprocess
import numpy as np
from scipy.integrate import quad
from scipy.special import zeta
R=Path(__file__).resolve().parents[1];P=Path(__file__).resolve().parent;A=R/'pairing_pbf_contract';sys.dont_write_bytecode=True

def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);sys.modules[n]=m;s.loader.exec_module(m);return m
Q=load('review_pbf_q',A/'quasiparticle.py');C=load('review_pbf_contract',A/'contract.py')
MANIFEST=hashlib.sha256((A/'MANIFEST.json').read_bytes()).hexdigest()
assert MANIFEST=='6e54face569daaac2be62681011dd229f910681843dd25899cc48d22c77aea05'
pins=json.loads((A/'MANIFEST.json').read_text());assert all(hashlib.sha256((A/k).read_bytes()).hexdigest()==v for k,v in pins.items())
# Independent z-coordinate integral, with quasiparticle excess z-v=t².
# dx/dz=z/sqrt(z²-v²); its endpoint singularity cancels dt factor.
def logA(v):
 f=lambda t:2*(v+t*t)**5/math.sqrt(2*v+t*t)*math.exp(-2*t*t)/(1+math.exp(-v-t*t))**2
 val,err=quad(f,0,9,epsabs=1e-11,epsrel=2e-13,limit=160)
 return 2*math.log(v)-2*v+math.log(val)
def logB(v):
 # u=minimum-gap*cosh(theta), mu=sinh(theta)/sqrt3.
 baseline=logA(v)
 f=lambda theta:math.cosh(theta)/math.sqrt(3)*math.exp(logA(v*math.cosh(theta))-baseline)
 val,err=quad(f,0,math.asinh(math.sqrt(3)),epsabs=1e-12,epsrel=3e-12,limit=120)
 return baseline+math.log(val)
def run():
 start=time.monotonic();checks=[];rows=[];endpoint=[]
 def gate(n,e,t):checks.append(dict(name=n,error=float(e),tolerance=t,passed=bool(math.isfinite(e)and abs(e)<=t)))
 for v in [1e-250,1e-163,.017,.43,2.7,9.3,37.,127.,300.,499.]:
  la,lb=logA(v),logB(v);qa,qb=Q.log_F_A(v),Q.log_F_B(v)
  gate('FA energyexcess '+str(v),math.expm1(qa-la),3e-10);gate('FB energyexcess angularrapidity '+str(v),math.expm1(qb-lb),3e-10)
  rows.append(dict(v=v,independent_logFA=la,independent_logFB=lb,author_logFA=qa,author_logFB=qb))
  if v in [1e-250,1e-163]:
   coef=24*((1-2**-4)*zeta(5)-(1-2**-3)*zeta(4));gate('tinyFA analyticseries '+str(v),la-2*math.log(v)-math.log(coef),3e-12);gate('tinyFB analyticseries '+str(v),lb-2*math.log(v)-math.log(2*coef),3e-12)
 for v in [350.,500.]:
  fa=logA(v);fb=logB(v)
  aa=math.log(math.sqrt(math.pi)/2)+6.5*math.log(v)-2*v;ab=math.log(math.pi/(4*math.sqrt(3)))+6*math.log(v)-2*v
  gate('largeFA Laplace '+str(v),math.expm1(fa-aa),8/v);gate('largeFB twoLaplace '+str(v),math.expm1(fb-ab),10/v)
 for v in [.37,5.1,23.]:
  # Direct isotropic sphere normalized by4pi; normalizedrmsgap yields SAME physicalgap.
  f=lambda mu:math.exp(logA((v*math.sqrt(2))*math.sqrt((1+3*mu*mu)/2))-logA(v))
  rms=logA(v)+math.log(quad(f,-1,1,epsabs=1e-12,epsrel=1e-12)[0]/2)
  gate('RMS/min normalizedsphere '+str(v),math.expm1(rms-Q.log_F_B(v)),3e-10)
 for fun,upper in [(Q.log_F_A,1000),(Q.log_F_B,500)]:
  for x in [True,float('nan'),float('inf'),-.01,upper+1,1j]:
   try:fun(x);ok=False
   except ValueError:ok=True
   gate('phase guard '+fun.__name__+repr(x),0 if ok else 1,0)
  gate('gapzero exact logzero '+fun.__name__,0 if fun(0)==-math.inf else 1,0)
 for x in [1e-250,499.]:
  d=Q.represented(Q.log_F_B(x));gate('tiny/deep log unrepresented '+str(x),0 if d['value']is None and d['status']=='positive_below_float64_representation'and math.isfinite(d['log_value'])else 1,0)
 # Primary independent constants SI routes: GF GeV^-2 ->MeV^-2, kB_J/K /MeV_J,
 # hbar_Js /MeV_J, hc_MeVcm. No197/940 rounding introduced here.
 MeV_J=1.602176634e-13;kB=1.380649e-23/MeV_J;hbar=1.054571817e-34/MeV_J;hc=197.3269804e-13;GF=1.1663785e-5/1e6
 coefficient=4*GF**2*940.**2*(kB*1e9)**7/(15*math.pi**5)*(MeV_J*1e7)/(hc**3*hbar)
 gate('absoluteCGS SIconstants source rounding',coefficient/1.170e21-1,.001)
 leinson=(.5*1.26**2)/(1+2*1.26**2);gate('matchedgap Leinson ratio',leinson-.190122628855337,1e-6)
 # Fresh localFortran source extraction, byte-for-byte originalbody verification.
 original=(R/'cold_murca_parity/source_snapshot/neutrino_core.f').read_bytes();assert hashlib.sha256(original).hexdigest()=='45fb3891d0d2c69a527b0fa64be82eb69fd261f6ccccbc01b4d706ef4e7d9def'
 text=original.decode();snippet=text[text.index('      subroutine nu_1s0_pbf('):text.index('      function control_pbf_3P2_C(')];(P/'literal_pbf.f').write_text(snippet)
 gate('authorFortranliteralbody',0 if snippet==(A/'local_pbf_extraction.f').read_text()else 1,0)
 driver='''      program independent_pbf
      implicit real*8 (a-h,k-z)
      common/gapscale/sxp1s0,sxn3p2,ca3p2
      sxp1s0=1.d0
      sxn3p2=1.d0
      do
        read(*,*,iostat=ios) t,tc,mst,kf,ca3p2
        if (ios.ne.0) exit
        call nu_1s0_pbf(t,tc,mst,kf,qa)
        call nu_n3p2_B_pbf(t,tc,mst,kf,qb)
        write(*,'(2(es26.17e3,1x))') qa,qb
      end do
      end
''';(P/'driver.f').write_text(driver)
 cmd=['/opt/homebrew/bin/gfortran','-std=legacy','-ffixed-line-length-none','-O0','-fcheck=all',str(P/'driver.f'),str(P/'literal_pbf.f'),'-o',str(P/'review_pbf')];rr=subprocess.run(cmd,capture_output=True,text=True,timeout=20);assert rr.returncode==0,rr.stderr
 states=[(tau*tc,tc,m,k,ca)for tau,tc,m,k,ca in [(.231,8e8,.82,1.43,.7),(.672,2e9,.71,.93,.2),(.973,6e8,.91,2.1,2.),(1.07,3e9,.65,.7,.5),(1.,1e9,.78,1.9,.5)]]
 inp=''.join(' '.join(map(str,row))+'\n'for row in states);(P/'fortran_input.txt').write_text(inp);rr=subprocess.run([str(P/'review_pbf')],input=inp,text=True,capture_output=True,timeout=10);assert rr.returncode==0,rr.stderr;(P/'fortran_stdout.txt').write_text(rr.stdout)
 parity=[]
 for state,line in zip(states,rr.stdout.splitlines()):
  T,Tc,m,k,ca=state;f=list(map(float,line.split()));a=[float(C.LEGACY['pbf_1s0'](T,Tc,m,k)),float(C.LEGACY['pbf_n3p2'](T,Tc,m,k,ca))];parity.append(dict(state=state,Fortran=f,NMB=a))
  for i,label in enumerate(['1S0','3P2B']):
   if T!=Tc:gate('freshFortran '+label+str(state),a[i]/f[i]-1 if f[i]else a[i],3e-12)
   else:endpoint.append(dict(channel=label,T=T,Tc=Tc,Fortran=f[i],NMB=a[i],literal_zero_pass=bool(a[i]==f[i]==0),classification='preserved exactTc clamp failure,excluded'))
 for name in ['pbf_1s0','pbf_n3p2']:
  val=float(C.LEGACY[name](0.,1e9,.78,1.9));endpoint.append(dict(channel=name,T=0.,NMB=val if math.isfinite(val)else None,finite=math.isfinite(val),literal_zero_pass=bool(math.isfinite(val)and val==0),classification='preserved raw zeroT failure,excluded'))
 # Source identity / exactAST audits independently identifydefinitions andselectedconstants.
 import ast
 for source,names in [('superfluid.py',['below','u_1s0','u_3p2b','f32']),('core_nu.py',['_ctrl_1s0','_ctrl_3p2b','_pbf','pbf_1s0','pbf_n3p2','pbf_neutron'])]:
  b=(A/'source_snapshot'/source).read_bytes();gate('gitblob '+source,0 if hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()==C.PINS[source]else 1,0)
  for node in ast.parse(b).body:
   if isinstance(node,ast.FunctionDef)and node.name in names:gate('exactASTcodeobject '+node.name,0 if C.LEGACY[node.name].__code__.co_filename==str(A/'source_snapshot'/source)else 1,0)
 # Cross-zone dispatch actualgivenbody, independentlytestboundary.
 for zone in [14,15,16]:
  a,b=map(float,C.LEGACY['pbf_neutron'](4e8,1e9,.81,1.7,zone,15));gate('literalboundary '+str(zone),0 if (a>0)==(zone>15)and (b>0)==(zone<=15)else 1,0)
 assert all(hashlib.sha256((A/k).read_bytes()).hexdigest()==v for k,v in pins.items())
 out=dict(checks=checks,passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],new_integral_points=rows,cgs_coefficient=coefficient,Leinson_matchedgap_ratio=leinson,Fortran_parity=parity,preserved_endpoint_exclusions=endpoint,compile_command=cmd,seconds=time.monotonic()-start,author_manifest=MANIFEST,author_files_unchanged=True)
 (P/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print({k:out[k]for k in ['passed','failed','seconds']})
if __name__=='__main__':run()
