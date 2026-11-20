"""Tiny local NSCool-PBF source extraction/execution, not whole cooling replay."""
from pathlib import Path
import subprocess,json,hashlib
from contract import LEGACY
P=Path(__file__).resolve().parent;source=P.parent/'cold_murca_parity/source_snapshot/neutrino_core.f';b=source.read_bytes()
assert hashlib.sha256(b).hexdigest()=='45fb3891d0d2c69a527b0fa64be82eb69fd261f6ccccbc01b4d706ef4e7d9def'
text=b.decode();start=text.index('      subroutine nu_1s0_pbf(');end=text.index('      function control_pbf_3P2_C(')
(P/'local_pbf_extraction.f').write_text(text[start:end])
driver='''      program pbf_review
      implicit real*8 (a-h,k-z)
      integer ident
      common/gapscale/sxp1s0,sxn3p2,ca3p2
      sxp1s0=1.d0
      sxn3p2=1.d0
      ca3p2=.5d0
      do
       read(*,*,iostat=ios) ident,t,tc,mst,kf
       if (ios.ne.0) exit
       call nu_1s0_pbf(t,tc,mst,kf,qa)
       call nu_n3p2_B_pbf(t,tc,mst,kf,qb)
       write(*,'(i4,2(1x,es26.17e3))') ident,qa,qb
      end do
      end
'''
(P/'local_pbf_driver.f').write_text(driver)
cmd=['/opt/homebrew/bin/gfortran','-std=legacy','-ffixed-line-length-none','-O1','-fcheck=all','-ffpe-trap=invalid,zero,overflow',str(P/'local_pbf_driver.f'),str(P/'local_pbf_extraction.f'),'-o',str(P/'local_pbf')]
r=subprocess.run(cmd,capture_output=True,text=True,timeout=25);assert r.returncode==0,r.stderr
states=[(tau*1e9,1e9,.77,1.87) for tau in [.19,.53,.88,1.,1.01]]
inp=''.join(' '.join(map(str,[i]+list(s)))+'\n' for i,s in enumerate(states))
r=subprocess.run([str(P/'local_pbf')],input=inp,capture_output=True,text=True,timeout=10);assert r.returncode==0,r.stderr
for name,value in [('input',inp),('stdout',r.stdout),('stderr',r.stderr)]: (P/('fortran_'+name+'.txt')).write_text(value)
rows=[];checks=[]
for s,line in zip(states,r.stdout.splitlines()):
 q=list(map(float,line.split()[1:]));a=float(LEGACY['pbf_1s0'](*s));bb=float(LEGACY['pbf_n3p2'](*s));rows.append(dict(state=s,Fortran_1S0=q[0],Fortran_3P2B=q[1],NMB_1S0=a,NMB_3P2B=bb))
 if s[0]!=s[1]:
  for label,x,y in [('1S0',a,q[0]),('3P2B',bb,q[1])]:
   error=x/y-1 if y else x;checks.append(dict(name='actual local Fortran/NMB '+str((s[0],label)),error=error,tolerance=3e-12,passed=abs(error)<=3e-12))
out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,compile_command=cmd,compile_stderr='',source_sha256=hashlib.sha256(b).hexdigest(),extraction_is_literal_substring=True,scope='Exact local installation with nmbackend ca patch; NOT fresh official release, full thermal response or cooling-profile qualification. ExactTc disparity is retained separately.')
(P/'fortran_results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed']}))
