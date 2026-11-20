"""Independent frozen-source cold-MU review; never imports author drivers."""
from pathlib import Path
import ast,hashlib,json,math,subprocess,time,resource
import numpy as np
from scipy.optimize import brentq
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
HERE=Path(__file__).resolve().parent; AUTHOR=HERE.parent/'cold_murca_parity'; SRC=AUTHOR/'source_snapshot'
HC=197.3269804; ME=.51099895069; MM=105.6583755
resource.setrlimit(resource.RLIMIT_CORE,(0,0))

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pins():return {str(p.relative_to(AUTHOR)):sha(p) for p in AUTHOR.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
def gitblob(p):
    b=p.read_bytes();return hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
def nmb():
    ns={'jnp':jnp,'np':np}
    for file,names in [('superfluid.py',{'fexp','below','u_1s0','u_3p2b'}),('core_nu.py',{'_sq','_r_mu_a','_r_mu_b','_r_mu_p_3p2','murca'})]:
        nodes=[x for x in ast.parse((SRC/file).read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
        assert {x.name for x in nodes}==names
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SRC/file),'exec'),ns)
    return ns['murca']

def state(nb,yp):
    kp=(3*math.pi**2*nb*yp)**(1/3);kn=(3*math.pi**2*nb*(1-yp))**(1/3)
    # Charge equation in electron momentum, independent of author's mu root.
    def km(ke):return math.sqrt(max(ke*ke+(ME*ME-MM*MM)/HC**2,0.))
    ke=brentq(lambda ke:ke**3+km(ke)**3-kp**3,0.,kp,xtol=1e-14)
    return dict(T=6.37e8,kfe=ke,kfm=km(ke),kfn=kn,kfp=kp,mstn=.731,mstp=.863,tcn=0.,tcp=0.,i=7,isf=7,nB=nb,Yp=yp)

KEYS=['T','kfe','kfm','kfn','kfp','mstn','mstp','tcn','tcp','i','isf']
def evaln(f,s):return float(f(*(s[k] for k in KEYS)))
def execute(exe,states,name):
    fields=['i','T','kfe','kfm','kfn','kfp','mstn','mstp','tcn','tcp','isf']
    inp=''.join(' '.join(map(str,[j+1]+[s[k] for k in fields]))+'\n' for j,s in enumerate(states))
    r=subprocess.run([str(exe)],input=inp,text=True,capture_output=True,timeout=20,cwd=HERE)
    for suffix,value in [('input',inp),('stdout',r.stdout),('stderr',r.stderr)]: (HERE/(name+'_'+suffix+'.txt')).write_text(value)
    vals=[float(line.split()[1]) for line in r.stdout.splitlines() if len(line.split())==2]
    return r,vals

def normal(s):
    a=(1.76-.63*(1.68/s['kfn'])**2)*.68*(s['T']/1e9)**8*8.55e21/1.68
    return a*s['mstn']**3*s['mstp']*(s['kfe']+s['kfm']),a*s['mstn']*s['mstp']**3*(s['kfe']+3*s['kfp']-s['kfn'])**2/(8*s['kfp'])

def main():
    start=time.monotonic(); before=pins();checks=[]
    def gate(name,value,tol=3e-12):checks.append(dict(name=name,value=float(value),tolerance=tol,passed=bool(math.isfinite(value) and abs(value)<=tol)))
    assert before['run_parity.py']=='373a8afdf94ae82170087564379f0f0fc1cdeac4b36ce712a4b769862d53e15e'
    assert gitblob(SRC/'core_nu.py')=='520d3b458c5eddbed1e1e1a5c6270dcc12643eb5'
    assert gitblob(SRC/'superfluid.py')=='e16a07cfa7b92530649822358c552b1fa7069042'
    assert gitblob(SRC/'mirror_neutrino_core.f')=='7a58562ed3c6a74dd469810838873b91878c3b57'
    # Independent build, exact author's tiny interface driver and frozen source.
    commands=[]
    for label,driver,source in [('local','driver.f','numurca_nucl.f'),('mirror','mirror_driver.f','mirror_numurca_nucl.f')]:
        cmd=['/opt/homebrew/bin/gfortran','-std=legacy','-ffixed-line-length-none','-O1','-fcheck=all','-ffpe-trap=invalid,zero,overflow','-I',str(SRC),str(AUTHOR/driver),str(SRC/source),'-o',str(HERE/('review_'+label))]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=25,cwd=HERE);commands.append(dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
    fn=nmb();states=[]
    for nb,yp in [(.1937,.0673),(.413,.0519)]:
        s=state(nb,yp);states.append(dict(s,label='fresh normal'))
        gate('charge '+str(nb),(s['kfe']**3+s['kfm']**3)/s['kfp']**3-1)
        gate('mu_e=mu_mu '+str(nb),math.hypot(s['kfe']*HC,ME)/math.hypot(s['kfm']*HC,MM)-1)
        gate('neutron density '+str(nb),s['kfn']**3/(3*math.pi**2)/(nb*(1-yp))-1)
    base=states[0]
    for tau in [.23,.57,.91]:
        for i in [6,7,8]:
            states.append(dict(base,T=8.7e8*tau,tcn=8.7e8,tcp=1.13e9,i=i,label='fresh simultaneous pairing'))
    for i in [6,7,8]:states.append(dict(base,T=8.7e8,tcn=8.7e8,tcp=0.,i=i,label='exactTc'))
    for T in [0.,1.17*base['T']]:states.append(dict(base,T=T,label='T scaling'))
    rl,ql=execute(HERE/'review_local',states,'fresh_local');rm,qm=execute(HERE/'review_mirror',states,'fresh_mirror')
    assert rl.returncode==rm.returncode==0 and len(ql)==len(qm)==len(states)
    rows=[]
    for j,(s,l,m) in enumerate(zip(states,ql,qm)):
        q=evaln(fn,s);scale=abs(l) if l else 1.
        gate('fresh NMB/local '+str(j),(q-l)/scale)
        gate('fresh local/mirror O1 '+str(j),(l-m)/scale)
        if s['label'] in ['fresh normal','exactTc','T scaling']:
            a,b=normal(s);gate('normal/exactTc analytic '+str(j),(q-a-b)/max(abs(q),1.))
        rows.append(dict(state=s,fortran_local=l,fortran_mirror=m,nmb=q))
    # Two independent calls identify branches, without author's formula helper.
    s=dict(base,T=1e9,kfn=2.01,kfp=.5,kfe=.5,kfm=0.,tcn=0.,tcp=0.)
    factor=1.37;t=dict(s,mstn=s['mstn']*factor)
    rr,qq=execute(HERE/'review_local',[s,t],'forbidden_polygon');assert rr.returncode==0
    inferred_p=(factor**3*qq[0]-qq[1])/(factor**3-factor)
    gate('forbidden total NMB/Fortran',evaln(fn,s)/qq[0]-1)
    gate('branch inference vs polynomial',(inferred_p-normal(s)[1])/qq[0])
    low=dict(s,kfn=.8,kfp=.4,kfe=.4)
    rr,qn=execute(HERE/'review_local',[low],'low_kfn');assert rr.returncode==0
    gate('negative total NMB/Fortran',evaln(fn,low)/qn[0]-1)
    physical=[dict(name='proton branch must vanish outside cold polygon',value=inferred_p,scaled_error=abs(inferred_p)/qq[0],tolerance=3e-12,passed=abs(inferred_p)/qq[0]<=3e-12),dict(name='physical nonnegative emissivity',value=qn[0],passed=qn[0]>=0)]
    excluded=[]
    for name,ss in [('zero_e_with_mu',dict(base,kfe=0.)),('zero_leptons',dict(base,kfe=0.,kfm=0.))]:
        r,values=execute(HERE/'review_local',[ss],name)
        n=evaln(fn,ss);gate(name+' Fortran trapped',0 if r.returncode!=0 and not values else 1,0);gate(name+' NMB finite',0 if math.isfinite(n) else 1,0)
        excluded.append(dict(name=name,state=ss,returncode=r.returncode,NMB=n,fortran_values=values,scope='electron-present port gate EXCLUDED; no invented finite original comparator'))
    # Independent source extraction/body audit, not inherited mirror claim.
    for short,full in [('numurca_nucl.f','neutrino_core.f'),('mirror_numurca_nucl.f','mirror_neutrino_core.f')]:
        gate(short+' exact bytes in full source',0 if (SRC/short).read_bytes().strip() in (SRC/full).read_bytes() else 1,0)
    def body(path):
        lines=path.read_text().splitlines(); selected=[];on=False
        for line in lines:
            if not line or line[0] in 'cC*!':continue
            text=line[6:].split('!')[0].lower().replace(' ','').replace('\t','')
            if text.startswith('fexp(x)='):on=True
            if on:selected.append(text)
        return ''.join(selected)
    gate('literal computational body local/mirror',0 if body(SRC/'numurca_nucl.f')==body(SRC/'mirror_numurca_nucl.f') else 1,0)
    for tau in [.23,.57,.91]:
        selected=[x['nmb'] for x in rows if x['state']['label']=='fresh simultaneous pairing' and x['state']['T']==8.7e8*tau]
        gate('singlet branch i=isf and i>isf '+str(tau),selected[1]/selected[2]-1)
        gate('triplet vs singlet distinct '+str(tau),0 if abs(selected[0]/selected[1]-1)>1e-5 else 1,0)
    # Exact author physical-failure inputs, in addition to fresh mass ratios.
    original_failure_states=[dict(s,mstn=.78,mstp=.69),dict(low,mstn=.78,mstp=.69)]
    ro,original_q=execute(HERE/'review_local',original_failure_states,'exact_original_failures');assert ro.returncode==0
    for j,(ss,q) in enumerate(zip(original_failure_states,original_q)):
        gate('original physical failure port replay '+str(j),evaln(fn,ss)/q-1)
    # Pin every byte including original failed reports: no author-driver import.
    after=pins();gate('all frozen author bytes unchanged',0 if before==after else 1,0)
    out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],rows=rows,physical_extrapolation_gates=physical,physical_failed=sum(not x['passed'] for x in physical),forbidden_state=s,forbidden_total=qq[0],inferred_proton=inferred_p,negative_state=low,negative_Q=qn[0],exact_original_failure_totals=original_q,electron_absent_exclusions=excluded,commands=commands,source_sha256_before=before,source_sha256_after=after,seconds=time.monotonic()-start)
    (HERE/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['passed','failed','physical_failed','seconds']}))

if __name__=='__main__':main()
