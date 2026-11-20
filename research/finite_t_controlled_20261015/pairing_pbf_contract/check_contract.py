"""Forward/phase controls plus preserved literal legacy endpoint exclusions."""
from pathlib import Path
import json,math,time,hashlib
import numpy as np
from scipy.special import zeta
from contract import LEGACY,legacy,gap,current_coefficients
from quasiparticle import log_F_A,log_F_B,represented
P=Path(__file__).resolve().parent

def main():
    start=time.monotonic();checks=[];rows=[];rates=[];excluded=[]
    def gate(name,err,tol):checks.append(dict(name=name,error=float(err),tolerance=tol,passed=bool(math.isfinite(err) and abs(err)<=tol)))
    exact_small=24*((1-2**-4)*zeta(5)-(1-2**-3)*zeta(4))
    for v in [.001,.03,.2,.7,1.5,3.,6.,12.,25.,60.,100.]:
        la=log_F_A(v);lb=log_F_B(v);a=float(LEGACY['_ctrl_1s0'](v));b=float(LEGACY['_ctrl_3p2b'](v));ra=math.expm1(math.log(a)-la);rb=math.expm1(math.log(b)-lb)
        gate('matched scalar FA fit '+str(v),ra,.05);gate('matched angular FB fit '+str(v),rb,.05)
        rows.append(dict(v=v,A=represented(la),B=represented(lb),legacy_fit_A=a,legacy_fit_B=b,fit_relative_A=ra,fit_relative_B=rb))
    for v in [1e-4,1e-5]:
        gate('nearTc FA exact coefficient '+str(v),math.exp(log_F_A(v))/v**2/exact_small-1,1e-6)
        gate('nearTc FB angularmean2 '+str(v),math.exp(log_F_B(v))/v**2/(2*exact_small)-1,1e-6)
    for v in [1e-100,1e-200]:
        gate('tiny positive gap angular log normalization '+str(v),log_F_B(v)-(2*math.log(v)+math.log(2*exact_small)),2e-10)
    asym=[]
    for v in [50.,100.,200.]:
        la=log_F_A(v);lb=log_F_B(v)
        aa=math.log(math.sqrt(math.pi)/2)+6.5*math.log(v)-2*v
        ab=math.log(math.pi/(4*math.sqrt(3)))+6*math.log(v)-2*v
        ea=math.expm1(la-aa);eb=math.expm1(lb-ab)
        gate('lowT FA asymptote '+str(v),ea,8/v);gate('lowT FB asymptote '+str(v),eb,10/v)
        asym.append(dict(v=v,A_relative_to_leading_asymptote=ea,B_relative_to_leading_asymptote=eb))
    for tau in [.03,.1,.3,.7,.95,1-1e-6]:
        T=1e9*tau
        for label,mst,kf in [('neutron',.78,1.9),('proton',.69,.7)]:
            q1=legacy('1S0',T,1e9,mst,kf);q3=legacy('3P2B',T,1e9,mst,kf) if label=='neutron' else None
            coeff=current_coefficients(mst,kf)
            vA=gap(tau,'YK1999_A');vB=gap(tau,'NSCool_NMB_B_legacy');base=1.170e21*mst**2*coeff['vF_over_c_legacy']*tau**7*3
            gate('forward singlet actual AST '+str((tau,label)),q1/(base*coeff['singlet_legacy_axial_only']*float(LEGACY['_ctrl_1s0'](vA)))-1,3e-13)
            if q3 is not None:gate('forward triplet actual AST '+str((tau,label)),q3/(base*.5*1.26**2*float(LEGACY['_ctrl_3p2b'](vB)))-1,3e-13)
            gate('positive admitted scalar '+str((tau,label)),0 if q1>0 and (q3 is None or q3>0) else 1,0)
            rates.append(dict(tau=tau,label=label,mst=mst,kf=kf,Q_1S0=q1,Q_3P2B=q3))
        qp=legacy('3P2B',T,1e9,.78,1.9,ca=2.)
        gate('matched gap current coefficient quarter '+str(tau),legacy('3P2B',T,1e9,.78,1.9)/qp-.25,2e-15)
        for i in [6,7,8]:
            qn1,qn3=map(float,LEGACY['pbf_neutron'](T,1e9,.78,1.9,i,7))
            gate('zone PBF onechannel '+str((tau,i)),0 if (qn1>0)==(i>7) and (qn3>0)==(i<=7) else 1,0)
    for channel in ['1S0','3P2B']:
        gate('aboveTc absent '+channel,legacy(channel,1.01e9,1e9,.78,1.9),0)
        gate('zeroTc absent '+channel,legacy(channel,1e8,0.,.78,1.9),0)
        gate('zero mass species absent '+channel,legacy(channel,1e8,1e9,0.,0.),0)
        fun=LEGACY['pbf_1s0'] if channel=='1S0' else LEGACY['pbf_n3p2']
        for T in [0.,1e9]:
            q=float(fun(T,1e9,.78,1.9));excluded.append(dict(channel=channel,T=T,Tc=1e9,value=q if math.isfinite(q) else None,finite=math.isfinite(q),literal_physical_exact_zero_gate=math.isfinite(q) and q==0.,classification='RAW legacy endpoint, NOT admitted zero-temperature reference; exactTc tau-clamp artifact separately named'))
    current=current_coefficients(.78,1.9)
    factor=current['triplet_legacy_default_axial']/current['triplet_original_bare_vector_plus_axial']
    gate('matched convention Leinson coefficient ratio',factor-(.5*1.26**2)/(1+2*1.26**2),1e-15)
    gaps=[]
    for tau in [.05,.2,.5,.9]:
        old=gap(tau,'NSCool_NMB_B_legacy');new=gap(tau,'YK1999_B_minimum')
        # Separate variant comparison only, NOT an error gate against legacy.
        gaps.append(dict(tau=tau,legacy_B_gap_T=old,YK1999_B_min_gap_T=new,legacy_to_YK_angular_energy_factor=math.exp(log_F_B(old)-log_F_B(new)),rms_normalization_relation='Leinson Delta_rms=sqrt(2)*Delta_min for same Delta(theta)'))
    guards=[]
    for args in [('1S0',0.,1e9,.78,1.9),('3P2B',1e5,1e9,.78,1.9),('1S0',True,1e9,.78,1.9),('1S0',float('nan'),1e9,.78,1.9)]:
        try:legacy(*args);ok=False
        except ValueError:ok=True
        guards.append(dict(arguments=[str(x) for x in args],rejected=ok));gate('wrapper rejection '+str(args),0 if ok else 1,0)
    logs=dict(A_v1000=represented(log_F_A(1000.)),B_v500=represented(log_F_B(500.)))
    for k,r in logs.items():gate('finite unrepresented positive log '+k,0 if r['status']=='positive_below_float64_representation' and r['value'] is None and math.isfinite(r['log_value']) else 1,0)
    out=dict(checks=checks,passed=sum(x['passed'] for x in checks),failed=[x for x in checks if not x['passed']],dimensionless=rows,lowT_asymptotic=asym,exact_small_v_A=exact_small,rates=rates,raw_endpoint_exclusions=excluded,current_coefficient=current,matched_gap_current_ratio=factor,gap_variants=gaps,wrapper_guards=guards,unrepresented=logs,seconds=time.monotonic()-start,scope='selected exactAST scalar legacy and independent reduced quasiparticle phase integral only; no full pairing EOS/current response')
    (P/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:out[k] for k in ['passed','failed','seconds']}))
    return len(out['failed'])

if __name__=='__main__':raise SystemExit(main())
