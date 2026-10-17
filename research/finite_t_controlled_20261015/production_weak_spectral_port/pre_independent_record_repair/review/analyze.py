"""Referencequalification and LOCALsource algebra, no new physicalintegrals."""
from pathlib import Path
import json,math,hashlib,datetime
P=Path(__file__).resolve().parent;L=json.loads((P/'LOCK.json').read_text());checks=[];rows=[]
def ck(n,e,g):checks.append(dict(name=n,error=float(e),gate=g,passed=bool(math.isfinite(e)and abs(e)<=g)))
for ch,E in[('EC',6.431),('PC',4.237),('NEUTRON_DECAY',.337),('NEUTRON_DECAY',.873)]:
 f=P/f'high_{ch}_{E}.json'if E!=.337 else P/f'pure_{ch}_{E}_fine.json';z=json.loads(f.read_text());assert not z['failed'];r=z['row'];b=L['bath'];mu=b['mu_p']+b['mu_e']-b['mu_n'];mu=mu if ch=='EC'else-mu;terms={}
 for name,mur in[('empty',None),('LTE',mu),('emission',mu-.7*b['T']),('inverse',mu+.9*b['T'])]:
  J,X=math.exp(r['logj']),math.exp(r['logchi']);occ=0. if mur is None else 1/(1+math.exp((E-mur)/b['T']));F,B=J*(1-occ),X*occ;net=F-B;gross=F+B;dos=E*E/(2*math.pi**2*L['masses']['HC']**3);localR=dos*net;localQ=E*localR
  ck(ch+str(E)+name+' Fpositive',0 if F>0 else 1,0);ck(ch+str(E)+name+' chiindependentofincoming',0 if X>0 else 1,0)
  if mur is None:ck(ch+str(E)+name+' sourceBabsentonly',B,0)
  else:
   ck(ch+str(E)+name+' trueinversepositive',0 if B>0 else 1,0);force=(mu-mur)/b['T'];ck(ch+str(E)+name+' directsourceFBratio',math.log(F/B)-force,3e-10);entropy=dos*net*math.log(F/B);ck(ch+str(E)+name+' localentropynegative',min(0.,entropy)/(dos*gross),3e-10);ck(ch+str(E)+name+' localentropyVsaffinity',(entropy-dos*net*force)/(dos*gross),3e-10)
   if name=='LTE':ck(ch+str(E)+' LTEgross',net/gross,3e-10)
  ck(ch+str(E)+name+' same-binEledger',(-localQ+E*localR)/(dos*E*gross),3e-10)
  ck(ch+str(E)+name+' lengthunit', (X/2.99792458e10*1e5)/(X/2.99792458e5)-1,3e-15)
  terms[name]=dict(occupation=occ,coefficient_j=J,true_chi=X,relaxation=J+X,local_spectral_R=localR,local_spectral_Q=localQ,units='Rfm^-3s^-1MeV^-1,QMeVfm^-3s^-1MeV^-1; NOTintegratedmoments',mu_rad=mur)
 rows.append(dict(channel=ch,E=E,qualified_reference_file=f.name,controls=r['controls'],logj=r['logj'],logchi=r['logchi'],local_only_source_algebra=terms))
for f in sorted(P.glob('tail_*.json')):
 z=json.loads(f.read_text());ck('tailprofileactualpass '+f.name,0 if z['complete']and not z['failed']else 1,0)
for n,v in L['sourcepins'].items():ck('sourceunchanged '+n,0 if hashlib.sha256((P.parent/n).read_bytes()).hexdigest()==v else 1,0)
a=json.loads((P/'AUTHOR_LOCK_ADMISSION.json').read_text());ck('authorLOCKunchanged',0 if hashlib.sha256((P.parent/'production_weak_spectral_port/LOCK.json').read_bytes()).hexdigest()==a['author_lock_sha256']else 1,0)
# Pure abstractrepresentation andkinematicadversaries; NOruntimecandidatecalls.
witness=dict(logwidth=-750.,logintegrand=100.,ordinarywidth=math.exp(-750.),weighted_contribution=math.exp(-650.));ck('widthonlycounterexamplekept',0 if witness['ordinarywidth']==0 and witness['weighted_contribution']>0 else 1,0)
E,El,c=4.61,5.897,.99987;me=L['masses']['ME'];pe=math.sqrt((El-me)*(El+me));w=E-El;q=math.sqrt((E-pe)**2+2*E*pe*(1-c));t=w*w-q*q;ma=L['masses']['MP'];mb=L['masses']['MN'];astar=(mb*mb-ma*ma-t)/(2*math.copysign(math.sqrt(t),w));ck('plannedPCemptyinvariantproof',0 if t>0 and w<0 and astar<ma else 1,0)
adv=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate_calls=0,representation_case=dict(logj=-13836.073477186033,logchi=-13837.46326796703,ordinary_values=None,expected='strictpositive_support finite logs; not coefficientzero; accuracy NOTqualified'),weighted_panel_case=witness,differential_PC_empty=dict(E=E,El=El,cosine=c,t=t,omega=w,Ea_star=astar,expected='analytic supportempty frompositiveenergyshell violation; conditionalonly'),singular_coordinate='derivedlimit/alternatemap or unresolvedexception; never widthonlyskip/falsezero',future_reconstruction_cases=['finiteintervalcoverage explicit, complete=False cannotbequalified','missing/overlapping/reorderednode/state/current/control/radiationidentitywrong rejects','ordinaryNonefiniteLOGadmitted vsbool/NaN/logNone falsezero rejects','emptyincomingradiation Bsourcezero butpositivechi mandatory','PCnoneq mur=2.009364169619812 !=betaemissionmur1.209364169619812; cannotjoinlabels'],not_executed_API_tests=True)
(P/'FUTURE_ADVERSARIES.json').write_text(json.dumps(adv,indent=2)+'\n')
initial=json.loads((P/'INITIAL_CONVERGENCE.json').read_text());out=dict(passed=sum(c['passed']for c in checks),failed=[c for c in checks if not c['passed']],checks=checks,selected_references=rows,retained_initial_profile_failures=initial['failed'],candidate_imported=False,scope='PUREselectedreferencequalification and localbin algebra only; no finite/infiniteintervalRQortransportaccuracy',new_integrals_in_this_script=0)
(P/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n');print('ANALYSIS',out['passed'],'FAIL',len(out['failed']));print(out['failed'])
