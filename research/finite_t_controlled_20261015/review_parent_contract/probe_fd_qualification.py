"""Independent parent checks of the FD source-qualification reuse contract."""
from pathlib import Path
import copy,hashlib,json,sys,tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'fd_fluid_element'))
import qualification_contract as q


def main():
    path=ROOT/'fd_fluid_element/source_qualification.json'
    source=Path(q.__file__)
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    if before!='92cbb4eede20914741d7c7b51f85c95f130e47bd6aa509f8915206ff3f12a1ff':
        raise ValueError('Unexpected reviewed source pin')
    report=json.loads(path.read_text());expected=q.fingerprint(2.,1e-4,.3);checks=[]
    def reject(name,fn):
        try:fn();ok=False
        except ValueError:ok=True
        checks.append(dict(name=name,passed=ok))
    checks.append(dict(name='exact current physical payload accepted',passed=q.validate(report,expected)['passed'] is True))
    reject('different solved FD density rejected',lambda:q.validate(report,q.fingerprint(2.,1.01e-4,.3)))
    wrong=copy.deepcopy(expected);key=next(iter(wrong['source_sha256']));wrong['source_sha256'][key]='0'*64
    reject('changed upstream source expected hash rejected',lambda:q.validate(report,wrong))
    wrong=copy.deepcopy(expected);wrong['controls']['edges'][-1]+=1
    reject('changed energy panel edges rejected',lambda:q.validate(report,wrong))
    old=copy.deepcopy(report);old.pop('qualification_schema')
    reject('legacy unqualified payload rejected',lambda:q.validate(old,expected))
    bad=copy.deepcopy(report);bad['extra_undigested_edit']=1
    reject('result corruption rejected',lambda:q.validate(bad,expected))
    bad=copy.deepcopy(report);bad['coarse_source']['outer_order']=7
    raw={k:v for k,v in bad.items() if k not in ('qualification_schema','qualification_fingerprint','qualification_fingerprint_sha256','qualified_payload_sha256')}
    bad['qualified_payload_sha256']=q.digest(raw)
    reject('inconsistent source metadata rejected even with recomputed payload digest',lambda:q.validate(bad,expected))
    with tempfile.TemporaryDirectory() as d:
        reject('missing physical qualification rejected',lambda:q.load(Path(d)/'absent.json',expected))
    after=hashlib.sha256(source.read_bytes()).hexdigest()
    checks.append(dict(name='author source unchanged',passed=before==after))
    out=dict(passed=all(c['passed'] for c in checks),checks=checks,source_sha256=before,
             scope='Source reuse provenance only; no new physics quadrature or temporal qualification')
    Path(__file__).with_name('results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(passed=out['passed'],checks=len(checks))))
    if not out['passed']:raise SystemExit(1)

if __name__=='__main__':main()
