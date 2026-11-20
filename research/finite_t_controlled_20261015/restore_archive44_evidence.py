"""Verify/restore archive44 selected evidence with full preflight."""
from pathlib import Path
import gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parent

def safe(relative):
    if not isinstance(relative,str):raise ValueError('Nonstring archive path')
    rel=Path(relative)
    if rel.is_absolute()or '..'in rel.parts:raise ValueError('Unsafe archive path')
    dest=(ROOT/rel).resolve()
    if ROOT not in dest.parents:raise ValueError('Outside archive')
    return dest

def main():
    args=sys.argv[1:]
    if any(x!='--verify-only'for x in args):raise ValueError('Only --verify-only accepted')
    manifest=json.loads((ROOT/'ARCHIVE44_EVIDENCE_MANIFEST.json').read_text())
    pending=[];verified={};compressed=0
    for original,record in manifest.items():
        dest=safe(original);source=safe(record['stored_path']);stored=source.read_bytes()
        if hashlib.sha256(stored).hexdigest()!=record['stored_sha256']:raise ValueError('Stored checksum mismatch: '+original)
        if record['encoding']=='gzip':raw=gzip.decompress(stored);compressed+=1
        elif record['encoding']=='utf-8':raw=stored
        else:raise ValueError('Unknown encoding')
        raw.decode('utf-8')
        if len(raw)!=record['original_size']or hashlib.sha256(raw).hexdigest()!=record['original_sha256']:raise ValueError('Original checksum mismatch: '+original)
        if dest in verified and verified[dest]!=raw:raise ValueError('Conflicting archive destinations')
        verified[dest]=raw
        if dest.exists()and dest.read_bytes()!=raw:raise FileExistsError('Refusing to overwrite changed evidence: '+original)
        if not dest.exists():pending.append((dest,raw))
    if '--verify-only'not in args:
        for dest,raw in pending:
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb')as out:out.write(raw)
    print(str(len(manifest))+' originals verified; '+str(compressed)+' compressed logical paths; '+str(0 if '--verify-only'in args else len(pending))+' restored')
if __name__=='__main__':main()
