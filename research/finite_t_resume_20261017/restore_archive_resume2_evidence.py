"""Verify/restore archive_resume2 evidence; refuse changed files and unsafe paths."""
from pathlib import Path
import gzip,hashlib,json,sys
ROOT=Path(__file__).resolve().parent

def safe(relative):
    rel=Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError('Unsafe archive-relative path: '+relative)
    path=(ROOT/rel).resolve()
    if ROOT not in path.parents:
        raise ValueError('Path outside archive: '+relative)
    return path

def main():
    args=sys.argv[1:]
    if any(x!='--verify-only' for x in args):
        raise ValueError('Only --verify-only is supported')
    manifest=json.loads((ROOT/'ARCHIVE_RESUME2_EVIDENCE_MANIFEST.json').read_text())
    pending=[];compressed=0
    # Preflight the ENTIRE archive before writing any extracted evidence.
    for original,record in manifest.items():
        source=safe(record['stored_path']);dest=safe(original)
        packed=source.read_bytes()
        if hashlib.sha256(packed).hexdigest()!=record['stored_sha256']:
            raise ValueError('Stored checksum mismatch: '+str(source))
        if record['encoding']=='gzip':
            raw=gzip.decompress(packed);compressed+=1
        elif record['encoding']=='utf-8':
            raw=packed;raw.decode('utf-8')
        else:
            raise ValueError('Unknown archive encoding')
        if hashlib.sha256(raw).hexdigest()!=record['original_sha256']:
            raise ValueError('Original checksum mismatch: '+original)
        if dest.exists() and dest.read_bytes()!=raw:
            raise FileExistsError('Refusing to overwrite changed evidence: '+str(dest))
        if not dest.exists():pending.append((dest,raw))
    if '--verify-only' not in args:
        for dest,raw in pending:
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Exclusive create protects a change appearing after preflight.
            with dest.open('xb') as stream:stream.write(raw)
    print(str(len(manifest))+' originals verified; '+str(compressed)+
          ' compressed; '+str(0 if '--verify-only' in args else len(pending))+' restored')

if __name__=='__main__':main()
