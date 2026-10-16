"""Restore losslessly compressed checkpoint evidence without overwriting edits."""
from pathlib import Path
import gzip
import hashlib
import json
import sys
ROOT=Path(__file__).resolve().parent


def main():
    manifest=json.loads((ROOT/'ARCHIVE32_EVIDENCE_MANIFEST.json').read_text())
    count=0
    for original,record in manifest.items():
        if record['encoding']!='gzip':
            continue
        source=ROOT/record['stored_path']
        packed=source.read_bytes()
        if hashlib.sha256(packed).hexdigest()!=record['stored_sha256']:
            raise ValueError(f'Compressed checksum mismatch: {source}')
        raw=gzip.decompress(packed)
        if hashlib.sha256(raw).hexdigest()!=record['original_sha256']:
            raise ValueError(f'Original checksum mismatch: {original}')
        dest=ROOT/original
        if dest.exists() and dest.read_bytes()!=raw:
            raise FileExistsError(f'Refusing to overwrite changed evidence: {dest}')
        if '--verify-only' not in sys.argv[1:] and not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(raw)
        count+=1
    print(f'{count} compressed evidence files verified; existing edits preserved')


if __name__=='__main__':
    main()
