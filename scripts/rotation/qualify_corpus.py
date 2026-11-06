"""Validate a returned independent RNS corpus, keeping failed cases explicit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
from nmbackend.rotating.rns import load_background


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('corpus',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    records=[]
    for archive in sorted(args.corpus.glob('*.tar.gz')):
        with tarfile.open(archive) as tar,tempfile.TemporaryDirectory() as directory:
            rows={}
            for name in ('rns_spherical.csv','rns_rotating.csv'):
                raw=tar.extractfile('result/'+name).read()
                target=Path(directory)/name
                target.write_bytes(raw)
                rows[name]=load_background(target).metadata
            records.append({'archive':archive.name,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'spherical':rows['rns_spherical.csv'],'rotating':rows['rns_rotating.csv']})
    if len(records)!=10:
        raise RuntimeError(f'Expected ten returned tasks, got {len(records)}')
    args.output.write_text(json.dumps({'records':records,'scope':'export/kinematic qualification only; stability and continuum accuracy not established'},indent=2)+'\n')
    print(f'{len(records)} tasks / {2*len(records)} exports passed schema and kinematic checks')


if __name__=='__main__':
    main()
