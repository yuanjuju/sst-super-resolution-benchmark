"""Verify downloaded/extracted release weights without loading pickle or Torch."""
from pathlib import Path
import argparse,hashlib,json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);parser.add_argument('--require-all',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'weights/manifest.json').read_text())
    found=0;missing=[]
    for item in manifest:
        path=args.directory/item['archive_path']
        if not path.exists():missing.append(item['archive_path']);continue
        assert path.stat().st_size==item['bytes'],str(path)
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(2**20),b''):h.update(chunk)
        assert h.hexdigest()==item['sha256'],str(path)
        found+=1
    assert found,'No indexed weights found in this directory'
    if args.require_all:assert not missing,f'Missing {len(missing)} files'
    print(json.dumps({'verified':found,'not_downloaded':len(missing),'all_present':not missing}))

if __name__=='__main__':main()
