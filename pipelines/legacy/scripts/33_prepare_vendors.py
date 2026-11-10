#!/usr/bin/env python3
"""Vendor only architecture files, recording integration-only source patches."""
from pathlib import Path
import hashlib,json,difflib
ROOT=Path('/root/wmsr-repro')
def main():
    out=ROOT/'extended_models/vendor';out.mkdir(parents=True,exist_ok=True)
    (out/'__init__.py').touch();(out.parent/'__init__.py').touch()
    files={
      'edsr_common.py':'EDSR/src/model/common.py','edsr.py':'EDSR/src/model/edsr.py',
      'elan_block.py':'ELAN/models/elan_block.py','elan.py':'ELAN/models/elan_network.py',
      'dat.py':'DAT/basicsr/archs/dat_arch.py','safmn.py':'SAFMN_source/basicsr/archs/safmn_arch.py',
      'mambair.py':'MambaIR/basicsr/archs/mambair_arch.py',
      'cran_common.py':'CRAN_unofficial/model/common.py','cran_reference.py':'CRAN_unofficial/model/rcan.py'}
    records={}
    for name,source in files.items():
        original=(ROOT/'external'/source).read_text();text=original
        text=text.replace('from model import common','from extended_models.vendor import edsr_common as common' if name=='edsr.py' else 'from extended_models.vendor import cran_common as common')
        text=text.replace('from models.elan_block import','from extended_models.vendor.elan_block import')
        text=text.replace('from basicsr.utils.registry import ARCH_REGISTRY','').replace('@ARCH_REGISTRY.register()','')
        if name=='cran_common.py':
            text=text.replace('requires_grad=True).cuda()','requires_grad=True)')
        (out/name).write_text(text)
        records[name]={'upstream_file':source,'upstream_sha256':hashlib.sha256(original.encode()).hexdigest(),
                       'integrated_sha256':hashlib.sha256(text.encode()).hexdigest(),
                       'patch':''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile=source,tofile=name))}
    (ROOT/'records/vendor_integration.json').write_text(json.dumps(records,indent=2)+'\n')
    print('VENDORS_PREPARED',len(records),flush=True)
if __name__=='__main__':main()
