"""Verify the natural-image SR archive; --images also recomputes RGB PSNR."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import math
import struct
import zipfile

ROOT=Path(__file__).resolve().parents[1]
MODULE=ROOT/'foundations/natural-image-sr'


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def main(images=False):
    provenance=json.loads((MODULE/'checks/publication_provenance.json').read_text())
    for item in provenance:
        p=ROOT/item['path']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['published_sha256'],item['path']
    manifest=json.loads((MODULE/'data/manifest.json').read_text())
    assert len(manifest)==10
    for item in manifest:
        raw=(MODULE/item['path']).read_bytes()
        assert raw[:8]==b'\x89PNG\r\n\x1a\n'
        assert struct.unpack('>II',raw[16:24])==(item['width'],item['height'])
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
    for sample in {r['sample_id'] for r in manifest}:
        hr=next(r for r in manifest if r['sample_id']==sample and r['role']=='HR')
        lr=next(r for r in manifest if r['sample_id']==sample and r['role']=='LR_x4')
        assert (hr['width'],hr['height'])==(4*lr['width'],4*lr['height'])
    ledger=rows(MODULE/'results/metrics.csv')
    assert len(ledger)==41 and rows(ROOT/'tables/natural_image_sr/metrics.csv')==ledger
    training=rows(ROOT/'tables/natural_image_sr/training_validation.csv')
    assert len(training)==25
    assert sum(r['kind']=='scheduled' for r in training)==24
    assert sum(r['kind']=='end_of_training_repeat' for r in training)==1
    for arm in ['fft','nofft']:
        selected=[r for r in training if r['experiment']=='SAFMN_pair_5090_'+arm]
        assert [int(r['step']) for r in selected]==[5000,10000,15000,20000,25000,30000]
    per=rows(ROOT/'tables/natural_image_sr/set5_per_image.csv')
    mean=rows(ROOT/'tables/natural_image_sr/set5_mean.csv')
    assert len(per)==20 and len(mean)==4
    metrics=['psnr_rgb_db','ssim_rgb','psnr_y_db','ssim_y']
    for group in mean:
        selected=[r for r in per if r['arm']==group['arm'] and r['iter']==group['iter']]
        assert {r['image'] for r in selected}=={'baby','bird','butterfly','head','woman'}
        for key in metrics:
            assert math.isclose(sum(float(r[key]) for r in selected)/5,float(group[key]),abs_tol=1e-12)
    assert all((ROOT/r['prediction_path']).is_file() for r in per)
    recomputed=0
    if images:
        import numpy as np
        from PIL import Image
        for row in per:
            pred=np.array(Image.open(ROOT/row['prediction_path']).convert('RGB'),dtype=np.float64)[4:-4,4:-4]
            gt=np.array(Image.open(MODULE/'datasets/Set5/HR'/f'{row["image"]}.png').convert('RGB'),dtype=np.float64)[4:-4,4:-4]
            measured=10*math.log10(255**2/float(np.mean((pred-gt)**2)))
            assert math.isclose(measured,float(row['psnr_rgb_db']),abs_tol=1e-10),(row['image'],measured)
            recomputed+=1
        for row in rows(ROOT/'tables/natural_image_sr/local_inference.csv'):
            record=json.loads((ROOT/row['source']).read_text())
            pred=np.array(Image.open((ROOT/row['source']).parent/'sr.png').convert('RGB'),dtype=np.float64)[4:-4,4:-4]
            gt=np.array(Image.open(MODULE/'datasets/Set5/HR'/f'{record["input"].split("/")[-1]}').convert('RGB'),dtype=np.float64)[4:-4,4:-4]
            measured=10*math.log10(255**2/float(np.mean((pred-gt)**2)))
            assert math.isclose(measured,float(row['psnr_rgb_db']),abs_tol=1e-10)
            recomputed+=1
    for path in MODULE.rglob('*.py'):
        ast.parse(path.read_text())
    workbook=ROOT/'tables/natural_image_sr/experiments.xlsx'
    with zipfile.ZipFile(workbook) as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if '/charts/chart' in n and n.endswith('.xml')])==2
        from xml.etree import ElementTree as ET
        ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
        book=ET.fromstring(z.read('xl/workbook.xml'))
        assert len(book.findall('s:sheets/s:sheet',ns))==5
        for name in z.namelist():
            if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                for cell in ET.fromstring(z.read(name)).findall('.//s:c',ns):
                    assert cell.get('t')!='e',(name,cell.attrib)
    result=dict(status='pass',source_files=len(provenance),metrics=len(ledger),
        validation_records=len(training),set5_source_images=len(manifest),
        set5_predictions=len(per),set5_means=len(mean),
        local_inference=3,rgb_psnr_recomputed=recomputed,workbook_sheets=5,workbook_charts=2)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',action='store_true')
    main(parser.parse_args().images)
