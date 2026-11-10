import json,subprocess,concurrent.futures
from pathlib import Path
r=Path('/root/wmsr-repro')
repos={'EDSR': 'sanghyun-son/EDSR-PyTorch', 'ELAN': 'xindongzhang/ELAN', 'DAT': 'zhengchen1999/DAT', 'SAFMN': 'sunny2109/SAFMN', 'MambaIR': 'csguoh/MambaIR', 'CRAN_unofficial': 'Ast-363/CRAN', 'CGC': 'XudongLinthu/context-gated-convolution'}
def fetch(item):
 name,repo=item;p=r/'external'/name
 if not p.exists():subprocess.run(['git','clone','--depth','1','https://github.com/'+repo+'.git',str(p)],check=True,timeout=180)
 commit=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
 return name,{'url':'https://github.com/'+repo,'commit':commit}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=dict(pool.map(fetch,repos.items()))
(r/'records/extended_sources.json').write_text(json.dumps(results,indent=2)+'\n')
print('ALL_BASELINE_SOURCES_READY',json.dumps(results),flush=True)
