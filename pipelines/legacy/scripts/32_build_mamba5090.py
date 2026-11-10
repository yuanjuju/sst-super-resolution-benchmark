#!/usr/bin/env python3
"""Build the official selective-scan implementation for sm_120 inside this project."""
import hashlib,json,os,shutil,subprocess,urllib.request,concurrent.futures,time
from pathlib import Path
ROOT=Path('/root/wmsr-repro')

def main():
    while not (ROOT/'.venv5090/lib/python3.10/site-packages/torch').exists():time.sleep(5)
    components=json.loads((ROOT/'records/cuda128_components.json').read_text())
    toolkit=ROOT/'.toolchains/cuda-12.8.1';toolkit.mkdir(parents=True,exist_ok=True)
    cache=ROOT/'.cache/cuda128';cache.mkdir(parents=True,exist_ok=True)
    def fetch(item):
        name,meta=item;path=cache/Path(meta['relative_path']).name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']:
            temporary=path.with_suffix('.part')
            with urllib.request.urlopen('https://developer.download.nvidia.com/compute/cuda/redist/'+meta['relative_path'],timeout=120) as src,temporary.open('wb') as dst:shutil.copyfileobj(src,dst)
            assert hashlib.sha256(temporary.read_bytes()).hexdigest()==meta['sha256'];temporary.replace(path)
        out=cache/name;out.mkdir(exist_ok=True)
        subprocess.run(['tar','-xJf',str(path),'--strip-components=1','-C',str(out)],check=True)
        return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:directories=list(pool.map(fetch,components.items()))
    for directory in directories:shutil.copytree(directory,toolkit,dirs_exist_ok=True,symlinks=False)
    subprocess.run([str(toolkit/'bin/nvcc'),'--version'],check=True)
    source=ROOT/'external/mamba5090'
    if not source.exists():subprocess.run(['git','clone','--depth','1','--branch','v2.2.5','https://github.com/state-spaces/mamba.git',str(source)],check=True)
    setup=source/'setup.py';text=setup.read_text()
    if 'arch=compute_53,code=sm_53' in text:
        begin=text.index('        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_53,code=sm_53")')
        end=text.index('\n    # HACK:',begin)
        text=text[:begin]+'        cc_flag += ["-gencode", "arch=compute_120,code=sm_120"]\n'+text[end:]
    setup.write_text(text)
    diff=subprocess.check_output(['git','-C',str(source),'diff'],text=True)
    (ROOT/'records/mamba5090_build.patch').write_text(diff)
    env=os.environ.copy();env.update(CUDA_HOME=str(toolkit),TORCH_CUDA_ARCH_LIST='12.0',MAMBA_FORCE_BUILD='TRUE',
        MAX_JOBS='2',UV_CACHE_DIR=str(ROOT/'.cache/uv5090'),XDG_CACHE_HOME=str(ROOT/'.cache'),
        PATH=str(toolkit/'bin')+':'+env['PATH'])
    headers=list((ROOT/'.toolchains/python').glob('*/include/python3.10/Python.h'))
    assert headers,'Install isolated uv Python3.10.12 to provide development headers without global packages'
    env['CPATH']=str(headers[0].parent)+(':'+env['CPATH'] if env.get('CPATH') else '')
    subprocess.run(['uv','pip','install','--python',str(ROOT/'.venv5090/bin/python'),'--no-build-isolation','--no-deps',str(source)],env=env,check=True)
    record={'source':'https://github.com/state-spaces/mamba','tag':'v2.2.5',
            'commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
            'architecture':'sm_120','patch':'Compiler architecture flags only; selective-scan equations unchanged',
            'cuda_home':str(toolkit)}
    (ROOT/'records/mamba5090_build.json').write_text(json.dumps(record,indent=2)+'\n')
    print('MAMBA_SM120_BUILD_COMPLETE',flush=True)

if __name__=='__main__':main()
