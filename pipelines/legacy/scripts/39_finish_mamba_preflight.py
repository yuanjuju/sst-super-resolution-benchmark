#!/usr/bin/env python3
"""Gate Mamba experiment readiness on successful native-kernel and replay acceptance."""
import json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
while not (ROOT/'records/mamba5090_build.json').exists():time.sleep(10)
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
    XDG_CACHE_HOME=str(ROOT/'.cache'),MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'))
env['CPATH']=str(next((ROOT/'.toolchains/python').glob('*/include/python3.10/Python.h')).parent)
env['TRITON_CACHE_DIR']=str(ROOT/'.cache/triton');env['CUDA_CACHE_PATH']=str(ROOT/'.cache/cuda5090')
subprocess.run([str(ROOT/'.venv5090/bin/python'),'-u',str(ROOT/'scripts/38_verify_blackwell.py')],cwd=ROOT,env=env,check=True)
subprocess.run([str(ROOT/'.venv5090/bin/python'),'-u',str(ROOT/'scripts/35_preflight_extended.py'),
                '--models','WMSR','WMSR_ABLATION','WMSR_GROUPED','MambaIR'],cwd=ROOT,env=env,check=True)
print('MAMBA_EXPERIMENT_PREFLIGHT_FINISHED',flush=True)
