"""Run verification and figures only after both fixed jobs complete."""
import os
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
env=os.environ.copy()
for key,name in {'PYTHONPYCACHEPREFIX':'pycache','XDG_CACHE_HOME':'xdg','TRITON_CACHE_DIR':'triton',
                 'TORCHINDUCTOR_CACHE_DIR':'inductor','CUDA_CACHE_PATH':'cuda','PYTORCH_KERNEL_CACHE_PATH':'torch_kernels',
                 'TMPDIR':'tmp','MPLCONFIGDIR':'matplotlib'}.items():
    path=HERE/'.cache'/name;path.mkdir(parents=True,exist_ok=True);env[key]=str(path)
env.update(CPATH='/root/wmsr-repro/.toolchains/python/cpython-3.10.12-linux-x86_64-gnu/include/python3.10',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
for script in ['finalize_results.py','visualize_results.py','visualize_ema.py','compare_variants.py']:
    subprocess.run([sys.executable,str(HERE/script)],cwd=HERE,env=env,check=True)
