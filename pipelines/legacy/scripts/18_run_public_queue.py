#!/usr/bin/env python3
"""Complete all nine released-network runs; retain reconstruction failures separately."""
import datetime as dt,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/wmsr-repro')

def save(snapshot):
    path=ROOT/'records/public_queue_status.json';temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(snapshot,indent=2)+'\n');temp.replace(path)

def main():
    jobs=[];active=[];completed=[];failed=[];parallel=4
    for scale in [4,3,2]:
        for name in ['OISST','HYCOM','GHRSST']:
            path=ROOT/'results/public_control'/f'{name}_x{scale}_seed10/final_results.json'
            if path.exists():
                result=json.loads(path.read_text())
                assert result['epochs']==100 and result['optimizer_steps']==10000
                completed.append({'dataset':name,'scale':scale,'result_file':str(path.relative_to(ROOT))})
            else:jobs.append({'dataset':name,'scale':scale,'attempts':0})
    env=os.environ.copy()
    env.update(XDG_CACHE_HOME=str(ROOT/'.cache'),MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),
               TRITON_CACHE_DIR=str(ROOT/'.cache/triton'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0')
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    while jobs or active:
        for entry in active[:]:
            code=entry['process'].poll()
            if code is None:continue
            entry['stream'].close();active.remove(entry);job=entry['job']
            path=ROOT/'results/public_control'/f"{job['dataset']}_x{job['scale']}_seed10/final_results.json"
            if code==0 and path.exists():
                result=json.loads(path.read_text());assert result['epochs']==100 and result['optimizer_steps']==10000
                completed.append({'dataset':job['dataset'],'scale':job['scale'],'result_file':str(path.relative_to(ROOT))})
                print('PUBLIC_COMPLETED',job['dataset'],job['scale'],flush=True)
            else:
                tail=Path(entry['log']).read_text()[-6000:]
                if 'out of memory' in tail.lower() and job['attempts']<3:
                    parallel=max(1,parallel-1);jobs.append(job)
                else:failed.append({'dataset':job['dataset'],'scale':job['scale'],'returncode':code,'tail':tail[-2000:]})
        for job in jobs[:]:
            if len(active)>=parallel:break
            log=ROOT/'logs'/f"public_{job['dataset']}_x{job['scale']}.log";stream=log.open('a')
            command=[str(ROOT/'.venv/bin/python'),'-u',str(ROOT/'scripts/11_train_public_control.py'),
                     '--dataset',job['dataset'],'--scale',str(job['scale']),'--epochs','100']
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            job['attempts']+=1;active.append({'job':job,'process':process,'stream':stream,'log':str(log)})
            jobs.remove(job);print('PUBLIC_STARTED',job['dataset'],job['scale'],process.pid,flush=True);time.sleep(2)
        snapshot={'status':'running' if jobs or active else ('complete' if not failed else 'needs_attention'),
                  'started_at':started,'updated_at':dt.datetime.now(dt.timezone.utc).isoformat(),
                  'total_jobs':9,'epochs_per_job':100,'train_images_per_job':800,'test_images_per_job':200,
                  'concurrency_limit':parallel,
                  'active':[{'dataset':x['job']['dataset'],'scale':x['job']['scale'],'pid':x['process'].pid,'log':x['log']} for x in active],
                  'pending':jobs,'completed':completed,'failed':failed,
                  'interpretation':'Unmodified released network, reconstructed public SST data, paper-interpreted training; not private-paper protocol.'}
        save(snapshot)
        if jobs or active:time.sleep(10)
    if failed:raise SystemExit(1)
    raise SystemExit(subprocess.call([str(ROOT/'.venv/bin/python'),str(ROOT/'scripts/19_make_public_report.py')],cwd=ROOT,env=env))

if __name__=='__main__':main()
