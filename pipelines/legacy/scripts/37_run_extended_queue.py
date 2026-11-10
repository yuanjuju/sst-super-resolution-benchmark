#!/usr/bin/env python3
"""Full-size queue; preserve batch/data/epochs and adapt concurrent job count only."""
import datetime as dt,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
class ExistingProcess:
    def __init__(self,pid,identifier):self.pid=pid;self.identifier=identifier
    def poll(self):
        path=Path('/proc')/str(self.pid)
        try:
            state=(path/'stat').read_text().split()[2]
            command=(path/'cmdline').read_bytes()
            if state!='Z' and b'34_train_experiment.py' in command and self.identifier.encode() in command:return None
        except FileNotFoundError:pass
        return 0 if (ROOT/'results/extended'/self.identifier/'final_results.json').exists() else 1
def save(value):
    path=ROOT/'records/extended_queue_status.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def main():
    plan=json.loads((ROOT/'records/extended_plan.json').read_text());jobs=[];active=[];done=[];failed=[];parallel=2
    specs={x['id']:x for x in plan['jobs']};old_status=ROOT/'records/extended_queue_status.json'
    if old_status.exists():
        old=json.loads(old_status.read_text())
        for entry in old.get('active',[]):
            identifier=entry['id'];pid=entry['pid']
            if identifier not in specs or (ROOT/'results/extended'/identifier/'final_results.json').exists():continue
            process=ExistingProcess(pid,identifier)
            if process.poll() is None:
                log=ROOT/'logs'/('extended_'+identifier+'.log')
                active.append({'job':{'spec':specs[identifier],'attempts':1},'process':process,'stream':log.open('a'),'log':str(log)})
                print('RECOVERED_RUNNING_JOB',identifier,pid,flush=True)
    priority={'ablation':0,'comparison':1,'channel_sweep':2,'group_sweep':3,'loss_sweep':4}
    for spec in sorted(plan['jobs'],key=lambda x:priority[x['kind']]):
        if any(x['job']['spec']['id']==spec['id'] for x in active):continue
        path=ROOT/'results/extended'/spec['id']/'final_results.json'
        if path.exists():
            d=json.loads(path.read_text());assert d['status']=='complete' and d['optimizer_steps']==spec['epochs']*100;done.append(spec['id'])
        else:jobs.append({'spec':spec,'attempts':0})
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
        XDG_CACHE_HOME=str(ROOT/'.cache'),MPLCONFIGDIR=str(ROOT/'.cache/matplotlib'),TRITON_CACHE_DIR=str(ROOT/'.cache/triton'),CUDA_VISIBLE_DEVICES='0')
    env['CPATH']=str(next((ROOT/'.toolchains/python').glob('*/include/python3.10/Python.h')).parent)
    env['CUDA_CACHE_PATH']=str(ROOT/'.cache/cuda5090')
    started=dt.datetime.now(dt.timezone.utc).isoformat();wait_reason=None
    while jobs or active:
        for entry in active[:]:
            code=entry['process'].poll()
            if code is None:continue
            entry['stream'].close();active.remove(entry);job=entry['job'];spec=job['spec']
            path=ROOT/'results/extended'/spec['id']/'final_results.json'
            if code==0 and path.exists():
                d=json.loads(path.read_text());assert d['status']=='complete' and d['epochs']==spec['epochs'] and d['optimizer_steps']==spec['epochs']*100 and d['test_model']['count']==200
                done.append(spec['id']);print('QUEUE_COMPLETE',spec['id'],flush=True)
            else:
                tail=Path(entry['log']).read_text()[-7000:]
                if 'out of memory' in tail.lower() and job['attempts']<3:
                    parallel=max(1,parallel-1);jobs.insert(0,job);print('RETRY_LOWER_CONCURRENCY',spec['id'],parallel,flush=True)
                else:failed.append({'id':spec['id'],'returncode':code,'log':entry['log'],'tail':tail[-2500:]});print('QUEUE_FAILED',spec['id'],flush=True)
        wait_reason=None
        for job in jobs[:]:
            if len(active)>=parallel:break
            spec=job['spec'];ready=ROOT/'records/extended_ready'/(spec['preflight_key']+'.json')
            if not ready.exists():continue
            proof=json.loads(ready.read_text());assert proof['pass'] and proof['real_batch']==8
            disk=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize
            if disk<8*2**30:wait_reason='Less than8GiB free. No old files are automatically removed.';break
            free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
            required=max(4*2**30,proof['peak_allocated_bytes']*1.6+proof['parameters']*8)
            if free<required:wait_reason='Waiting for GPU memory with unchanged batch8';continue
            log=ROOT/'logs'/('extended_'+spec['id']+'.log');stream=log.open('a')
            p=subprocess.Popen([str(ROOT/'.venv5090/bin/python'),'-u',str(ROOT/'scripts/34_train_experiment.py'),
                '--spec',str(ROOT/'configs/extended'/(spec['id']+'.json'))],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            job['attempts']+=1;active.append({'job':job,'process':p,'stream':stream,'log':str(log)});jobs.remove(job)
            print('QUEUE_STARTED',spec['id'],p.pid,flush=True);time.sleep(2)
        missing=sorted({x['spec']['model'] for x in jobs if not (ROOT/'records/extended_ready'/(x['spec']['preflight_key']+'.json')).exists()})
        save({'status':'running' if jobs or active else ('complete' if not failed else 'needs_attention'),
              'started_at':started,'updated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'total_jobs':len(plan['jobs']),
              'concurrency_limit':parallel,'active':[{'id':x['job']['spec']['id'],'pid':x['process'].pid} for x in active],
              'pending':[x['spec']['id'] for x in jobs],'completed':done,'failed':failed,
              'waiting_for_preflight':missing,'wait_reason':wait_reason,'aliases':plan['aliases'],
              'data_protocol':'All runs retain800train/200test, HR96, batch8; ablations20epochs, others100epochs.'})
        if jobs or active:time.sleep(15)
    print('EXTENDED_QUEUE_FINISHED',flush=True)
if __name__=='__main__':main()
