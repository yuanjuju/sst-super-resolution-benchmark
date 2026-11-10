#!/usr/bin/env python3
"""Run all3datasets x3scales at full100epochs; vary concurrency only, never data size."""
import datetime as dt,json,os,subprocess,time
from pathlib import Path
ROOT=Path("/root/wmsr-repro")
def write(value):
    path=ROOT/"records/formal_queue_status.json";tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2)+"\n");tmp.replace(path)
def main():
    jobs=[{"dataset":name,"scale":scale,"attempts":0} for scale in [4,3,2] for name in ["OISST","GHRSST","HYCOM"]]
    active=[];completed=[];failed=[];parallel=6
    env=os.environ.copy()
    env.update(XDG_CACHE_HOME=str(ROOT/".cache"),MPLCONFIGDIR=str(ROOT/".cache/matplotlib"),
               TRITON_CACHE_DIR=str(ROOT/".cache/triton"),OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1",CUDA_VISIBLE_DEVICES="0")
    started=dt.datetime.now(dt.timezone.utc).isoformat()
    while jobs or active:
        for entry in active[:]:
            code=entry["process"].poll()
            if code is None:continue
            entry["stream"].close();active.remove(entry)
            job=entry["job"]
            path=ROOT/"results/formal"/(job["dataset"]+"_x"+str(job["scale"])+"_seed10")/"final_results.json"
            if code==0 and path.exists():
                result=json.loads(path.read_text())
                completed.append({"dataset":job["dataset"],"scale":job["scale"],"result_file":str(path.relative_to(ROOT)),
                                  "test_wmsr":result["test_wmsr"],"test_bicubic":result["test_bicubic"]})
                print("QUEUE_COMPLETED",job["dataset"],job["scale"],flush=True)
            else:
                text=Path(entry["log"]).read_text()[-6000:]
                if "out of memory" in text.lower() and job["attempts"]<3:
                    parallel=max(1,parallel-1);jobs.append(job)
                    print("RETRY_WITH_LESS_CONCURRENCY",job["dataset"],job["scale"],parallel,flush=True)
                else:
                    failed.append({"dataset":job["dataset"],"scale":job["scale"],"returncode":code,
                                   "log":entry["log"],"tail":text[-2000:]})
                    print("QUEUE_FAILED",job["dataset"],job["scale"],code,flush=True)
        for job in jobs[:]:
            if len(active)>=parallel:break
            manifest=ROOT/"records/formal"/(job["dataset"]+"_manifest.json")
            if not manifest.exists():continue
            data=json.loads(manifest.read_text())
            assert data["count"]==1000 and data["splits"]["train"]["count"]==800 and data["splits"]["test"]["count"]==200
            free=int(subprocess.check_output(["nvidia-smi","--query-gpu=memory.free","--format=csv,noheader,nounits"],text=True).strip())
            if free<4096:break
            log=ROOT/"logs"/("formal_"+job["dataset"]+"_x"+str(job["scale"])+".log")
            stream=log.open("a")
            command=[str(ROOT/".venv/bin/python"),"-u",str(ROOT/"scripts/11_train_paper.py"),
                     "--dataset",job["dataset"],"--scale",str(job["scale"]),"--epochs","100"]
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            job["attempts"]+=1
            active.append({"job":job,"process":process,"stream":stream,"log":str(log)})
            jobs.remove(job)
            print("QUEUE_STARTED",job["dataset"],job["scale"],"PID",process.pid,flush=True)
            time.sleep(2)
        snapshot={"status":"running" if jobs or active else ("complete" if not failed else "needs_attention"),
                  "started_at":started,"updated_at":dt.datetime.now(dt.timezone.utc).isoformat(),
                  "total_jobs":9,"epochs_per_job":100,"train_images_per_job":800,"test_images_per_job":200,
                  "concurrency_limit":parallel,
                  "active":[{"dataset":x["job"]["dataset"],"scale":x["job"]["scale"],"pid":x["process"].pid,"log":x["log"]} for x in active],
                  "pending":jobs,"completed":completed,"failed":failed,
                  "interpretation":"Full paper-size datasets; explicitly inferred topology/encoding details, not author-confirmed exact reproduction."}
        write(snapshot)
        if jobs or active:time.sleep(10)
    print("FORMAL_QUEUE_FINISHED",flush=True)
if __name__=="__main__":main()

