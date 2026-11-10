#!/usr/bin/env python3
"""Full-data, resumable experiment driver for comparisons, ablations and sweeps."""
import argparse,csv,datetime as dt,hashlib,json,math,os,random,sys,time
from pathlib import Path
ROOT=Path('/root/wmsr-repro')
def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--spec',type=Path,required=True);args=parser.parse_args()
    spec=json.loads(args.spec.read_text());run=ROOT/'results/extended'/spec['id']
    if (run/'final_results.json').exists():print('ALREADY_COMPLETE',spec['id'],flush=True);return
    import numpy as np,torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset,DataLoader
    from focal_frequency_loss import FocalFrequencyLoss
    from pytorch_msssim import ssim
    sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'upstream'))
    from extended_models.factory import create_model
    from utils import rgb_to_ycbcr
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=True
    seed=spec.get('seed',10);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    dataset=spec['dataset'];scale=spec['scale'];epochs=spec['epochs']
    manifest_path=ROOT/'records/formal'/f'{dataset}_manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['count']==1000 and manifest['splits']['train']['count']==800 and manifest['splits']['test']['count']==200
    model,architecture=create_model(spec)
    source_hash=hashlib.sha256((ROOT/'extended_models/factory.py').read_bytes()).hexdigest()
    sources=json.loads((ROOT/'records/extended_sources.json').read_text())
    config={'spec':spec,'architecture':architecture,'train_count':800,'test_count':200,'hr_patch':96,'lr_patch':96//scale,
            'batch_size':8,'repeat':1,'seed':seed,'epochs':epochs,'expected_steps':epochs*100,
            'optimizer':{'type':'Adam','lr':.001,'betas':[.9,.999],'eps':1e-8},
            'scheduler':{'type':'StepLR','step_size':20,'gamma':.5},
            'loss':{'L1':spec.get('l1_weight',1.),'FFL':spec.get('ffl_weight',0.),'alpha':1},
            'sources':sources,'factory_sha256':source_hash,
            'trainer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'precision':{'dtype':'float32','cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32},
            'vendor_manifest_sha256':hashlib.sha256((ROOT/'records/vendor_integration.json').read_bytes()).hexdigest(),
            'data_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            'evaluation':{'checkpoint':'final epoch, no test-based selection','image':'BT601 Y, range255, scale border, per-image mean',
                          'physical':'unclipped mean of3channels inverse fixed[-2,40]C mapping, same border'},
            'boundary':'Explicit public-data protocol and source variants, not author-confirmed original paper experiment',
            'storage':'Keep active last.pt; after final checkpoint and metrics are safely written, retain compact final.pt only'}
    run.mkdir(parents=True,exist_ok=True);(run/'checkpoints').mkdir(exist_ok=True)
    if (run/'config.json').exists():assert json.loads((run/'config.json').read_text())==config,'Existing experiment configuration cannot change'
    save(run/'config.json',config)
    class SST(Dataset):
        def __init__(self,split,augment=False):
            part=manifest['splits'][split];path=ROOT/part['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==part['sha256']
            with np.load(path) as archive:self.hr=archive['hr_celsius'].copy();self.lr=archive[f'lr_x{scale}_celsius'].copy()
            self.samples=part['samples'];self.augment=augment
        def __len__(self):return len(self.hr)
        def __getitem__(self,i):
            low=self.lr[i];high=self.hr[i]
            if self.augment:
                if random.random()<.5:low=low[:,::-1];high=high[:,::-1]
                if random.random()<.5:low=low[::-1,:];high=high[::-1,:]
                if random.random()<.5:low=low.T;high=high.T
            return (torch.from_numpy(((low+2)/42*255).copy())[None].repeat(3,1,1),
                    torch.from_numpy(((high+2)/42*255).copy())[None].repeat(3,1,1))
    generator=torch.Generator().manual_seed(seed);train=SST('train',True)
    loader=DataLoader(train,batch_size=8,shuffle=True,num_workers=0,pin_memory=True,generator=generator)
    assert len(loader)==100
    model=model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.999),eps=1e-8)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,20,gamma=.5);frequency=FocalFrequencyLoss(loss_weight=1,alpha=1)
    first_epoch=1;step=0;prior=0;last=run/'checkpoints/last.pt'
    if last.exists():
        checkpoint=torch.load(last,map_location='cuda',weights_only=False);assert checkpoint['config']==config
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']);first_epoch=checkpoint['epoch']+1;step=checkpoint['step'];prior=checkpoint['elapsed_seconds']
        random.setstate(checkpoint['rng_python']);np.random.set_state(checkpoint['rng_numpy']);torch.set_rng_state(checkpoint['rng_torch'].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['rng_cuda']]);generator.set_state(checkpoint['rng_loader'].cpu())
    started=time.monotonic();fields=['epoch','step','lr','loss','l1','ffl','epoch_seconds','elapsed_seconds']
    with (run/'training_metrics.csv').open('a' if first_epoch>1 else 'w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        if first_epoch==1:writer.writeheader()
        for epoch in range(first_epoch,epochs+1):
            begin=time.monotonic();model.train();sums=np.zeros(3);used_lr=optimizer.param_groups[0]['lr']
            for low,high in loader:
                low=low.cuda(non_blocking=True);high=high.cuda(non_blocking=True);optimizer.zero_grad(set_to_none=True)
                prediction=model(low);l1=F.l1_loss(prediction,high)
                ffl=frequency(prediction if config['loss']['FFL'] else prediction.detach(),high)
                loss=config['loss']['L1']*l1+config['loss']['FFL']*ffl
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss at '+str(step))
                loss.backward()
                if step%100==0:assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                optimizer.step();step+=1;sums+=np.array([loss.item(),l1.item(),ffl.item()])
            scheduler.step();elapsed=prior+time.monotonic()-started
            row=dict(epoch=epoch,step=step,lr=used_lr,loss=sums[0]/100,l1=sums[1]/100,ffl=sums[2]/100,
                     epoch_seconds=time.monotonic()-begin,elapsed_seconds=elapsed)
            writer.writerow(row);stream.flush()
            state={'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'scheduler_state_dict':scheduler.state_dict(),
                   'epoch':epoch,'step':step,'elapsed_seconds':elapsed,'config':config,'rng_python':random.getstate(),
                   'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_loader':generator.get_state()}
            temp=last.with_suffix('.pt.tmp');torch.save(state,temp);temp.replace(last)
            save(run/'progress.json',{'status':'training','id':spec['id'],'epoch':epoch,'epochs':epochs,'step':step,'expected_steps':epochs*100,'last_epoch':row,'pid':os.getpid()})
            print('EPOCH',spec['id'],json.dumps(row),flush=True)
    assert step==epochs*100
    final=run/'checkpoints/final.pt';temp=final.with_suffix('.pt.tmp')
    torch.save({'model_state_dict':model.state_dict(),'epoch':epochs,'step':step,'config':config},temp);temp.replace(final)
    model.eval();test_data=SST('test');test=DataLoader(test_data,batch_size=8,shuffle=False,num_workers=0,pin_memory=True)
    def summarize(rows):
        return {'count':len(rows),'rmse_c':math.sqrt(sum(x['mse_c2'] for x in rows)/len(rows)),
                'mae_c':sum(x['mae_c'] for x in rows)/len(rows),'psnr_y_db':sum(x['psnr_y_db'] for x in rows)/len(rows),
                'ssim_y':sum(x['ssim_y'] for x in rows)/len(rows)}
    @torch.inference_mode()
    def evaluate(fn):
        rows=[]
        for low,high in test:
            low=low.cuda(non_blocking=True);high=high.cuda(non_blocking=True);prediction=fn(low);b=scale
            error=((prediction.mean(1,keepdim=True)-high.mean(1,keepdim=True))/255*42)[:,:,b:-b,b:-b]
            mse=error.square().flatten(1).mean(1);mae=error.abs().flatten(1).mean(1)
            y_sr=rgb_to_ycbcr(prediction.clamp(0,255))[:,0:1,b:-b,b:-b];y_hr=rgb_to_ycbcr(high.clamp(0,255))[:,0:1,b:-b,b:-b]
            psnr=-10*torch.log10(((y_sr.double()-y_hr.double())/255).square().flatten(1).mean(1).clamp_min(1e-16))
            structure=ssim(y_sr,y_hr,data_range=255,size_average=False)
            for a,c,d,e in zip(mse.tolist(),mae.tolist(),psnr.tolist(),structure.tolist()):rows.append({'mse_c2':a,'mae_c':c,'psnr_y_db':d,'ssim_y':e})
        return summarize(rows),rows
    metrics,rows=evaluate(model);baseline,base_rows=evaluate(lambda x:F.interpolate(x,scale_factor=scale,mode='bicubic',align_corners=False,antialias=False))
    for name,items in [('model',rows),('bicubic',base_rows)]:
        with (run/f'{name}_test_per_sample.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['sample_id','date','region_label','mse_c2','mae_c','psnr_y_db','ssim_y']);writer.writeheader()
            for meta,item in zip(test_data.samples,items):writer.writerow({'sample_id':meta['sample_id'],'date':meta['date'],'region_label':meta.get('region_label',''),**item})
    regions={}
    if dataset=='HYCOM':
        for label in ['NP','AGM','NIO','EWP']:
            indices=[i for i,meta in enumerate(test_data.samples) if meta['region_label']==label]
            regions[label]={'model':summarize([rows[i] for i in indices]),'bicubic':summarize([base_rows[i] for i in indices])}
    result={'status':'complete','spec':spec,'config':config,'epochs':epochs,'optimizer_steps':step,
            'test_model':metrics,'test_bicubic':baseline,'per_region':regions,
            'delta_psnr_y_db':metrics['psnr_y_db']-baseline['psnr_y_db'],'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'completed_at':dt.datetime.now(dt.timezone.utc).isoformat(),'elapsed_seconds':prior+time.monotonic()-started,
            'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_runtime':torch.version.cuda,
            'checkpoint_sha256':hashlib.sha256(final.read_bytes()).hexdigest()}
    assert metrics['count']==200 and all(math.isfinite(float(x)) for x in metrics.values())
    save(run/'final_results.json',result);save(run/'progress.json',result)
    last.unlink(missing_ok=True)
    print('EXPERIMENT_COMPLETE',spec['id'],json.dumps(metrics),flush=True)
if __name__=='__main__':main()
