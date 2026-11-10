"""Verify and copy trusted1000-step inputs; never modify source experiments."""
import datetime
import json
import shutil
import torch
from screen_helpers import HERE, BASE, sha


def main():
    assert not (HERE / 'records/initialization_verification.json').exists(), 'Already initialized; do not overwrite'
    for name in ['records', 'logs', 'resume_inputs', 'origin_records/ESC', 'origin_records/WMSR']:
        (HERE / name).mkdir(parents=True, exist_ok=True)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    lineage = json.loads((HERE / 'resume_lineage.json').read_text())
    common = ['dataset', 'scale', 'seed', 'development_manifest_sha256', 'batch_size',
              'micro_batch', 'accumulation_steps', 'optimizer', 'loss', 'warmup_updates',
              'ema_decay', 'validation_every', 'save_every', 'augmentations', 'precision', 'initialization']
    assert protocol['models'] == ['ESC', 'WMSR'] and protocol['updates'] == 5000 and protocol['starting_updates'] == 1000
    split_path = HERE / 'development_splits/OISST.json'
    assert sha(split_path) == protocol['development_manifest_sha256']
    split = json.loads(split_path.read_text())
    source_manifest = BASE / 'records/formal/OISST_manifest.json'
    assert sha(source_manifest) == split['source_manifest_sha256']
    manifest = json.loads(source_manifest.read_text())
    assert sha(BASE / manifest['splits']['train']['file']) == split['source_train_npz_sha256']
    assert len(split['fit_indices']) == 720 and len(split['validation_indices']) == 80
    states, proofs = {}, {}
    for name, item in lineage.items():
        source = __import__('pathlib').Path(item['source_checkpoint'])
        assert sha(source) == item['source_checkpoint_sha256'], name
        state = torch.load(source, map_location='cpu', weights_only=False)
        states[name] = state
        assert state['step'] == 1000 and state['config']['model'] == name
        old = state['config']['protocol']
        assert old['updates'] == 1000 and old['warmup_updates'] == 100
        for key in common:
            assert old[key] == protocol[key], (name, key)
        assert old['scheduler'].startswith('Linear warmup100 then constantLR')
        assert state['config']['base_factory_sha256'] == sha(BASE / 'extended_models/factory.py')
        for key in ['model', 'ema']:
            assert all(torch.isfinite(t).all() for t in state[key].values())
        steps = sorted(set(float(s['step']) for s in state['optimizer']['state'].values() if 'step' in s))
        assert steps == [1000.]
        for group in state['optimizer']['param_groups']:
            assert group['lr'] == .0002 and group['betas'] == (.9, .99) and group['weight_decay'] == 0.
        dest = HERE / item['local_input']
        if dest.exists(): assert sha(dest) == sha(source)
        else: shutil.copy2(source, dest)
        origin = HERE / 'origin_records' / name
        origin.mkdir(parents=True, exist_ok=True)
        source_run = source.parent.parent
        assert json.loads((source_run / 'config.json').read_text()) == state['config']
        for filename in ['config.json', 'training_metrics.csv', 'validation_history.jsonl', 'pilot_results.json']:
            shutil.copy2(source_run / filename, origin / filename)
        run = HERE / 'pilot/results' / f'{name}_OISST_x4_s10'
        run.mkdir(parents=True, exist_ok=True)
        for filename in ['training_metrics.csv', 'validation_history.jsonl']:
            assert not (run / filename).exists()
            shutil.copy2(source_run / filename, run / filename)
        for f in source_run.glob('validation_*.json'): shutil.copy2(f, run / f.name)
        (run / 'progress.json').write_text(json.dumps({'model': name, 'status': 'pending_continuation', 'step': 1000, 'total_steps': 5000, 'final_test_images_evaluated': 0}, indent=2)+'\n')
        proofs[name] = {'source_checkpoint': str(source), 'source_sha256': sha(source),
                        'copied_input_sha256': sha(dest), 'source_config_sha256': sha(origin / 'config.json'),
                        'optimizer_steps': steps, 'step': state['step'], 'sampling_position': state['position'],
                        'all_raw_and_ema_tensors_finite': True}
    a, b = states['ESC'], states['WMSR']
    assert a['order'] == b['order'] and a['position'] == b['position']
    assert a['rng_augmentation'] == b['rng_augmentation']
    assert torch.equal(a['rng_loader'], b['rng_loader'])
    result = {'at': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status': 'pass',
              'models': proofs, 'equal_common_fields': common, 'same_sampler_and_augmentation_state': True,
              'source_updates': 1000, 'authorized_total_updates': 5000, 'test200_evaluated': False,
              'historical_validation_note': 'WMSR historical validation used batch2; independent1000-step batch1 recheck differs by5.6e-8dB. New validations use batch1 for both.'}
    (HERE / 'records/initialization_verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
