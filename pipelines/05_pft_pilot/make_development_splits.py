#!/usr/bin/env python3
"""Freeze development manifests within the original 800; final 800/200 stays intact."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent
FORMAL = HERE.parents[1] / '云端实验结果/records/formal'


def group_for(sample, dataset):
    if dataset != 'GHRSST': return sample['date']
    return str((math.floor(sum(sample['lat_bounds']) / 10), math.floor(sum(sample['lon_bounds']) / 10)))


def overlap(a, b):
    return all(max(a[k][0], b[k][0]) < min(a[k][1], b[k][1]) for k in ['lat_bounds', 'lon_bounds'])


def main():
    out = HERE / 'development_splits'
    out.mkdir(exist_ok=True)
    hashes = {}
    for dataset in ['OISST', 'HYCOM', 'GHRSST']:
        path = FORMAL / f'{dataset}_manifest.json'
        manifest = json.loads(path.read_text())
        samples = manifest['splits']['train']['samples']
        assert len(samples) == 800
        groups = defaultdict(list)
        for i, sample in enumerate(samples): groups[group_for(sample, dataset)].append(i)
        keys = sorted(groups)
        random.Random(10).shuffle(keys)
        reachable = {0: []}
        for key in keys:
            for count, picked in list(reachable.items()):
                new = count + len(groups[key])
                if new <= 80 and new not in reachable: reachable[new] = picked + [key]
            if 80 in reachable: break
        assert 80 in reachable
        val = sorted(i for key in reachable[80] for i in groups[key])
        train = [i for i in range(800) if i not in val]
        excluded = []
        if dataset == 'GHRSST':
            excluded = [i for i in train if any(overlap(samples[i], samples[j]) for j in val)]
            train = [i for i in train if i not in excluded]
            assert not any(overlap(samples[i], samples[j]) for i in train for j in val)
        assert set(train).isdisjoint(val) and set(train) | set(val) | set(excluded) == set(range(800))
        entry = {'dataset': dataset, 'seed': 10, 'source_manifest_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                 'source_train_npz_sha256': manifest['splits']['train']['sha256'],
                 'rule': 'Whole-date groups, seeded order and exact80 subset-sum' if dataset != 'GHRSST' else 'Whole5degree center blocks, seeded exact80 subset-sum; exclude any development-fit patch intersecting validation patch',
                 'fit_indices': train, 'validation_indices': val, 'excluded_for_overlap_indices': excluded,
                 'fit_ids': [samples[i]['sample_id'] for i in train], 'validation_ids': [samples[i]['sample_id'] for i in val],
                 'counts': {'fit': len(train), 'validation': len(val), 'excluded_for_overlap': len(excluded)},
                 'final_fit_count': 800, 'final_test_count': 200,
                 'boundary': 'Development uses only original training800. Fixed final test200 unchanged and not used in selection. Random-date validation is correlated ocean data, not independent temporal generalization. GHRSST is one date, so spatial development only.'}
        f = out / f'{dataset}.json'
        if f.exists(): assert json.loads(f.read_text()) == entry, 'Frozen split differs'
        else: f.write_text(json.dumps(entry, indent=2) + '\n')
        hashes[dataset] = {'file': str(f.relative_to(HERE)), 'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'counts': entry['counts']}
    (out / 'manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(hashes, indent=2))


if __name__ == '__main__': main()
