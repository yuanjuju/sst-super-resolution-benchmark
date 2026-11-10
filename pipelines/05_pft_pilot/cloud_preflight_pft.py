#!/usr/bin/env python3
"""Cloud-only PFT kernel/data/batch checks. Never evaluates SST test images."""
import ast
import collections
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import warnings

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get('WMSR_BASE_ROOT', '/root/wmsr-repro'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_namespace(torch):
    source = HERE / 'sources/PFT'
    namespace = {'torch': torch, 'math': math, 'warnings': warnings,
                 'collections': collections, 'repeat': itertools.repeat}
    helper = ast.parse((source / 'basicsr/archs/arch_util.py').read_text())
    wanted = {'_no_grad_trunc_normal_', 'trunc_normal_', '_ntuple'}
    nodes = [n for n in helper.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == 3
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<pinned PFT helpers>', 'exec'), namespace)
    output = {'__name__': 'pinned_pft_preflight', 'to_2tuple': namespace['_ntuple'](2),
              'trunc_normal_': namespace['trunc_normal_']}
    path = source / 'basicsr/archs/pft_arch.py'
    tree = ast.parse(path.read_text())
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom)
                 or n.module not in {'basicsr.archs.arch_util', 'basicsr.utils.registry'}]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'PFT':
            assert [ast.unparse(d) for d in node.decorator_list] == ['ARCH_REGISTRY.register()']
            node.decorator_list = []
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), output)
    return output


def load_smm(torch, atomic=False):
    """Keep pinned upstream bytes; any correction is compiled from a separate copy."""
    from torch.utils.cpp_extension import load
    sources = HERE / 'sources/PFT/ops_smm/src'
    name = 'smm_cuda'
    if atomic:
        patch = HERE / 'patches/smm_atomic/src'
        patch.mkdir(parents=True, exist_ok=True)
        original = (sources / 'smm_cuda.cu').read_bytes()
        corrected = original
        for target, replacement in [
            (b'grad_B[batch * C_dim * B_cols + row * B_cols + b_col] += a_val * g;',
             b'atomicAdd(&grad_B[batch * C_dim * B_cols + row * B_cols + b_col], a_val * g);'),
            (b'grad_B[batch * M * C_dim + b_row * C_dim + col] += a_val * g;',
             b'atomicAdd(&grad_B[batch * M * C_dim + b_row * C_dim + col], a_val * g);')]:
            assert corrected.count(target) == 1
            corrected = corrected.replace(target, replacement)
        # Upstream hardcodes the module name; allow side-by-side original/corrected audits.
        assert corrected.count(b'PYBIND11_MODULE(smm_cuda, m)') == 1
        corrected = corrected.replace(b'PYBIND11_MODULE(smm_cuda, m)', b'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)')
        for path, contents in [(patch / 'smm_cuda.cu', corrected), (patch / 'smm_cuda.h', (sources / 'smm_cuda.h').read_bytes())]:
            if not path.exists() or path.read_bytes() != contents: path.write_bytes(contents)
        import difflib
        (HERE / 'patches/smm_atomic/atomic_add.diff').write_text(''.join(difflib.unified_diff(
            original.decode().splitlines(True), corrected.decode().splitlines(True),
            fromfile='upstream/smm_cuda.cu', tofile='atomic_corrected/smm_cuda.cu')))
        (HERE / 'patches/smm_atomic/provenance.json').write_text(json.dumps({
            'upstream_sha256': hashlib.sha256(original).hexdigest(),
            'patched_sha256': hashlib.sha256(corrected).hexdigest(),
            'reason': 'Avoid lost gradient accumulation from concurrent non-atomic writes to grad_B',
            'changes': 'Two += accumulations replaced by atomicAdd, plus dynamic module-name binding for side-by-side audit; forward, topology, parameter initialization unchanged'
        }, indent=2) + '\n')
        sources = patch
        name = 'smm_cuda_atomic'
    build = HERE / '.cache' / name
    build.mkdir(parents=True, exist_ok=True)
    cuda_headers = [str(f) for f in (Path(torch.__file__).resolve().parent.parent / 'nvidia').glob('*/include') if f.is_dir()]
    extension = load(name=name, sources=[str(f) for f in sorted(sources.glob('*')) if f.suffix in {'.cu', '.cpp'}],
                     extra_cflags=['-O3'], extra_cuda_cflags=['-O3', '-lineinfo'],
                     extra_include_paths=[str(sources)] + cuda_headers, build_directory=str(build), verbose=True)
    sys.modules['smm_cuda'] = extension
    return extension


def kernel_checks(torch, namespace, require_pass=True, size=32, topk=8):
    torch.manual_seed(10)
    proof = []
    for case, kind in itertools.product(['topk_unique', 'with_duplicates'], ['QmK', 'AmV']):
        index = (torch.rand(2, size, size, device='cuda').topk(topk, dim=2).indices.int() if case == 'topk_unique'
                 else torch.randint(0, size, (2, size, topk), device='cuda', dtype=torch.int32))
        a = torch.randn((2, size, 40) if kind == 'QmK' else (2, size, topk), device='cuda', requires_grad=True)
        b = torch.randn((2, 40, size) if kind == 'QmK' else (2, size, 40), device='cuda', requires_grad=True)
        actual = namespace['SMM_' + kind].apply(a, b, index)
        if kind == 'QmK':
            expected = torch.bmm(a, b).gather(2, index.long())
        else:
            dense = torch.zeros((2, size, size), device='cuda').scatter_add(2, index.long(), a)
            expected = torch.bmm(dense, b)
        grad = torch.randn_like(actual)
        actual_grad = torch.autograd.grad(actual, (a, b), grad)
        expected_grad = torch.autograd.grad(expected, (a, b), grad)
        checks = []
        for label, one, two in [('forward', actual, expected), ('grad_A', actual_grad[0], expected_grad[0]), ('grad_B', actual_grad[1], expected_grad[1])]:
            checks.append({'quantity': label, 'max_abs_difference': (one - two).abs().max().item(),
                           'pass': bool(torch.allclose(one, two, rtol=5e-4, atol=1e-4))})
        proof.append({'operator': kind, 'case': case, 'size': size, 'topk': topk, 'checks': checks})
    if require_pass:
        assert all(q['pass'] for p in proof for q in p['checks']), proof
    return proof


def accumulated_l1(torch, model, low, high, micro_batch=2):
    """One effective batch, loss normalized over all images, optimizer called by caller once."""
    outputs = []
    total = 0.
    for start in range(0, len(low), micro_batch):
        prediction = model(low[start:start + micro_batch])
        target = high[start:start + micro_batch]
        assert prediction.shape == target.shape
        loss = torch.nn.functional.l1_loss(prediction, target) * len(target) / len(low)
        assert torch.isfinite(loss)
        loss.backward()
        total += loss.item()
        outputs.append(prediction.detach())
    return total, torch.cat(outputs)


def main():
    assert str(HERE).startswith('/root/wmsr-repro/'), 'Run only in the independent cloud project directory'
    records = HERE / 'records'
    records.mkdir(exist_ok=True)
    result = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status': 'checking', 'training_started': False, 'test_images_evaluated': 0}
    try:
        selection = json.loads((HERE / 'candidate_selection.json').read_text())
        manifest = json.loads((HERE / 'source_manifest.json').read_text())
        for item in manifest:
            if item['model'] == 'PFT':
                assert sha(HERE / 'sources/PFT' / item['path']) == item['sha256'], item['path']
        os.environ['TORCH_EXTENSIONS_DIR'] = str(HERE / '.cache/torch_extensions')
        import numpy as np
        import torch
        assert torch.cuda.is_available(), 'CUDA unavailable'
        major, minor = torch.cuda.get_device_capability(0)
        os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        audit = json.loads((HERE / 'records/kernel_audit.json').read_text())
        assert audit['corrected_all_pass'], 'Corrected CUDA gradients must pass independent audit before model checks'
        load_smm(torch, atomic=True)
        namespace = model_namespace(torch)
        result.update(gpu=torch.cuda.get_device_name(0), compute_capability=[major, minor],
                      torch=torch.__version__, cuda=torch.version.cuda, source_commit=selection['commit'],
                      precision={'dtype': 'float32', 'matmul_tf32': False, 'cudnn_tf32': False},
                      implementation_note='Pinned PFT network with separately recorded two-line atomic gradient accumulation correction',
                      kernel_checks=kernel_checks(torch, namespace), data={}, batches=[])
        for dataset, expected in selection['reuse_data_manifest_sha256'].items():
            f = BASE / 'records/formal' / f'{dataset}_manifest.json'
            assert sha(f) == expected
            data = json.loads(f.read_text())
            assert data['count'] == 1000 and data['splits']['train']['count'] == 800 and data['splits']['test']['count'] == 200
            for split in data['splits'].values():
                assert sha(BASE / split['file']) == split['sha256']
            result['data'][dataset] = 'Existing1000 HR /800 train /200 test manifests and NPZ hashes match'
        data = json.loads((BASE / 'records/formal/OISST_manifest.json').read_text())
        # Use only training images; never use the fixed200 test images for the preflight or selection.
        with np.load(BASE / data['splits']['train']['file']) as archive:
            for scale in [2, 3, 4]:
                random.seed(10); np.random.seed(10); torch.manual_seed(10); torch.cuda.manual_seed_all(10)
                config = dict(selection['official_network_and_training_configs'][str(scale)]['network_g'])
                config.pop('type')
                model = namespace['PFT'](**config).cuda().train()
                low = torch.from_numpy(((archive[f'lr_x{scale}_celsius'][:8] + 2) / 42).copy())[:, None].repeat(1, 3, 1, 1).cuda()
                high = torch.from_numpy(((archive['hr_celsius'][:8] + 2) / 42).copy())[:, None].repeat(1, 3, 1, 1).cuda()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); started = time.monotonic()
                loss, output = accumulated_l1(torch, model, low, high, micro_batch=2)
                assert output.shape == high.shape == (8, 3, 96, 96)
                gradients = [p.grad for p in model.parameters() if p.grad is not None]
                assert gradients and all(torch.isfinite(g).all() for g in gradients)
                torch.cuda.synchronize()
                row = {'scale': scale, 'batch': 8, 'LR': list(low.shape), 'HR': list(output.shape),
                       'parameters': sum(p.numel() for p in model.parameters()),
                       'micro_batch': 2, 'gradient_accumulation': 4,
                       'finite_training_loss': loss, 'finite_gradients': True,
                       'first_forward_backward_seconds': time.monotonic() - started,
                       'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                       'timing_boundary': 'First batch only, not a converged-model runtime benchmark or full optimizer-memory profile'}
                result['batches'].append(row)
                print(json.dumps(row), flush=True)
                del model, low, high, output, loss, gradients
                torch.cuda.empty_cache()
        result['status'] = 'pass'
    except Exception:
        result['status'] = 'fail'
        result['error'] = traceback.format_exc()
        raise
    finally:
        result['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (records / 'pft_preflight.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
