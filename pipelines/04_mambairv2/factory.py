"""Pinned MambaIRv2-Light architecture, with unchanged upstream helper functions."""
import ast
import collections
import functools
import hashlib
import itertools
import json
import math
import warnings
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'sources/MambaIRv2'


@functools.lru_cache(maxsize=1)
def architecture_class():
    manifest = json.loads((SOURCE / 'source_manifest.json').read_text())
    for record in manifest['files']:
        path = SOURCE / record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], path
    helper_namespace = {'torch': torch, 'math': math, 'warnings': warnings,
                        'collections': collections, 'repeat': itertools.repeat}
    helper_path = SOURCE / 'basicsr/archs/arch_util.py'
    helpers = ast.parse(helper_path.read_text())
    wanted = {'_no_grad_trunc_normal_', 'trunc_normal_', '_ntuple'}
    nodes = [n for n in helpers.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert len(nodes) == 3
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper_path), 'exec'), helper_namespace)
    namespace = {'__name__': 'pinned_mambairv2_light',
                 'to_2tuple': helper_namespace['_ntuple'](2),
                 'trunc_normal_': helper_namespace['trunc_normal_']}
    path = SOURCE / 'basicsr/archs/mambairv2light_arch.py'
    tree = ast.parse(path.read_text())
    imports = {'basicsr.archs.arch_util', 'basicsr.utils.registry'}
    tree.body = [n for n in tree.body if not isinstance(n, ast.ImportFrom) or n.module not in imports]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'MambaIRv2Light':
            assert len(node.decorator_list) == 1
            assert ast.unparse(node.decorator_list[0]) == 'ARCH_REGISTRY.register()'
            node.decorator_list = []
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace['MambaIRv2Light']


class Range255(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image):
        return self.network(image / 255.0) * 255.0


def create_model(spec):
    assert spec['model'] == 'MambaIRv2Light'
    kwargs = dict(spec['network_kwargs'])
    assert kwargs['upscale'] == spec['scale']
    model = Range255(architecture_class()(**kwargs))
    files = {x['path']: x['sha256'] for x in json.loads((SOURCE / 'source_manifest.json').read_text())['files']}
    details = {
        'variant': 'official MambaIRv2-Light architecture and lightweight SR YAML',
        'repository': 'https://github.com/csguoh/MambaIR',
        'commit': '33d7b3460c4665229334cc8de38c7f4c766ed3be',
        'kwargs': kwargs, 'upstream_sha256': files,
        'compatibility_changes': 'Skip BasicSR registry-only decorator/import; execute exact pinned tuple/init helpers. No forward, ASSM, SGN, layer or parameter change.',
        'external_range': '0..255 -> model 0..1 -> output 0..255; no image quantization',
        'evaluation_rng': 'Upstream gumbel_softmax remains unchanged in eval; reset Torch CPU/CUDA to seed10 before full fixed-batch test.',
        'boundary': 'Controlled SST protocol comparison, not the original natural-image training recipe or an original-paper numeric reproduction.',
    }
    return model, details
