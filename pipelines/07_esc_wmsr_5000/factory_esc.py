"""Pinned standard ESC, native Naive attention; omit only BasicSR registry wiring."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def architecture_class():
    source = HERE / 'sources/ESC/esc/archs/esc_arch.py'
    selection = json.loads((HERE / 'selected_candidate.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == selection['architecture_sha256']
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or node.module != 'basicsr.utils.registry']
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'ESC':
            assert [ast.unparse(d) for d in node.decorator_list] == ['ARCH_REGISTRY.register()']
            node.decorator_list = []
    namespace = {'__name__': 'pinned_esc_standard'}
    exec(compile(ast.fix_missing_locations(tree), str(source), 'exec'), namespace)
    return namespace['ESC']


def create_model(scale):
    selection = json.loads((HERE / 'selected_candidate.json').read_text())
    assert scale in [2,3,4]
    kwargs = dict(selection['network_kwargs'][str(scale)])
    return architecture_class()(**kwargs)


def cache_rpe_indices(model,device):
    """Move unregistered, derived integer index tensors once; no weights/equations change."""
    for module in model.modules():
        if hasattr(module,'rpe_idxs'):module.rpe_idxs=module.rpe_idxs.to(device)
    return model
