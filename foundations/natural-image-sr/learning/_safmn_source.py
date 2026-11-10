"""Load the local upstream network for a torch-only forward lesson.

BasicSR's package initializer imports its data, losses, metrics, and training
modules. This lesson only needs the original architecture and registry. Execute
their local source files separately, omitting the architecture's unused
torchvision.ops import. Network classes and forward computations are unchanged.
This helper is not the import route for a real BasicSR training run.
"""

import ast
import importlib.util
from pathlib import Path


def load_safmn(checkout: Path):
    architecture_path = checkout / "basicsr/archs/safmn_arch.py"
    registry_path = checkout / "basicsr/utils/registry.py"
    if not architecture_path.is_file() or not registry_path.is_file():
        raise FileNotFoundError(f"SAFMN checkout not found at {checkout}")

    spec = importlib.util.spec_from_file_location("lesson_registry", registry_path)
    registry_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(registry_module)

    tree = ast.parse(architecture_path.read_text(), filename=str(architecture_path))
    tree.body = [
        node for node in tree.body
        if not (
            isinstance(node, ast.ImportFrom)
            and node.module in {"torchvision", "basicsr.utils.registry"}
        )
    ]
    namespace = {
        "__name__": "safmn_forward_lesson",
        "__file__": str(architecture_path),
        "ARCH_REGISTRY": registry_module.ARCH_REGISTRY,
    }
    exec(compile(tree, str(architecture_path), "exec"), namespace)
    return namespace["SAFMN"]
